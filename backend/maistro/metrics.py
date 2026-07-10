"""Objective quality metrics for a generated note sequence.

Listening tests are the ground truth for music, which is what arena.py is for.
But human votes are slow and noisy, so these cheap statistics catch the two
failure modes that dominate this model class long before a human hears them:

* **Repetition.** A degenerate decoder emits the same 4-note figure forever.
  `repetition_rate` measures exactly that, and it is the metric that moves when
  greedy decoding is replaced with temperature sampling.
* **Distribution drift.** A sequence can be perfectly varied and still not sound
  like the training corpus. `pitch_class_kl` compares the generated pitch-class
  histogram against the dataset's, in nats -- near 0 means the model learned
  the corpus's tonal palette rather than a uniform smear across the keyboard.

Neither is a substitute for taste. Both are reproducible, which the votes are not.
"""

from __future__ import annotations

import pickle
from collections import Counter
from functools import lru_cache

import numpy as np

from . import config
from .theory import parse_token

NGRAM_SIZE = 4
_EPSILON = 1e-9


def pitch_class_histogram(tokens: list[str]) -> np.ndarray:
    """Normalised 12-bin histogram of pitch classes across all notes and chords."""
    counts = np.zeros(12, dtype=np.float64)
    for token in tokens:
        features = parse_token(token)
        for pc in features.pitch_classes:
            counts[pc] += 1

    total = counts.sum()
    return counts / total if total else counts


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats, with both distributions smoothed to avoid infinities."""
    p = (p + _EPSILON) / (p + _EPSILON).sum()
    q = (q + _EPSILON) / (q + _EPSILON).sum()
    return float(np.sum(p * np.log(p / q)))


def repetition_rate(tokens: list[str], n: int = NGRAM_SIZE) -> float:
    """Fraction of n-grams that are not the first occurrence of that n-gram.

    0.0 means every window of n notes is new; values above ~0.5 mean the decoder
    is looping. Greedy (argmax) decoding on this model typically scores >0.9.
    """
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


@lru_cache(maxsize=1)
def dataset_pitch_class_histogram() -> tuple[float, ...] | None:
    """Reference histogram from the prepared training notes, or None if absent."""
    if not config.NOTES_FILE.exists():
        return None
    with open(config.NOTES_FILE, "rb") as fh:
        notes = pickle.load(fh)
    return tuple(pitch_class_histogram(notes))


def evaluate(tokens: list[str]) -> dict:
    """Summary statistics for one generated sequence. All values are JSON-safe."""
    if not tokens:
        return {}

    features = [parse_token(token) for token in tokens]
    pitched = [f for f in features if not f.is_rest and f.mean_midi is not None]
    midis = [f.mean_midi for f in pitched]
    total_duration = sum(f.duration for f in features)

    result: dict = {
        "note_count": len(tokens),
        "unique_token_ratio": len(set(tokens)) / len(tokens),
        "repetition_rate": repetition_rate(tokens),
        "rest_fraction": sum(1 for f in features if f.is_rest) / len(features),
        "distinct_pitch_classes": len({pc for f in features for pc in f.pitch_classes}),
        "mean_pitch": float(np.mean(midis)) if midis else None,
        "pitch_range": int(max(midis) - min(midis)) if midis else None,
        "notes_per_quarter": len(pitched) / total_duration if total_duration else None,
    }

    reference = dataset_pitch_class_histogram()
    if reference is not None:
        result["pitch_class_kl"] = kl_divergence(
            pitch_class_histogram(tokens), np.asarray(reference)
        )

    return result


def most_common_tokens(tokens: list[str], limit: int = 5) -> list[tuple[str, int]]:
    """The decoder's favourite tokens — a fast eyeball check for mode collapse."""
    return Counter(tokens).most_common(limit)
