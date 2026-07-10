"""Generation with no TensorFlow, no music21, and no filesystem writes.

This is the path the deployed API takes. It reuses the parts of MAiSTRO that are
already pure Python and NumPy -- theory, sampling, conditioning, metrics -- and
swaps the two heavyweight pieces:

    Keras model  ->  npmodel.py   (NumPy forward pass, weights from a .npz)
    music21      ->  midi_writer  (mido)

Everything about *how* notes are chosen is shared with the local backend, so a
piece generated here is the piece you would get locally from the same seed. Only
the machinery underneath differs.

Serverless functions are stateless with a read-only filesystem, so `generate()`
returns MIDI bytes rather than writing a file, and there is no background job.
A 300-note piece takes ~10s, well inside Vercel's 300s ceiling.
"""

from __future__ import annotations

import functools
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import metrics, npmodel
from .conditioning import Conditioning, build_logit_bias, choose_seed_index, vocabulary_features
from .midi_writer import tokens_to_midi_bytes
from .sampling import sample_token
from .theory import MOODS

DEFAULT_TEMPERATURE = 1.5
DEFAULT_TOP_P = 0.98
DEFAULT_TOP_K = 40

# Serverless has a wall clock. 300 notes ~= 10s; the cap keeps a pathological
# request from eating the whole function budget.
MAX_NOTES = 600


@dataclass
class LiteConfig:
    """The subset of GenerationConfig the deployed model supports."""

    n_notes: int = 300
    temperature: float | None = None
    top_k: int = DEFAULT_TOP_K
    top_p: float | None = None
    key: str | None = None
    scale: str = "chromatic"
    mood: str | None = None
    tempo_bpm: int = 96
    seed: int | None = None

    def resolved(self) -> "LiteConfig":
        mood = MOODS.get(self.mood or "")
        return LiteConfig(
            n_notes=max(16, min(self.n_notes, MAX_NOTES)),
            temperature=self.temperature
            if self.temperature is not None
            else (mood.temperature if mood else DEFAULT_TEMPERATURE),
            top_k=self.top_k,
            top_p=self.top_p if self.top_p is not None else (mood.top_p if mood else DEFAULT_TOP_P),
            key=self.key,
            scale=self.scale
            if self.scale != "chromatic"
            else (mood.suggested_scale if mood else "chromatic"),
            mood=self.mood,
            tempo_bpm=self.tempo_bpm,
            seed=self.seed,
        )

    def to_conditioning(self) -> Conditioning:
        mood = MOODS.get(self.mood or "")
        return Conditioning(
            key=self.key,
            scale=self.scale,
            mood=self.mood,
            register_center=mood.register_center if mood else None,
            register_width=mood.register_width if mood else 12.0,
            duration_pref=mood.duration_pref if mood else 0.0,
            rest_bias=mood.rest_bias if mood else 0.0,
        )


@dataclass
class LiteResult:
    midi: bytes
    tokens: list[str]
    metrics: dict
    config: dict


@dataclass
class Bundle:
    """The deployed model: weights, vocabulary, and the corpus to seed from."""

    weights: npmodel.TransformerWeights
    pitchnames: list[str]
    corpus: np.ndarray  # uint16 token ids
    corpus_histogram: np.ndarray


@functools.lru_cache(maxsize=1)
def load_bundle(model_dir: str) -> Bundle:
    """Load once per warm function instance; a cold start pays ~200ms for this."""
    directory = Path(model_dir)
    vocabulary = np.load(directory / "vocabulary.npz", allow_pickle=True)

    return Bundle(
        weights=npmodel.load(directory / "transformer.npz"),
        pitchnames=[str(token) for token in vocabulary["pitchnames"]],
        corpus=vocabulary["corpus"],
        corpus_histogram=vocabulary["corpus_histogram"],
    )


def _seed_windows(bundle: Bundle) -> np.ndarray:
    """Every sliding window the corpus offers, as a strided view -- no copy."""
    length = bundle.weights.sequence_length
    count = len(bundle.corpus) - length
    stride = bundle.corpus.strides[0]
    return np.lib.stride_tricks.as_strided(
        bundle.corpus, shape=(count, length), strides=(stride, stride)
    )


def generate_tokens(bundle: Bundle, cfg: LiteConfig) -> list[str]:
    cfg = cfg.resolved()
    features = vocabulary_features(bundle.pitchnames)
    conditioning = cfg.to_conditioning()
    logit_bias = build_logit_bias(features, conditioning)

    rng = np.random.default_rng(cfg.seed)
    windows = _seed_windows(bundle)
    start = choose_seed_index(windows, features, conditioning, rng)
    pattern = windows[start].astype(np.int32).copy()

    generated: list[str] = []
    for _ in range(cfg.n_notes):
        probabilities = npmodel.predict_next(bundle.weights, pattern)
        index = sample_token(
            probabilities,
            temperature=cfg.temperature or 0.0,
            top_k=cfg.top_k,
            top_p=cfg.top_p or 0.0,
            logit_bias=logit_bias,
            rng=rng,
        )
        generated.append(bundle.pitchnames[index])
        pattern = np.concatenate([pattern[1:], [index]])

    return generated


def generate(bundle: Bundle, cfg: LiteConfig | None = None) -> LiteResult:
    """Generate one piece and return its MIDI bytes, metrics and reproducible config."""
    cfg = (cfg or LiteConfig()).resolved()
    if cfg.seed is None:
        cfg.seed = int(np.random.SeedSequence().entropy % (2**32))

    tokens = generate_tokens(bundle, cfg)
    scores = metrics.evaluate(tokens)

    # metrics.evaluate reads the corpus histogram off disk, which the deployed
    # bundle does not have. Recompute the one entry that needs it.
    scores["pitch_class_kl"] = metrics.kl_divergence(
        metrics.pitch_class_histogram(tokens), bundle.corpus_histogram
    )

    return LiteResult(
        midi=tokens_to_midi_bytes(tokens, tempo_bpm=cfg.tempo_bpm),
        tokens=tokens,
        metrics=scores,
        config=asdict(cfg),
    )
