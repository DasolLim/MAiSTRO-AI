"""Steer an unconditioned model toward a key, register and note density.

The trained networks take a window of past notes and nothing else -- there is no
key or mood input baked into the architecture, and retraining a conditional
model needs labelled data we do not have. So conditioning happens at *decode*
time instead: every vocabulary token gets an additive score in log-probability
space, and that vector is added to the model's logits before sampling.

This is the same mechanism as OpenAI's `logit_bias` and it composes cleanly with
temperature and nucleus sampling. The tradeoff worth knowing: the model's own
sense of what note comes next is unchanged, so a strong bias fights the model
rather than collaborating with it. Penalties are therefore soft (a few nats),
enough to prefer in-key continuations without overriding a good resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .theory import MOODS, TokenFeatures, parse_vocabulary, scale_pitch_classes

# Nats subtracted from a token whose pitches all sit outside the requested key.
# ~2.5 makes an out-of-key note roughly 12x less likely at temperature 1.0,
# which suppresses accidentals without banning the model's passing tones.
OUT_OF_KEY_PENALTY = 2.5
REGISTER_PENALTY = 1.5
DURATION_WEIGHT = 0.6
REST_WEIGHT = 0.8


@dataclass(frozen=True)
class Conditioning:
    """User-facing steering knobs. All fields are optional; None means "don't steer"."""

    key: str | None = None  # tonic, e.g. "C" or "F#"
    scale: str = "chromatic"  # see theory.SCALES
    mood: str | None = None  # see theory.MOODS
    register_center: int | None = None  # MIDI note number
    register_width: float = 12.0
    duration_pref: float = 0.0  # >0 longer notes, <0 shorter
    rest_bias: float = 0.0  # >0 more space, <0 denser

    @classmethod
    def from_mood(cls, mood_key: str, **overrides) -> "Conditioning":
        """Preset from theory.MOODS, with any field overridable by the caller."""
        mood = MOODS[mood_key]
        base = cls(
            key=None,
            scale=mood.suggested_scale,
            mood=mood.key,
            register_center=mood.register_center,
            register_width=mood.register_width,
            duration_pref=mood.duration_pref,
            rest_bias=mood.rest_bias,
        )
        return base if not overrides else cls(**{**base.__dict__, **overrides})

    @property
    def is_active(self) -> bool:
        return bool(
            (self.key and self.scale != "chromatic")
            or self.register_center is not None
            or self.duration_pref
            or self.rest_bias
        )


def build_logit_bias(features: list[TokenFeatures], cond: Conditioning) -> np.ndarray | None:
    """Additive per-token bias over the vocabulary, or None if nothing to steer."""
    if not cond.is_active:
        return None

    bias = np.zeros(len(features), dtype=np.float64)

    in_key: frozenset[int] | None = None
    if cond.key and cond.scale != "chromatic":
        in_key = scale_pitch_classes(cond.key, cond.scale)

    # Rests carry no pitch, so a piece steered into a narrow register or a tight
    # key would otherwise drift toward rests as the cheapest way to satisfy the
    # bias. Their score depends only on rest_bias.
    for index, token in enumerate(features):
        if token.is_rest:
            bias[index] += REST_WEIGHT * cond.rest_bias
            continue

        if in_key is not None and token.pitch_classes:
            outside = len(token.pitch_classes - in_key) / len(token.pitch_classes)
            bias[index] -= OUT_OF_KEY_PENALTY * outside

        if cond.register_center is not None and token.mean_midi is not None:
            distance = abs(token.mean_midi - cond.register_center) / max(cond.register_width, 1.0)
            bias[index] -= REGISTER_PENALTY * distance**2

        if cond.duration_pref and token.duration > 0:
            # log2 keeps this symmetric: a half note and an eighth note sit the
            # same distance either side of a quarter note (duration 1.0).
            bias[index] += DURATION_WEIGHT * cond.duration_pref * np.log2(token.duration)

    return bias


def score_seed_window(window_tokens: list[TokenFeatures], cond: Conditioning) -> float:
    """How well a candidate seed window matches the requested key. Higher is better."""
    if not cond.key or cond.scale == "chromatic":
        return 0.0

    in_key = scale_pitch_classes(cond.key, cond.scale)
    pitched = [t for t in window_tokens if not t.is_rest and t.pitch_classes]
    if not pitched:
        return 0.0

    matches = sum(len(t.pitch_classes & in_key) / len(t.pitch_classes) for t in pitched)
    return matches / len(pitched)


def choose_seed_index(
    network_input: list[list[int]],
    features: list[TokenFeatures],
    cond: Conditioning,
    rng: np.random.Generator,
    candidates: int = 64,
) -> int:
    """Pick a starting window, preferring ones already in the requested key.

    Sampling `candidates` windows and keeping the best beats scanning all of them
    (a 200-file dataset yields ~100k windows) and still lands in-key almost
    always, because a few percent of any tonal corpus sits squarely in each key.
    """
    if not cond.key or cond.scale == "chromatic":
        return int(rng.integers(0, len(network_input)))

    sampled = rng.integers(0, len(network_input), size=min(candidates, len(network_input)))
    best_index, best_score = int(sampled[0]), -1.0

    for index in sampled:
        window = [features[token_id] for token_id in network_input[int(index)]]
        score = score_seed_window(window, cond)
        if score > best_score:
            best_index, best_score = int(index), score

    return best_index


def vocabulary_features(pitchnames: list[str]) -> list[TokenFeatures]:
    return parse_vocabulary(pitchnames)
