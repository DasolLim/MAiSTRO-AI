"""Token sampling strategies for autoregressive generation.

The original generate.py picked `argmax(prediction)` at every step. With a
deterministic decoder the model can only ever walk one path out of a given
context, so any repeated context produces the same continuation forever --
which is why generated pieces fell into short loops. Sampling from the
distribution instead of collapsing it fixes that, and exposes the
exploration/coherence tradeoff as a single "creativity" dial (temperature).

The trained networks end in a softmax, so `model.predict()` hands back
probabilities rather than logits. `log(p)` recovers the logits up to an
additive constant, which softmax is invariant to -- so temperature scaling
and additive logit biases are applied in log-probability space here.

## Calibrating temperature for *this* model

The usual text-LM defaults (T=0.8, top_p=0.9) do nothing here. Measured over 25
real next-token distributions from the trained lstm_attention checkpoint, the
model's output is far sharper than a language model's:

    mean max probability  0.95        mean entropy  0.21 nats

The argmax token already holds more mass than any sane nucleus threshold, so
top_p <= 0.95 truncates to a single token and silently reproduces greedy
decoding. Sweeping temperature over those same distributions (vocabulary 3388):

    T     max prob   perplexity   nucleus@0.95
    1.0     0.96          1.1          1
    1.5     0.94          1.4         12
    1.8     0.87          3.7        335
    2.0     0.77         12.7       1713
    2.5     0.41        239.0       2899

Below ~1.3 the decoder is effectively greedy; above ~2.1 it is sampling from a
flat tail of 2000+ tokens, which is noise. The usable band is 1.4-2.0, and that
is the range the UI's creativity slider spans. `top_k` guards the tail so the
upper end stays musical.
"""

from __future__ import annotations

import numpy as np

# log(0) is -inf; clip first so probabilities that rounded to zero become a
# large-but-finite negative logit that sampling can still (never) pick.
_MIN_PROB = 1e-9


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    return np.log(np.clip(probs.astype(np.float64), _MIN_PROB, 1.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def apply_top_k(logits: np.ndarray, top_k: int) -> np.ndarray:
    """Mask out everything except the `top_k` highest-scoring tokens."""
    if top_k <= 0 or top_k >= logits.size:
        return logits
    cutoff = np.partition(logits, -top_k)[-top_k]
    return np.where(logits < cutoff, -np.inf, logits)


def apply_top_p(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Nucleus filter: keep the smallest set of tokens whose mass reaches `top_p`."""
    if not 0.0 < top_p < 1.0:
        return logits

    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])

    # `searchsorted` gives the first index where the mass reaches top_p; +1 keeps
    # that token, so the nucleus is never empty even if one token dominates.
    keep_count = int(np.searchsorted(cumulative, top_p) + 1)
    keep = order[:keep_count]

    filtered = np.full_like(logits, -np.inf)
    filtered[keep] = logits[keep]
    return filtered


def sample_token(
    probs: np.ndarray,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    logit_bias: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Pick the next token index from a softmax output.

    temperature: 0 is greedy (the old argmax behaviour), <1 sharpens toward the
        model's favourite notes, >1 flattens the distribution toward surprise.
    top_k / top_p: truncate the tail before sampling so temperature can be raised
        without letting genuinely wrong notes through. Applied after the bias.
    logit_bias: per-token additive score in log space, used to steer generation
        toward a key, register or note density (see conditioning.py).
    """
    rng = rng or np.random.default_rng()
    logits = probs_to_logits(np.asarray(probs).ravel())

    if logit_bias is not None:
        logits = logits + logit_bias

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / float(temperature)
    logits = apply_top_k(logits, top_k)
    logits = apply_top_p(logits, top_p)

    return int(rng.choice(logits.size, p=softmax(logits)))
