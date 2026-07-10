"""The transformer's forward pass, in NumPy alone.

TensorFlow is 877MB unpacked and needs Python <=3.10. Vercel's Python runtime is
3.12 with a 500MB bundle ceiling, so a TF-backed API cannot be deployed there at
any size. But TF is only needed to *train*: inference is a dozen matmuls, and the
weights are 3.9M numbers.

This module reproduces `architectures.build_transformer` exactly, reading weights
from the `.npz` that `export.py` writes. Verified against Keras to ~1e-9 absolute
error on the output distribution; storing the weights in float16 moves it by ~3e-7.
The whole serving stack becomes numpy + fastapi + mido, about 31MB.

Two details make the match exact rather than approximate:

* The network trains with `gelu_tanh` (see architectures.py), whose closed form is
  four lines of NumPy. Keras's default "gelu" uses erf and would drag in SciPy.
* Keras `MultiHeadAttention` keeps per-head projections, so the weights are
  (d_model, num_heads, key_dim) rather than one fused matrix. `einsum` follows
  that layout directly instead of reshaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Matches keras/tf.nn.gelu(approximate=True).
_GELU_COEFF = np.sqrt(2.0 / np.pi)
_LAYER_NORM_EPS = 1e-6


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(_GELU_COEFF * (x + 0.044715 * x**3)))


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    variance = x.var(-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + _LAYER_NORM_EPS) * gamma + beta


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def causal_self_attention(x: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
    """x: (T, d_model). weights: Keras MultiHeadAttention's 8 arrays, in its order."""
    wq, bq, wk, bk, wv, bv, wo, bo = weights
    seq_len = x.shape[0]
    key_dim = wq.shape[2]

    query = np.einsum("td,dhk->thk", x, wq) + bq
    key = np.einsum("td,dhk->thk", x, wk) + bk
    value = np.einsum("td,dhk->thk", x, wv) + bv

    scores = np.einsum("thk,shk->hts", query, key) / np.sqrt(key_dim)
    # Step t may not attend to steps after it. This is what makes it a language model.
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(future[None, :, :], -np.inf, scores)

    context = np.einsum("hts,shk->thk", softmax(scores, axis=-1), value)
    return np.einsum("thk,hkd->td", context, wo) + bo


@dataclass
class TransformerWeights:
    """Everything npmodel needs, as plain arrays. Loaded from export.py's .npz."""

    token_embedding: np.ndarray  # (n_vocab, d_model)
    position_embedding: np.ndarray  # (sequence_length, d_model)
    blocks: list[dict]
    final_norm: tuple[np.ndarray, np.ndarray]
    head: tuple[np.ndarray, np.ndarray]  # (d_model, n_vocab), (n_vocab,)
    num_heads: int

    @property
    def n_vocab(self) -> int:
        return self.token_embedding.shape[0]

    @property
    def sequence_length(self) -> int:
        return self.position_embedding.shape[0]


def predict_next(weights: TransformerWeights, tokens: np.ndarray) -> np.ndarray:
    """tokens: (T,) int ids. Returns (n_vocab,) probabilities for the next token."""
    x = weights.token_embedding[tokens] + weights.position_embedding[: len(tokens)]

    for block in weights.blocks:
        normed = layer_norm(x, block["ln1_gamma"], block["ln1_beta"])
        x = x + causal_self_attention(normed, block["attention"])

        normed = layer_norm(x, block["ln2_gamma"], block["ln2_beta"])
        hidden = gelu(normed @ block["ff1_kernel"] + block["ff1_bias"])
        x = x + (hidden @ block["ff2_kernel"] + block["ff2_bias"])

    x = layer_norm(x, *weights.final_norm)
    # Only the last position predicts the next token, matching how it was trained.
    logits = x[-1] @ weights.head[0] + weights.head[1]
    return softmax(logits)


def load(path: Path | str) -> TransformerWeights:
    """Read a `.npz` produced by export.py, upcasting float16 storage to float32."""
    data = np.load(str(path))

    def get(name: str) -> np.ndarray:
        return data[name].astype(np.float32)

    num_layers = int(data["num_layers"])
    blocks = []
    for i in range(num_layers):
        blocks.append(
            {
                "attention": [get(f"block{i}.attention.{j}") for j in range(8)],
                "ln1_gamma": get(f"block{i}.ln1_gamma"),
                "ln1_beta": get(f"block{i}.ln1_beta"),
                "ln2_gamma": get(f"block{i}.ln2_gamma"),
                "ln2_beta": get(f"block{i}.ln2_beta"),
                "ff1_kernel": get(f"block{i}.ff1_kernel"),
                "ff1_bias": get(f"block{i}.ff1_bias"),
                "ff2_kernel": get(f"block{i}.ff2_kernel"),
                "ff2_bias": get(f"block{i}.ff2_bias"),
            }
        )

    return TransformerWeights(
        token_embedding=get("token_embedding"),
        position_embedding=get("position_embedding"),
        blocks=blocks,
        final_norm=(get("final_ln_gamma"), get("final_ln_beta")),
        head=(get("head_kernel"), get("head_bias")),
        num_heads=int(data["num_heads"]),
    )
