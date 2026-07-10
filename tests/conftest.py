"""Shared fixtures.

The point of most of these tests is that MAiSTRO's serving path needs nothing but
numpy: no TensorFlow, no music21. So the fixtures build a tiny transformer by hand
rather than training one, and the tests import only what the deployed function does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Small enough to be instant, large enough to exercise every code path.
N_VOCAB = 24
SEQ_LEN = 8
D_MODEL = 8
NUM_HEADS = 2
KEY_DIM = D_MODEL // NUM_HEADS
FF_DIM = 16
NUM_LAYERS = 2

# Tokens in the exact format dataset.extract_notes writes.
TOKENS = [
    "C4 0.5", "D4 0.5", "E4 1.0", "F4 0.25", "G4 1.0", "A4 0.5", "B4 0.25", "C5 1.0",
    "rest 0.5", "C4.E4.G4 1.0", "D4.F4 0.5", "A3 1.0", "B-3 0.5", "F#4 0.25", "G#4 0.5",
    "C3 2.0", "E3 1.0", "G3 0.5", "rest 1.0", "A5 0.25", "D5 0.5", "E5 1.0", "F5 0.5", "G5 0.25",
]
assert len(TOKENS) == N_VOCAB


def _attention_weights(rng: np.random.Generator) -> list[np.ndarray]:
    """Keras MultiHeadAttention's 8 arrays, in the order it stores them."""
    return [
        rng.normal(size=(D_MODEL, NUM_HEADS, KEY_DIM), scale=0.1),  # query kernel
        np.zeros((NUM_HEADS, KEY_DIM)),                             # query bias
        rng.normal(size=(D_MODEL, NUM_HEADS, KEY_DIM), scale=0.1),  # key kernel
        np.zeros((NUM_HEADS, KEY_DIM)),                             # key bias
        rng.normal(size=(D_MODEL, NUM_HEADS, KEY_DIM), scale=0.1),  # value kernel
        np.zeros((NUM_HEADS, KEY_DIM)),                             # value bias
        rng.normal(size=(NUM_HEADS, KEY_DIM, D_MODEL), scale=0.1),  # output kernel
        np.zeros((D_MODEL,)),                                       # output bias
    ]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    """A directory holding transformer.npz + vocabulary.npz, as export.py would write."""
    directory = tmp_path_factory.mktemp("model")
    rng = np.random.default_rng(0)

    arrays: dict[str, np.ndarray] = {
        "token_embedding": rng.normal(size=(N_VOCAB, D_MODEL), scale=0.1),
        "position_embedding": rng.normal(size=(SEQ_LEN, D_MODEL), scale=0.1),
        "num_layers": np.asarray(NUM_LAYERS),
        "num_heads": np.asarray(NUM_HEADS),
        "final_ln_gamma": np.ones(D_MODEL),
        "final_ln_beta": np.zeros(D_MODEL),
        "head_kernel": rng.normal(size=(D_MODEL, N_VOCAB), scale=0.1),
        "head_bias": np.zeros(N_VOCAB),
    }
    for i in range(NUM_LAYERS):
        for j, weight in enumerate(_attention_weights(rng)):
            arrays[f"block{i}.attention.{j}"] = weight
        arrays[f"block{i}.ln1_gamma"] = np.ones(D_MODEL)
        arrays[f"block{i}.ln1_beta"] = np.zeros(D_MODEL)
        arrays[f"block{i}.ln2_gamma"] = np.ones(D_MODEL)
        arrays[f"block{i}.ln2_beta"] = np.zeros(D_MODEL)
        arrays[f"block{i}.ff1_kernel"] = rng.normal(size=(D_MODEL, FF_DIM), scale=0.1)
        arrays[f"block{i}.ff1_bias"] = np.zeros(FF_DIM)
        arrays[f"block{i}.ff2_kernel"] = rng.normal(size=(FF_DIM, D_MODEL), scale=0.1)
        arrays[f"block{i}.ff2_bias"] = np.zeros(D_MODEL)

    half = {k: (v if v.dtype.kind in "iu" else v.astype(np.float16)) for k, v in arrays.items()}
    np.savez_compressed(directory / "transformer.npz", **half)

    corpus = np.arange(200, dtype=np.uint16) % N_VOCAB
    histogram = np.full(12, 1 / 12, dtype=np.float32)
    np.savez_compressed(
        directory / "vocabulary.npz",
        pitchnames=np.asarray(TOKENS, dtype=object),
        corpus=corpus,
        corpus_histogram=histogram,
    )
    return directory
