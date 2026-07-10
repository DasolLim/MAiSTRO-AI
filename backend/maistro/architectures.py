"""The model zoo: three architectures trained on the same note vocabulary.

Having more than one architecture is the point. `lstm` is the textbook baseline,
`lstm_attention` is what MAiSTRO shipped with, and `transformer` is the modern
answer to the same problem. They share a dataset, a tokenizer and a decoder, so
the arena (see arena.py) compares architectures rather than pipelines.

Two things differ per architecture and are declared in the registry below:

* `encoding` -- how a window of token ids reaches the network. The LSTMs read a
  single normalised float per step, which is what the original implementation
  did; it wastes the vocabulary's structure (token 41 is not "between" 40 and
  42 in any musical sense) but the weights already on disk depend on it. The
  transformer uses a learned embedding per token instead, which is why it needs
  the same tokens fed as raw integer ids.
* `sequence_length` is shared, but the transformer sees the whole window at once
  through causal self-attention rather than folding it through a recurrent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras_self_attention import SeqSelfAttention

from . import config, weights_compat

Encoding = Literal["scalar", "tokens"]


def build_lstm(sequence_length: int, n_vocab: int) -> Model:
    """Two stacked LSTMs, no attention. The baseline everyone starts from."""
    model = Sequential(name="lstm")
    model.add(layers.LSTM(512, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(512))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(n_vocab))
    model.add(layers.Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model


def build_lstm_attention(sequence_length: int, n_vocab: int) -> Model:
    """Bidirectional LSTM + self-attention. Matches the checkpoints already on disk."""
    model = Sequential(name="lstm_attention")
    model.add(
        layers.Bidirectional(
            layers.LSTM(512, return_sequences=True), input_shape=(sequence_length, 1)
        )
    )
    model.add(SeqSelfAttention(attention_activation="sigmoid"))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(n_vocab))
    model.add(layers.Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model


class TokenAndPositionEmbedding(layers.Layer):
    """Learned token embedding plus a learned absolute position embedding."""

    def __init__(self, sequence_length: int, n_vocab: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.token_embedding = layers.Embedding(n_vocab, d_model)
        self.position_embedding = layers.Embedding(sequence_length, d_model)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[-1], delta=1)
        return self.token_embedding(inputs) + self.position_embedding(positions)

    def get_config(self):
        return {
            **super().get_config(),
            "sequence_length": self.sequence_length,
            "n_vocab": self.n_vocab,
            "d_model": self.d_model,
        }


class TransformerBlock(layers.Layer):
    """Pre-norm decoder block: causal self-attention, then a position-wise FFN."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout
        )
        self.ffn = Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dense(d_model)]
        )
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, inputs, training=None):
        normed = self.norm1(inputs)
        # Causal masking is what makes this a language model rather than an
        # autoencoder: step t may not attend to steps > t.
        attended = self.attention(normed, normed, use_causal_mask=True, training=training)
        residual = inputs + self.drop1(attended, training=training)

        normed = self.norm2(residual)
        return residual + self.drop2(self.ffn(normed, training=training), training=training)

    def get_config(self):
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout,
        }


def build_transformer(
    sequence_length: int,
    n_vocab: int,
    d_model: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    ff_dim: int = 512,
    dropout: float = 0.1,
) -> Model:
    """A small GPT-style decoder over the same note vocabulary."""
    inputs = layers.Input(shape=(sequence_length,), dtype="int32", name="tokens")
    x = TokenAndPositionEmbedding(sequence_length, n_vocab, d_model)(inputs)
    x = layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Only the final position predicts the next token, matching how the LSTMs are
    # trained (one target per window) so the arena compares like with like.
    x = layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)
    outputs = layers.Dense(n_vocab, activation="softmax", name="next_token")(x)

    model = Model(inputs, outputs, name="transformer")
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=1e-3))
    return model


@dataclass(frozen=True)
class Architecture:
    key: str
    label: str
    description: str
    encoding: Encoding
    build: Callable[[int, int], Model]


ARCHITECTURES: dict[str, Architecture] = {
    "lstm": Architecture(
        key="lstm",
        label="LSTM",
        description=(
            "Two stacked LSTM layers. Cheapest to train; the recurrent state has to "
            "carry the whole 100-note context, so long-range structure fades."
        ),
        encoding="scalar",
        build=build_lstm,
    ),
    "lstm_attention": Architecture(
        key="lstm_attention",
        label="LSTM + Attention",
        description=(
            "Bidirectional LSTM with a self-attention layer over the recurrent states. "
            "Attention lets the decoder look back at any step in the window directly."
        ),
        encoding="scalar",
        build=build_lstm_attention,
    ),
    "transformer": Architecture(
        key="transformer",
        label="Transformer",
        description=(
            "4-layer causal decoder with learned token + position embeddings. No "
            "recurrence, so the whole window is one parallel attention op; trains "
            "several times faster per epoch and models repeats and returns better."
        ),
        encoding="tokens",
        build=build_transformer,
    ),
}

DEFAULT_ARCHITECTURE = "lstm_attention"


def get(arch: str) -> Architecture:
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; expected one of {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[arch]


def encode_windows(windows: np.ndarray | list[list[int]], n_vocab: int, encoding: Encoding):
    """Shape a batch of token-id windows into the tensor its architecture expects."""
    array = np.asarray(windows, dtype=np.int32)
    if encoding == "tokens":
        return array
    # "scalar": one normalised float per step, shaped (batch, sequence_length, 1).
    return array.reshape(array.shape[0], array.shape[1], 1).astype(np.float32) / float(n_vocab)


def checkpoint_dir(arch: str) -> Path:
    return config.CHECKPOINTS_DIR / arch


def build_network(arch: str, sequence_length: int, n_vocab: int) -> Model:
    return get(arch).build(sequence_length, n_vocab)


def load_trained_network(arch: str, n_vocab: int, weights_path: Path | None = None) -> Model:
    """Build `arch` and load its most recent checkpoint."""
    weights_path = weights_path or config.latest_weights_file(arch)
    model = build_network(arch, config.SEQUENCE_LENGTH, n_vocab)
    print(f"Loading {arch} weights from: {weights_path}")
    weights_compat.load_weights(model, weights_path)
    return model


def trained_architectures() -> list[str]:
    """Architectures that have at least one checkpoint on disk."""
    available = []
    for arch in ARCHITECTURES:
        try:
            config.latest_weights_file(arch)
        except FileNotFoundError:
            continue
        available.append(arch)
    return available
