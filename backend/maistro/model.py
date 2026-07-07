"""Shared LSTM + self-attention architecture used for both training and inference.

Previously this network definition was copy-pasted across generate.py, collaborate.py
and collaborateTTE.py. Centralizing it here also fixes a bug where weight loading only
looked for legacy `.h5` files while training actually saves `.keras` checkpoints.
"""

from pathlib import Path

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, Bidirectional, Flatten
from keras_self_attention import SeqSelfAttention

from . import config


def build_network(input_shape: tuple[int, int], n_vocab: int) -> Sequential:
    """Build the (uncompiled-weights) LSTM + attention architecture.

    input_shape: (sequence_length, features)
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(512, return_sequences=True), input_shape=input_shape))
    model.add(SeqSelfAttention(attention_activation="sigmoid"))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(n_vocab))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    return model


def load_trained_network(n_vocab: int, weights_path: Path | None = None) -> Sequential:
    """Build the network and load the most recently trained weights."""
    weights_path = weights_path or config.latest_weights_file()
    model = build_network((config.SEQUENCE_LENGTH, 1), n_vocab)
    print(f"Loading weights from: {weights_path}")
    model.load_weights(str(weights_path))
    return model
