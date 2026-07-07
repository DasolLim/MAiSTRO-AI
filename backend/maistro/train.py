"""Train the LSTM+attention model on the prepared note dataset.

Refactored out of the original train.py, which trained a model as a side effect
of merely importing the file. `train_model()` is now a plain function the API
layer can call from a background job and report progress from.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from keras import utils
from keras.callbacks import Callback, ModelCheckpoint

from . import config
from .dataset import extract_notes
from .model import build_network

ProgressCallback = Callable[[int, int, float], None]  # (epoch, total_epochs, loss)


def prepare_sequences(notes: list[str], n_vocab: int):
    """Turn a flat note sequence into (network_input, network_output) training pairs."""
    sequence_length = config.SEQUENCE_LENGTH
    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[c] for c in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = utils.to_categorical(network_output)

    return network_input, network_output


class _ProgressReportingCallback(Callback):
    def __init__(self, total_epochs: int, on_epoch_end: Optional[ProgressCallback]):
        super().__init__()
        self.total_epochs = total_epochs
        self._on_epoch_end = on_epoch_end

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end is not None:
            loss = (logs or {}).get("loss", float("nan"))
            self._on_epoch_end(epoch + 1, self.total_epochs, loss)


def train_model(
    epochs: int = 5,
    batch_size: int = 64,
    checkpoint_every_n_epochs: int = 5,
    on_epoch_end: Optional[ProgressCallback] = None,
) -> str:
    """Extract notes, train the network, and checkpoint weights.

    Returns the checkpoint filepath template used for this run.
    """
    notes = extract_notes()
    n_vocab = len(set(notes))
    print(f"Vocabulary size (n_vocab): {n_vocab}")

    network_input, network_output = prepare_sequences(notes, n_vocab)
    print(f"Total training sequences generated: {len(network_input)}")

    model = build_network((network_input.shape[1], network_input.shape[2]), n_vocab)

    steps_per_epoch = max(len(network_input) // batch_size, 1)
    save_freq = steps_per_epoch * checkpoint_every_n_epochs
    filepath = str(config.PROJECT_ROOT / "weights-epoch{epoch:03d}-{loss:.4f}.keras")

    checkpoint = ModelCheckpoint(
        filepath,
        save_freq=save_freq,
        monitor="loss",
        verbose=1,
        save_best_only=False,
        mode="min",
    )
    progress = _ProgressReportingCallback(epochs, on_epoch_end)

    model.fit(
        network_input,
        network_output,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, progress],
    )

    return filepath
