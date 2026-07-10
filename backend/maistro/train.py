"""Train one of the architectures in architectures.py on the prepared note dataset.

Refactored out of the original train.py, which trained a model as a side effect
of merely importing the file. `train_model()` is now a plain function the API
layer can call from a background job and report progress from.

Two additions make the arena's comparison meaningful. Checkpoints are written to
`checkpoints/<arch>/` so architectures never overwrite each other, and a slice of
the data is held out for validation so `val_loss` -- not training loss -- is what
gets reported. A model that memorises 200 MIDI files can drive training loss down
indefinitely while sounding worse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from keras.callbacks import Callback, ModelCheckpoint

from . import architectures, config
from .dataset import extract_notes

# (epoch, total_epochs, loss, val_loss)
ProgressCallback = Callable[[int, int, float, Optional[float]], None]

VALIDATION_SPLIT = 0.1


@dataclass
class TrainingResult:
    arch: str
    epochs: int
    checkpoint_dir: str
    history: dict
    n_vocab: int
    sequences: int


def prepare_sequences(notes: list[str], n_vocab: int, arch: str):
    """Turn a flat note sequence into (network_input, network_output) training pairs.

    The input encoding depends on the architecture: the LSTMs read one normalised
    float per step, the transformer reads raw token ids and embeds them itself.

    Targets stay as integer ids. Every architecture compiles with
    `sparse_categorical_crossentropy`, so one-hotting them would allocate a
    (194341, 3388) float32 matrix -- 2.6GB -- to say exactly what 0.8MB of int32
    already says.
    """
    sequence_length = config.SEQUENCE_LENGTH
    pitchnames = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(pitchnames)}

    windows = []
    targets = []
    for i in range(len(notes) - sequence_length):
        windows.append([note_to_int[token] for token in notes[i : i + sequence_length]])
        targets.append(note_to_int[notes[i + sequence_length]])

    encoding = architectures.get(arch).encoding
    network_input = architectures.encode_windows(windows, n_vocab, encoding)
    network_output = np.asarray(targets, dtype=np.int32)

    return network_input, network_output


class _ProgressReportingCallback(Callback):
    def __init__(self, total_epochs: int, on_epoch_end: Optional[ProgressCallback]):
        super().__init__()
        self.total_epochs = total_epochs
        self._on_epoch_end = on_epoch_end

    def on_epoch_end(self, epoch, logs=None):
        if self._on_epoch_end is not None:
            logs = logs or {}
            self._on_epoch_end(
                epoch + 1, self.total_epochs, logs.get("loss", float("nan")), logs.get("val_loss")
            )


def train_model(
    arch: str = architectures.DEFAULT_ARCHITECTURE,
    epochs: int = 5,
    batch_size: int = 64,
    checkpoint_every_n_epochs: int = 5,
    validation_split: float = VALIDATION_SPLIT,
    on_epoch_end: Optional[ProgressCallback] = None,
) -> TrainingResult:
    """Extract notes, train `arch`, checkpoint weights, and return the run's history."""
    architectures.get(arch)  # fail fast on a typo'd architecture name

    notes = extract_notes()
    n_vocab = len(set(notes))
    print(f"Vocabulary size (n_vocab): {n_vocab}")

    network_input, network_output = prepare_sequences(notes, n_vocab, arch)
    print(f"Training {arch} on {len(network_input)} sequences")

    model = architectures.build_network(arch, config.SEQUENCE_LENGTH, n_vocab)

    checkpoint_dir = architectures.checkpoint_dir(arch)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    steps_per_epoch = max(len(network_input) // batch_size, 1)
    monitor = "val_loss" if validation_split > 0 else "loss"

    checkpoint = ModelCheckpoint(
        str(checkpoint_dir / ("weights-epoch{epoch:03d}-{" + monitor + ":.4f}.keras")),
        save_freq=steps_per_epoch * checkpoint_every_n_epochs,
        monitor=monitor,
        verbose=1,
        save_best_only=False,
        mode="min",
    )

    history = model.fit(
        network_input,
        network_output,
        epochs=epochs,
        batch_size=batch_size,
        # The windows overlap by one note, so a random split would leak almost every
        # validation window into training. Keras takes the *last* fraction without
        # shuffling, which holds out a contiguous tail of the corpus instead.
        validation_split=validation_split,
        callbacks=[checkpoint, _ProgressReportingCallback(epochs, on_epoch_end)],
    )

    serialisable_history = {
        key: [float(value) for value in values] for key, values in history.history.items()
    }
    (checkpoint_dir / "history.json").write_text(
        json.dumps(
            {"arch": arch, "epochs": epochs, "batch_size": batch_size, "history": serialisable_history},
            indent=2,
        )
    )

    return TrainingResult(
        arch=arch,
        epochs=epochs,
        checkpoint_dir=str(checkpoint_dir),
        history=serialisable_history,
        n_vocab=n_vocab,
        sequences=len(network_input),
    )


def load_history(arch: str) -> dict | None:
    """The last recorded training run for `arch`, for plotting loss curves."""
    path = architectures.checkpoint_dir(arch) / "history.json"
    return json.loads(path.read_text()) if path.exists() else None


def parameter_count(arch: str, n_vocab: int) -> int:
    """Trainable parameters, so the arena can report cost alongside quality."""
    model = architectures.build_network(arch, config.SEQUENCE_LENGTH, n_vocab)
    return int(np.sum([np.prod(w.shape) for w in model.trainable_weights]))
