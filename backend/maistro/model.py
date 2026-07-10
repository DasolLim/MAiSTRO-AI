"""Back-compat shim for the single-architecture API this module used to expose.

The network definitions moved to architectures.py when the transformer and the
plain-LSTM baseline were added. `backend/legacy_gui/collaborate.py` and any
notebook that imported `build_network`/`load_trained_network` still work, and
still get the bidirectional LSTM + attention model they were written against.
"""

from pathlib import Path

from keras.models import Model

from . import architectures, config

DEFAULT_ARCH = config.LEGACY_WEIGHTS_ARCH


def build_network(input_shape: tuple[int, int], n_vocab: int) -> Model:
    """Build the LSTM + attention architecture. input_shape: (sequence_length, features)."""
    return architectures.build_network(DEFAULT_ARCH, input_shape[0], n_vocab)


def load_trained_network(n_vocab: int, weights_path: Path | None = None) -> Model:
    """Build the LSTM + attention network and load its most recent weights."""
    return architectures.load_trained_network(DEFAULT_ARCH, n_vocab, weights_path)
