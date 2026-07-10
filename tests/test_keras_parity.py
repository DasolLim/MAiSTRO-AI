"""The deployment's central claim: NumPy inference reproduces the Keras transformer.

If this drifts, every generated piece on the deployed site differs from what the
trained model would actually have played -- silently, with no error anywhere. The
training notebook runs the same assertion before it lets you download weights.

Needs TensorFlow, so it is skipped wherever TF is not installed. CI runs it in a
separate, slower job.
"""

from __future__ import annotations

import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow", reason="TensorFlow-only parity check")

from backend.maistro import architectures, export, npmodel  # noqa: E402

N_VOCAB = 128
SEQ_LEN = 16


@pytest.fixture(scope="module")
def keras_model():
    return architectures.build_transformer(
        SEQ_LEN, N_VOCAB, d_model=32, num_heads=2, num_layers=2, ff_dim=64
    )


def test_numpy_matches_keras_in_float32(keras_model, tmp_path):
    path = export.export_weights(keras_model, tmp_path, half=False)
    weights = npmodel.load(path)

    rng = np.random.default_rng(0)
    tokens = rng.integers(0, N_VOCAB, size=SEQ_LEN).astype(np.int32)

    keras_probs = np.asarray(keras_model(tokens[None, :], training=False))[0]
    numpy_probs = npmodel.predict_next(weights, tokens)

    np.testing.assert_allclose(numpy_probs, keras_probs, atol=1e-5)
    assert int(numpy_probs.argmax()) == int(keras_probs.argmax())


def test_float16_export_still_picks_the_same_note(keras_model, tmp_path):
    path = export.export_weights(keras_model, tmp_path, half=True)
    weights = npmodel.load(path)

    rng = np.random.default_rng(1)
    for _ in range(5):
        tokens = rng.integers(0, N_VOCAB, size=SEQ_LEN).astype(np.int32)
        keras_probs = np.asarray(keras_model(tokens[None, :], training=False))[0]
        numpy_probs = npmodel.predict_next(weights, tokens)

        # float16 storage moves the distribution by ~1e-7; it must not move the argmax.
        assert np.abs(keras_probs - numpy_probs).max() < 1e-3
        assert int(numpy_probs.argmax()) == int(keras_probs.argmax())


def test_gelu_matches_the_activation_the_network_trains_with(keras_model):
    x = np.linspace(-4, 4, 64).astype(np.float32)
    keras_gelu = np.asarray(architectures.gelu_tanh(tensorflow.constant(x)))
    np.testing.assert_allclose(npmodel.gelu(x), keras_gelu, atol=1e-6)


def test_every_architecture_builds_and_takes_integer_targets():
    """Sparse targets: one-hotting 194k x 3388 would allocate 2.6GB."""
    for arch in architectures.ARCHITECTURES:
        model = architectures.build_network(arch, SEQ_LEN, N_VOCAB)
        encoding = architectures.get(arch).encoding
        inputs = architectures.encode_windows(
            np.random.randint(0, N_VOCAB, (4, SEQ_LEN)), N_VOCAB, encoding
        )
        targets = np.random.randint(0, N_VOCAB, size=(4,)).astype(np.int32)
        loss = model.train_on_batch(inputs, targets)
        assert np.isfinite(loss)
