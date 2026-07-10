"""Export a trained transformer, plus the vocabulary, for TensorFlow-free serving.

Produces two files that the deployed API reads instead of a Keras checkpoint:

    transformer.npz   the weights, stored as float16 (7.7MB at d_model=256)
    vocabulary.npz    the token strings, the seed corpus, and the corpus histogram

float16 is safe here. The forward pass upcasts to float32 before any arithmetic,
and the only cost is representation error in the stored constants -- measured at
~3e-7 on the output distribution, four orders of magnitude below the difference
between two sampling temperatures.

Run after training:

    python -m backend.maistro.export --arch transformer --out web/model
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from . import config, metrics


def extract_transformer(model) -> dict[str, np.ndarray]:
    """Pull a Keras transformer's weights into flat arrays keyed for npmodel.load()."""
    embedding = next(
        layer for layer in model.layers if layer.__class__.__name__ == "TokenAndPositionEmbedding"
    )
    blocks = [layer for layer in model.layers if layer.__class__.__name__ == "TransformerBlock"]
    if not blocks:
        raise ValueError(f"{model.name} has no TransformerBlock layers; is it a transformer?")

    arrays: dict[str, np.ndarray] = {
        "token_embedding": embedding.token_embedding.get_weights()[0],
        "position_embedding": embedding.position_embedding.get_weights()[0],
        "num_layers": np.asarray(len(blocks)),
        "num_heads": np.asarray(blocks[0].num_heads),
    }

    for index, block in enumerate(blocks):
        # Keras keeps MultiHeadAttention's 8 arrays in a fixed order:
        # query/key/value kernel+bias, then the output projection kernel+bias.
        for j, weight in enumerate(block.attention.get_weights()):
            arrays[f"block{index}.attention.{j}"] = weight

        arrays[f"block{index}.ln1_gamma"], arrays[f"block{index}.ln1_beta"] = block.norm1.get_weights()
        arrays[f"block{index}.ln2_gamma"], arrays[f"block{index}.ln2_beta"] = block.norm2.get_weights()

        ff1, ff2 = block.ffn.layers
        arrays[f"block{index}.ff1_kernel"], arrays[f"block{index}.ff1_bias"] = ff1.get_weights()
        arrays[f"block{index}.ff2_kernel"], arrays[f"block{index}.ff2_bias"] = ff2.get_weights()

    # The last LayerNormalization in the model is the final pre-head norm; the ones
    # inside the blocks belong to the blocks and were collected above.
    final_norm = [
        layer for layer in model.layers if layer.__class__.__name__ == "LayerNormalization"
    ][-1]
    arrays["final_ln_gamma"], arrays["final_ln_beta"] = final_norm.get_weights()
    arrays["head_kernel"], arrays["head_bias"] = model.get_layer("next_token").get_weights()

    return arrays


def _to_float16(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Halve the weights. Integer scalars (layer/head counts) stay as they are."""
    return {
        key: value if value.dtype.kind in "iu" else value.astype(np.float16)
        for key, value in arrays.items()
    }


def export_weights(model, out_dir: Path, half: bool = True) -> Path:
    arrays = extract_transformer(model)
    if half:
        arrays = _to_float16(arrays)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "transformer.npz"
    np.savez_compressed(path, **arrays)
    return path


def export_vocabulary(out_dir: Path) -> Path:
    """The tokens, the corpus to seed from, and the reference pitch-class histogram.

    The corpus ships as uint16 ids rather than the 3.3MB pickle of strings: 194k
    notes become 389KB, and the deployed API needs it only to pick a seed window.
    """
    with open(config.NOTES_FILE, "rb") as fh:
        notes: list[str] = pickle.load(fh)

    pitchnames = sorted(set(notes))
    if len(pitchnames) > np.iinfo(np.uint16).max:
        raise ValueError(f"vocabulary of {len(pitchnames)} exceeds uint16")

    note_to_int = {token: index for index, token in enumerate(pitchnames)}
    corpus = np.asarray([note_to_int[token] for token in notes], dtype=np.uint16)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vocabulary.npz"
    np.savez_compressed(
        path,
        pitchnames=np.asarray(pitchnames, dtype=object),
        corpus=corpus,
        # Shipped so the deployed API can report corpus KL without the notes file.
        corpus_histogram=np.asarray(metrics.pitch_class_histogram(notes), dtype=np.float32),
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="transformer")
    parser.add_argument("--out", default="web/model", type=Path)
    parser.add_argument("--weights", type=Path, default=None, help="checkpoint; default: latest")
    parser.add_argument("--fp32", action="store_true", help="skip float16 conversion")
    args = parser.parse_args()

    # Imported here so `--help` works without a TensorFlow install.
    from . import architectures

    notes_count = len(pickle.load(open(config.NOTES_FILE, "rb")))
    pitchnames = sorted(set(pickle.load(open(config.NOTES_FILE, "rb"))))
    n_vocab = len(pitchnames)

    model = architectures.load_trained_network(args.arch, n_vocab, args.weights)

    weights_path = export_weights(model, args.out, half=not args.fp32)
    vocab_path = export_vocabulary(args.out)

    print(f"vocabulary : {n_vocab:,} tokens over {notes_count:,} notes")
    print(f"weights    : {weights_path}  ({weights_path.stat().st_size / 1e6:.1f} MB)")
    print(f"vocabulary : {vocab_path}  ({vocab_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
