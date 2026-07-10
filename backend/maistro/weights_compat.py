"""Load Keras 3 `.keras` checkpoints into the Keras 2.10 networks this project builds.

The checkpoint shipped with the repo (`weights-epoch035-0.2775.keras`) was saved
by Keras 3.8, but the pinned stack is TensorFlow/Keras 2.10 -- `keras_self_attention`
has no Keras 3 build, so upgrading is not free. Keras 2's `load_weights()` expects
either an HDF5 file or a TF checkpoint and chokes on the Keras 3 archive, which is
a zip of `config.json` + `model.weights.h5`.

Rather than retrain, this reads the weight arrays straight out of the archive and
assigns them layer by layer. The two formats agree on the arrays themselves; they
disagree only on how those arrays are keyed. Every assignment is shape-checked, so
a silent mis-mapping fails loudly instead of producing a model that generates noise.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

WEIGHTS_ENTRY = "model.weights.h5"
METADATA_ENTRY = "metadata.json"

# Keras 3 nests a Bidirectional layer's two directions under named subgroups, while
# Keras 2 flattens them into one weight list: forward first, then backward.
_SUBGROUP_PRIORITY = ("forward_layer", "backward_layer")


def is_keras3_archive(path: Path) -> bool:
    """True if `path` is a Keras 3 `.keras` zip rather than a Keras 2 HDF5 file."""
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if WEIGHTS_ENTRY not in names or METADATA_ENTRY not in names:
            return False
        metadata = json.loads(archive.read(METADATA_ENTRY))
    return str(metadata.get("keras_version", "")).startswith("3")


def _sort_key(name: str):
    if name in _SUBGROUP_PRIORITY:
        return (0, _SUBGROUP_PRIORITY.index(name), "")
    if name == "vars":
        return (1, 0, "")
    return (2, 0, name)


def _collect_weights(group) -> list[np.ndarray]:
    """Flatten one layer's weight datasets into Keras 2's `layer.set_weights()` order."""
    arrays: list[np.ndarray] = []
    for name in sorted(group.keys(), key=_sort_key):
        item = group[name]
        if name == "vars":
            # Variable indices are strings; "10" must not sort before "2".
            arrays.extend(np.asarray(item[key]) for key in sorted(item.keys(), key=int))
        elif isinstance(item, h5py.Dataset):
            arrays.append(np.asarray(item))
        else:
            arrays.extend(_collect_weights(item))
    return arrays


def _base_name(layer_name: str) -> str:
    """`lstm_1` -> `lstm`. Keras 2 and Keras 3 disambiguate duplicate names differently."""
    head, _, tail = layer_name.rpartition("_")
    return head if head and tail.isdigit() else layer_name


def _find_group(layer_groups, layer_name: str, used: set[str]) -> str | None:
    for candidate in (layer_name, _base_name(layer_name)):
        if candidate in layer_groups and candidate not in used:
            return candidate
    return None


def load_keras3_weights(model, archive_path: Path) -> None:
    """Assign every weight in a Keras 3 archive onto an equivalent Keras 2 `model`."""
    with zipfile.ZipFile(archive_path) as archive:
        # Entries are stored uncompressed, so h5py can read the member in place
        # instead of unpacking a multi-gigabyte file to disk first.
        with archive.open(WEIGHTS_ENTRY) as handle:
            try:
                _assign(model, handle, archive_path)
                return
            except (OSError, ValueError) as exc:
                if "seek" not in str(exc).lower():
                    raise

        extracted = Path(archive.extract(WEIGHTS_ENTRY, archive_path.parent / ".keras3_tmp"))

    try:
        with open(extracted, "rb") as handle:
            _assign(model, handle, archive_path)
    finally:
        extracted.unlink(missing_ok=True)


def _assign(model, handle, archive_path: Path) -> None:
    with h5py.File(handle, "r") as weights_file:
        if "layers" not in weights_file:
            raise ValueError(f"{archive_path} has no /layers group; not a Keras 3 weights file")
        layer_groups = weights_file["layers"]

        used: set[str] = set()
        assigned = 0

        for layer in model.layers:
            if not layer.weights:
                continue

            group_name = _find_group(layer_groups, layer.name, used)
            if group_name is None:
                raise ValueError(
                    f"No weights for layer {layer.name!r} in {archive_path.name}. "
                    f"Archive has: {sorted(layer_groups)}"
                )
            used.add(group_name)

            arrays = _collect_weights(layer_groups[group_name])
            expected = [tuple(weight.shape) for weight in layer.weights]
            found = [array.shape for array in arrays]
            if expected != found:
                raise ValueError(
                    f"Shape mismatch for layer {layer.name!r}: the model wants {expected}, "
                    f"the checkpoint has {found}. The architectures have diverged."
                )

            layer.set_weights(arrays)
            assigned += 1

        if assigned == 0:
            raise ValueError(f"{archive_path.name} matched no layers in {model.name}")


def load_weights(model, weights_path: Path) -> None:
    """Load `weights_path` into `model`, transparently handling either Keras format.

    Native loading is tried first. It is the only correct path when the *runtime* is
    Keras 3 (Colab), where the manual mapper below would mis-order the weights of
    nested layers -- Keras 3 groups a block's sublayers alphabetically, Keras 2 lists
    them in creation order. The mapper exists only for the reverse case: a Keras 3
    archive being read by the Keras 2 runtime this project pins, where `load_weights`
    cannot open the zip at all.
    """
    try:
        model.load_weights(str(weights_path))
        return
    except Exception:
        if not is_keras3_archive(weights_path):
            raise

    print(f"Reading Keras 3 archive {weights_path.name} into a Keras 2 model…")
    load_keras3_weights(model, weights_path)
