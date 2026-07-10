"""Central path configuration so every module works regardless of cwd."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
NOTES_FILE = DATA_DIR / "notes"
SOUNDFONTS_DIR = PROJECT_ROOT / "soundfonts"
OUTPUT_DIR = PROJECT_ROOT / "output"
GUI_ASSETS_DIR = PROJECT_ROOT / "gui_assets"

# One subdirectory per architecture, so `lstm`, `lstm_attention` and
# `transformer` checkpoints never shadow each other during A/B comparison.
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ARENA_DIR = OUTPUT_DIR / "arena"
ARENA_STORE = DATA_DIR / "arena.json"

SOUNDFONT_PATH = SOUNDFONTS_DIR / "SalC5Light2.sf2"

SEQUENCE_LENGTH = 100

# Checkpoints predating the multi-architecture layout live at the project root
# and were all trained with the bidirectional LSTM + attention network.
LEGACY_WEIGHTS_ARCH = "lstm_attention"


def ensure_dirs() -> None:
    for path in (DATASET_DIR, DATA_DIR, OUTPUT_DIR, CHECKPOINTS_DIR, ARENA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def weights_candidates(arch: str) -> list[Path]:
    """Every checkpoint file that belongs to `arch`, newest last."""
    arch_dir = CHECKPOINTS_DIR / arch
    candidates = list(arch_dir.glob("*.keras")) + list(arch_dir.glob("*.h5"))

    if arch == LEGACY_WEIGHTS_ARCH:
        candidates += list(PROJECT_ROOT.glob("*.keras")) + list(PROJECT_ROOT.glob("*.h5"))

    return sorted(candidates, key=lambda p: p.stat().st_mtime)


def latest_weights_file(arch: str = LEGACY_WEIGHTS_ARCH) -> Path:
    """Most recent trained weights for `arch`, preferring the modern .keras format."""
    candidates = weights_candidates(arch)
    if not candidates:
        raise FileNotFoundError(
            f"No {arch} weights (.keras or .h5) found in {CHECKPOINTS_DIR / arch}. "
            f"Train it first: POST /train/start with arch={arch!r}."
        )
    return candidates[-1]
