"""Central path configuration so every module works regardless of cwd."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = PROJECT_ROOT / "data"
NOTES_FILE = DATA_DIR / "notes"
SOUNDFONTS_DIR = PROJECT_ROOT / "soundfonts"
OUTPUT_DIR = PROJECT_ROOT / "output"
GUI_ASSETS_DIR = PROJECT_ROOT / "gui_assets"

SOUNDFONT_PATH = SOUNDFONTS_DIR / "SalC5Light2.sf2"

SEQUENCE_LENGTH = 100


def ensure_dirs() -> None:
    for path in (DATASET_DIR, DATA_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def latest_weights_file() -> Path:
    """Most recent trained model weights, preferring the modern .keras format."""
    candidates = list(PROJECT_ROOT.glob("*.keras")) + list(PROJECT_ROOT.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"No model weights (.keras or .h5) found in {PROJECT_ROOT}. Train a model first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)
