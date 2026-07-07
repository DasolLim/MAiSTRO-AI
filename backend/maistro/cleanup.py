"""Delete transient generated audio/MIDI files left over from previous runs."""

from __future__ import annotations

from . import config

FILES_TO_DELETE = [
    "output.mid",
    "recording.mid",
    "generated_output.mid",
    "output.wav",
    "extended_output.mid",
    "temp_output.wav",
]

WILDCARD_PATTERNS = ["generated_music_*.mid"]


def delete_generated_files() -> list[str]:
    """Remove known transient files from the project root and output dir. Returns deleted paths."""
    deleted: list[str] = []

    for directory in (config.PROJECT_ROOT, config.OUTPUT_DIR):
        if not directory.exists():
            continue

        for name in FILES_TO_DELETE:
            path = directory / name
            if path.exists():
                path.unlink()
                deleted.append(str(path))

        for pattern in WILDCARD_PATTERNS:
            for path in directory.glob(pattern):
                path.unlink()
                deleted.append(str(path))

    return deleted
