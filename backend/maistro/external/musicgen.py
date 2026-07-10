"""Text-to-music generation via Meta's MusicGen (audiocraft).

Where MAiSTRO's LSTM predicts the next *note*, MusicGen predicts the next frame
of an audio codec's latent tokens, conditioned on a text prompt. That difference
is the whole reason it is here: it composes in timbre, so it can render "lo-fi
hip hop with a dusty Rhodes" -- a description with no meaningful representation
in a piano-note vocabulary.

audiocraft pulls in torch and several GB of model weights, so it is an optional
extra (`pip install -r requirements-external.txt`) and imported lazily. Every
entry point degrades to a clear error rather than an ImportError traceback when
the extra is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .. import config

# MusicGen ships at several sizes. "small" (300M) runs on CPU in roughly real
# time and is the only one worth defaulting to without a GPU.
MODELS = {
    "facebook/musicgen-small": "300M params. Runs on CPU. Good enough to judge a prompt.",
    "facebook/musicgen-medium": "1.5B params. Noticeably richer; wants a GPU.",
    "facebook/musicgen-large": "3.3B params. Best quality, GPU only.",
}
DEFAULT_MODEL = "facebook/musicgen-small"

MAX_DURATION_SECONDS = 30
EXTERNAL_DIR = config.OUTPUT_DIR / "external"


@dataclass
class MusicGenResult:
    audio_path: Path
    prompt: str
    model_name: str
    duration: float


def is_available() -> bool:
    """True when the optional audiocraft extra is importable."""
    try:
        import audiocraft.models  # noqa: F401
    except ImportError:
        return False
    return True


def unavailable_reason() -> str | None:
    if is_available():
        return None
    return (
        "MusicGen needs the optional audio stack. Install it with "
        "`pip install -r requirements-external.txt` (downloads torch + audiocraft), "
        "then restart the API."
    )


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load and cache a MusicGen checkpoint. First call downloads several GB."""
    from audiocraft.models import MusicGen

    print(f"Loading {model_name} (first run downloads weights)…")
    return MusicGen.get_pretrained(model_name)


def _slugify(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return (slug[:48] or "prompt").rstrip("-")


def generate_audio(
    prompt: str,
    duration: float = 12.0,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 1.0,
    top_k: int = 250,
    on_progress=None,
) -> MusicGenResult:
    """Render `prompt` to a .wav in output/external/ and return its path."""
    reason = unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    if model_name not in MODELS:
        raise ValueError(f"unknown model {model_name!r}; expected one of {sorted(MODELS)}")

    duration = max(1.0, min(float(duration), MAX_DURATION_SECONDS))

    from audiocraft.data.audio import audio_write

    if on_progress:
        on_progress(f"Loading {model_name}…")
    model = _load_model(model_name)
    model.set_generation_params(duration=duration, temperature=temperature, top_k=top_k)

    if on_progress:
        on_progress(f"Generating {duration:.0f}s from: “{prompt}”")
    waveform = model.generate([prompt])[0]  # (channels, samples)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"musicgen-{_slugify(prompt)}-{datetime.now():%Y%m%d-%H%M%S}"

    # audio_write appends the extension and handles loudness normalisation, which
    # matters because MusicGen output is quiet relative to the soundfont renders.
    audio_write(
        str(EXTERNAL_DIR / stem),
        waveform.cpu(),
        model.sample_rate,
        strategy="loudness",
        loudness_compressor=True,
    )

    return MusicGenResult(
        audio_path=EXTERNAL_DIR / f"{stem}.wav",
        prompt=prompt,
        model_name=model_name,
        duration=duration,
    )
