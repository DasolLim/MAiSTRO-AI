"""Text-to-music generation via Meta's MusicGen, loaded through Hugging Face.

Where MAiSTRO's LSTM predicts the next *note*, MusicGen predicts the next frame of
an audio codec's latent tokens, conditioned on a text prompt. That difference is
the whole reason it is here: it composes in timbre, so it can render "lo-fi hip
hop with a dusty Rhodes" -- a description with no meaningful representation in a
piano-note vocabulary.

## Why transformers rather than audiocraft

Meta ships MusicGen in `audiocraft`, which depends on `xformers`. xformers has no
arm64 macOS wheel and needs torch importable merely to resolve its build metadata,
so `pip install audiocraft` fails outright on Apple Silicon. Hugging Face's
`MusicgenForConditionalGeneration` loads the identical `facebook/musicgen-*`
checkpoints, needs no xformers, and runs on CPU, CUDA or Apple's MPS backend.

torch and transformers are an optional extra (`requirements-external.txt`) and are
imported lazily. Every entry point degrades to a clear message rather than an
ImportError traceback when the extra is not installed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

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

# MusicGen's codec runs at 50 latent frames per second of audio.
TOKENS_PER_SECOND = 50

# Classifier-free guidance strength. 3.0 is Meta's default: high enough to follow
# the prompt, low enough to leave the model some room.
GUIDANCE_SCALE = 3.0

# Peak-normalise to just under full scale. MusicGen output is quiet next to the
# soundfont renders, and this avoids a loudness dependency for one adapter.
TARGET_PEAK = 0.89


@dataclass
class MusicGenResult:
    audio_path: Path
    prompt: str
    model_name: str
    duration: float
    sample_rate: int


def is_available() -> bool:
    """True when the optional torch + transformers extra is importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def unavailable_reason() -> str | None:
    if is_available():
        return None
    return (
        "MusicGen needs the optional audio stack. Install it with "
        "`pip install -r requirements-external.txt` (downloads torch + transformers), "
        "then restart the API."
    )


def _resolve_device():
    """Pick a torch device, honouring MAISTRO_MUSICGEN_DEVICE if it is set.

    CPU is the default even on Apple Silicon: several ops in MusicGen's sampling
    loop fall back off MPS anyway, and a wrong-device crash mid-generation is a
    worse experience than a slower correct one. Set the env var to opt in.
    """
    import torch

    override = os.environ.get("MAISTRO_MUSICGEN_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load and cache a MusicGen checkpoint. The first call downloads several GB."""
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    print(f"Loading {model_name} (first run downloads weights)…")
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    device = _resolve_device()
    model.to(device)
    model.eval()
    return processor, model, device


def _slugify(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return (slug[:48] or "prompt").rstrip("-")


def _normalise(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.abs(waveform).max())
    return waveform if peak == 0 else waveform * (TARGET_PEAK / peak)


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

    import soundfile as sf
    import torch

    if on_progress:
        on_progress(f"Loading {model_name}…")
    processor, model, device = _load_model(model_name)

    if on_progress:
        on_progress(f"Generating {duration:.0f}s on {device.type} from: “{prompt}”")

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=GUIDANCE_SCALE,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=int(duration * TOKENS_PER_SECOND),
        )

    sample_rate = model.config.audio_encoder.sampling_rate
    # (batch, channels, samples) -> mono float32
    waveform = tokens[0, 0].to(torch.float32).cpu().numpy()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"musicgen-{_slugify(prompt)}-{datetime.now():%Y%m%d-%H%M%S}"
    audio_path = EXTERNAL_DIR / f"{stem}.wav"
    sf.write(str(audio_path), _normalise(waveform), sample_rate)

    if on_progress:
        on_progress(f"Wrote {audio_path.name} ({len(waveform) / sample_rate:.1f}s)")

    return MusicGenResult(
        audio_path=audio_path,
        prompt=prompt,
        model_name=model_name,
        duration=len(waveform) / sample_rate,
        sample_rate=sample_rate,
    )
