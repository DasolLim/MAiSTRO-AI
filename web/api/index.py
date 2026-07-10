"""The deployed MAiSTRO API: a single Vercel Python function, no TensorFlow.

Vercel's Python runtime is 3.12 with a 500MB bundle ceiling. TensorFlow is 877MB
unpacked and requires Python <=3.10, so the full backend cannot deploy there at any
size. It does not need to: training is what needs TensorFlow, and inference is a
dozen matmuls over 3.9M weights. This function serves the transformer from a 7MB
`.npz` with numpy, giving a ~40MB bundle.

What this API deliberately does *not* expose, and why:

* **Jobs.** Functions are stateless, so there is nowhere to keep a job between the
  POST that starts it and the GET that polls it. Generation is fast enough (~10s for
  300 notes, against a 300s ceiling) to answer synchronously instead.
* **Train / dataset upload.** The filesystem is read-only and an invocation is
  measured in minutes, not hours.
* **Arena.** Needs two trained models and somewhere to persist votes.
* **MusicGen.** torch alone is 511MB.

All of those still work when the full backend runs locally. `GET /capabilities`
tells the frontend which of them this deployment actually has, so the UI can say so
plainly rather than failing.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from maistro import lite
from maistro.theory import MOODS, PITCH_CLASS_NAMES, SCALES

MODEL_DIR = Path(os.environ.get("MAISTRO_MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))

app = FastAPI(title="MAiSTRO API (serverless)", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bundle() -> lite.Bundle:
    try:
        return lite.load_bundle(str(MODEL_DIR))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No exported model found. Train the transformer "
                "(jupyter_notebooks/train_transformer_colab.ipynb) and place "
                "transformer.npz + vocabulary.npz in web/model/."
            ),
        ) from exc


class GenerateRequest(BaseModel):
    n_notes: int = Field(default=300, ge=16, le=lite.MAX_NOTES)
    temperature: float | None = Field(default=None, ge=0.0, le=2.5)
    top_k: int = Field(default=lite.DEFAULT_TOP_K, ge=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    key: str | None = None
    scale: str = "chromatic"
    mood: str | None = None
    tempo_bpm: int = Field(default=96, ge=40, le=208)
    seed: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "serverless"}


@app.get("/capabilities")
def capabilities() -> dict:
    """What this deployment can and cannot do. The UI reads this to degrade honestly."""
    return {
        "mode": "serverless",
        "model_loaded": (MODEL_DIR / "transformer.npz").exists(),
        "features": {
            "generate": True,
            "library": False,
            "arena": False,
            "train": False,
            "dataset": False,
            "musicgen": False,
        },
        "reason": (
            "This deployment runs a NumPy transformer in a stateless serverless function. "
            "Training, the library, the arena and MusicGen need a persistent filesystem and "
            "TensorFlow or torch — run the backend locally for those."
        ),
    }


@app.get("/generate/options")
def options() -> dict:
    """Mirrors the local backend's shape so the frontend needs no special case."""
    return {
        "architectures": [
            {
                "key": "transformer",
                "label": "Transformer",
                "description": (
                    "4-layer causal decoder, 3.9M parameters. The only architecture small "
                    "enough to serve without TensorFlow."
                ),
                "trained": (MODEL_DIR / "transformer.npz").exists(),
            }
        ],
        "default_architecture": "transformer",
        "sampling": {
            "temperature": {
                "default": lite.DEFAULT_TEMPERATURE,
                "min": 1.0,
                "max": 2.1,
                "step": 0.05,
                "greedy_below": 1.3,
            },
            "top_p": lite.DEFAULT_TOP_P,
            "top_k": lite.DEFAULT_TOP_K,
        },
        "scales": sorted(SCALES),
        "keys": list(PITCH_CLASS_NAMES),
        "moods": [
            {
                "key": mood.key,
                "label": mood.label,
                "description": mood.description,
                "temperature": mood.temperature,
                "scale": mood.suggested_scale,
            }
            for mood in MOODS.values()
        ],
    }


@app.post("/generate/sync")
def generate(req: GenerateRequest) -> dict:
    """Generate a piece and return the MIDI inline. No job, no file, no polling.

    The MIDI is base64 in the JSON body rather than a URL, because there is no
    writable disk to serve a later GET from. A 300-note piece is ~2KB.
    """
    bundle = _bundle()

    cfg = lite.LiteConfig(
        n_notes=req.n_notes,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        key=req.key,
        scale=req.scale,
        mood=req.mood,
        tempo_bpm=req.tempo_bpm,
        seed=req.seed,
    )
    result = lite.generate(bundle, cfg)

    return {
        "midi_base64": base64.b64encode(result.midi).decode("ascii"),
        "metrics": result.metrics,
        "config": result.config,
    }
