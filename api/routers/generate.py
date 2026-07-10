from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.maistro import architectures
from backend.maistro import generate as generation
from backend.maistro.generate import GenerationConfig, generate
from backend.maistro.theory import MOODS, SCALES

from .. import jobs

router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    filename: str = "generated_output.mid"
    arch: str = architectures.DEFAULT_ARCHITECTURE
    n_notes: int = Field(default=300, ge=16, le=2000)

    # 0 reproduces the old greedy decoder. The useful band for this model is
    # 1.4-2.0 (see backend/maistro/sampling.py); the cap allows a little past it.
    temperature: float | None = Field(default=None, ge=0.0, le=2.5)
    top_k: int = Field(default=generation.DEFAULT_TOP_K, ge=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)

    key: str | None = None
    scale: str = "chromatic"
    mood: str | None = None
    tempo_bpm: int = Field(default=96, ge=40, le=208)
    seed: int | None = None

    def to_config(self) -> GenerationConfig:
        return GenerationConfig(
            arch=self.arch,
            n_notes=self.n_notes,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            key=self.key,
            scale=self.scale,
            mood=self.mood,
            tempo_bpm=self.tempo_bpm,
            seed=self.seed,
        )


@router.post("")
def start_generate(req: GenerateRequest = GenerateRequest()) -> dict:
    def task(job: jobs.Job):
        cfg = req.to_config().resolved()
        job.log.append(
            f"Sampling {cfg.n_notes} notes from {cfg.arch} "
            f"(temperature {cfg.temperature:.2f}, top-p {cfg.top_p:.2f})"
        )
        if cfg.key and cfg.scale != "chromatic":
            job.log.append(f"Steering toward {cfg.key} {cfg.scale.replace('_', ' ')}")
        if cfg.mood:
            job.log.append(f"Mood preset: {MOODS[cfg.mood].label}")

        result = generate(req.filename, cfg)
        job.log.append(
            f"Repetition rate {result.metrics['repetition_rate']:.2f}, "
            f"{result.metrics['distinct_pitch_classes']} distinct pitch classes"
        )
        job.progress = {"filename": result.midi_path.name}
        return {
            "filename": result.midi_path.name,
            "metrics": result.metrics,
            "config": result.config,
        }

    return {"job_id": jobs.submit(task)}


@router.get("/options")
def options() -> dict:
    """Everything the UI needs to render its controls, sourced from the backend registries."""
    return {
        "architectures": [
            {
                "key": spec.key,
                "label": spec.label,
                "description": spec.description,
                "trained": spec.key in architectures.trained_architectures(),
            }
            for spec in architectures.ARCHITECTURES.values()
        ],
        "default_architecture": architectures.DEFAULT_ARCHITECTURE,
        # The slider spans the band where temperature actually changes the output
        # for this checkpoint; below `min` the decoder is greedy. See sampling.py.
        "sampling": {
            "temperature": {
                "default": generation.DEFAULT_TEMPERATURE,
                "min": 1.0,
                "max": 2.1,
                "step": 0.05,
                "greedy_below": 1.3,
            },
            "top_p": generation.DEFAULT_TOP_P,
            "top_k": generation.DEFAULT_TOP_K,
        },
        "scales": sorted(SCALES),
        "keys": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"],
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
