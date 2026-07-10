"""Genre generation that MAiSTRO's symbolic model cannot reach, via MusicGen."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.maistro.external import musicgen

from .. import jobs

router = APIRouter(prefix="/external", tags=["external"])

# Starting points chosen to sit outside the training corpus: none of these are
# solo classical piano, which is the point of routing them to an audio model.
PROMPT_IDEAS = [
    "lo-fi hip hop with a dusty Rhodes piano and vinyl crackle",
    "driving synthwave with analog bass and gated reverb drums",
    "warm bossa nova guitar with brushed snare, recorded to tape",
    "cinematic orchestral build with taiko drums and low brass",
    "dub techno chords, deep sub bass, long tape delay",
]


class MusicGenRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=400)
    duration: float = Field(default=12.0, ge=1.0, le=musicgen.MAX_DURATION_SECONDS)
    model_name: str = musicgen.DEFAULT_MODEL
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)


@router.get("/musicgen/status")
def status() -> dict:
    """Whether the optional audio stack is installed, and what to run if it isn't."""
    return {
        "available": musicgen.is_available(),
        "reason": musicgen.unavailable_reason(),
        "models": [{"name": name, "description": desc} for name, desc in musicgen.MODELS.items()],
        "default_model": musicgen.DEFAULT_MODEL,
        "max_duration": musicgen.MAX_DURATION_SECONDS,
        "prompt_ideas": PROMPT_IDEAS,
    }


@router.post("/musicgen/generate")
def generate(req: MusicGenRequest) -> dict:
    reason = musicgen.unavailable_reason()
    if reason:
        raise HTTPException(status_code=503, detail=reason)

    def task(job: jobs.Job):
        result = musicgen.generate_audio(
            prompt=req.prompt,
            duration=req.duration,
            model_name=req.model_name,
            temperature=req.temperature,
            on_progress=job.log.append,
        )
        job.progress = {"filename": result.audio_path.name}
        return {
            "wav_filename": f"external/{result.audio_path.name}",
            "prompt": result.prompt,
            "model_name": result.model_name,
            "duration": result.duration,
        }

    return {"job_id": jobs.submit(task)}
