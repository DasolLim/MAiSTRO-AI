from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.maistro import config
from backend.maistro.audio import render_midi_to_wav

from .. import jobs

router = APIRouter(prefix="/audio", tags=["audio"])


class RenderRequest(BaseModel):
    filename: str  # e.g. "generated_output.mid", relative to the output directory
    mode: str = "generation"  # "generation" or "collaboration"


@router.post("/render")
def render(req: RenderRequest) -> dict:
    def task(job: jobs.Job):
        job.log.append(f"Rendering {req.filename} to audio...")
        midi_path = config.OUTPUT_DIR / req.filename
        wav_path = render_midi_to_wav(midi_path, mode=req.mode)
        return {"wav_filename": wav_path.name}

    return {"job_id": jobs.submit(task)}


@router.get("/file/{filename}")
def get_file(filename: str) -> FileResponse:
    path = config.OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(path), media_type="audio/wav")


@router.get("/library")
def library() -> dict:
    if not config.OUTPUT_DIR.exists():
        return {"tracks": []}

    tracks = []
    for midi_path in sorted(
        config.OUTPUT_DIR.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        wav_path = midi_path.with_suffix(".wav")
        tracks.append(
            {
                "name": midi_path.stem,
                "midi_filename": midi_path.name,
                "wav_filename": wav_path.name if wav_path.exists() else None,
                "created_at": midi_path.stat().st_mtime,
            }
        )
    return {"tracks": tracks}
