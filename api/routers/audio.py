import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.maistro import config
from backend.maistro.audio import render_midi_to_wav

from .. import jobs

router = APIRouter(prefix="/audio", tags=["audio"])

MEDIA_TYPES = {".mid": "audio/midi", ".midi": "audio/midi", ".wav": "audio/wav", ".mp3": "audio/mpeg"}


def _resolve_output_file(filename: str):
    """Resolve `filename` inside the output directory, refusing to escape it.

    Filenames now carry a subdirectory (`arena/…`, `external/…`), so this can no
    longer be a plain join: `../../etc/passwd` would otherwise be served happily.
    """
    root = config.OUTPUT_DIR.resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="invalid filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return path


class RenderRequest(BaseModel):
    filename: str  # e.g. "generated_output.mid", relative to the output directory
    mode: str = "generation"  # "generation" or "collaboration"


@router.post("/render")
def render(req: RenderRequest) -> dict:
    def task(job: jobs.Job):
        job.log.append(f"Rendering {req.filename} to audio...")
        midi_path = _resolve_output_file(req.filename)
        wav_path = render_midi_to_wav(midi_path, mode=req.mode)
        return {"wav_filename": str(wav_path.relative_to(config.OUTPUT_DIR.resolve()))}

    return {"job_id": jobs.submit(task)}


@router.get("/file/{filename:path}")
def get_file(filename: str) -> FileResponse:
    """Serve a rendered .wav or a raw .mid — the browser player parses MIDI itself."""
    path = _resolve_output_file(filename)
    media_type = MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(str(path), media_type=media_type)


@router.get("/library")
def library() -> dict:
    if not config.OUTPUT_DIR.exists():
        return {"tracks": []}

    tracks = []
    for midi_path in sorted(
        config.OUTPUT_DIR.glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True
    ):
        wav_path = midi_path.with_suffix(".wav")
        sidecar = midi_path.with_suffix(".json")

        # generate() writes a sidecar with the exact config and metrics for the take,
        # so the library can show how a piece was made and reproduce it from its seed.
        details = json.loads(sidecar.read_text()) if sidecar.exists() else {}

        tracks.append(
            {
                "name": midi_path.stem,
                "midi_filename": midi_path.name,
                "wav_filename": wav_path.name if wav_path.exists() else None,
                "created_at": midi_path.stat().st_mtime,
                "config": details.get("config"),
                "metrics": details.get("metrics"),
            }
        )
    return {"tracks": tracks}
