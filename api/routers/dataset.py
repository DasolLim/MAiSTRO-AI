from fastapi import APIRouter, File, UploadFile

from backend.maistro import config
from backend.maistro.dataset import prepare_dataset

from .. import jobs

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.post("/upload")
async def upload_midi(files: list[UploadFile] = File(...)) -> dict:
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for upload in files:
        if not upload.filename.lower().endswith((".mid", ".midi")):
            continue
        dest = config.DATASET_DIR / upload.filename
        dest.write_bytes(await upload.read())
        saved.append(upload.filename)
    return {"saved": saved, "count": len(saved)}


@router.post("/prepare")
def prepare(force: bool = False) -> dict:
    def task(job: jobs.Job):
        job.log.append("Parsing MIDI files into a note/chord/rest sequence...")
        stats = prepare_dataset(force=force)
        job.progress = {
            "midi_file_count": stats.midi_file_count,
            "note_count": stats.note_count,
            "vocab_size": stats.vocab_size,
        }
        return job.progress

    return {"job_id": jobs.submit(task)}


@router.get("/stats")
def stats() -> dict:
    midi_file_count = len(list(config.DATASET_DIR.glob("*.mid"))) if config.DATASET_DIR.exists() else 0
    return {"midi_file_count": midi_file_count, "notes_prepared": config.NOTES_FILE.exists()}
