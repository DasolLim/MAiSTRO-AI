from fastapi import APIRouter
from pydantic import BaseModel

from backend.maistro.generate import generate

from .. import jobs

router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateRequest(BaseModel):
    filename: str = "generated_output.mid"


@router.post("")
def start_generate(req: GenerateRequest = GenerateRequest()) -> dict:
    def task(job: jobs.Job):
        job.log.append("Sampling notes from the trained model...")
        path = generate(req.filename)
        job.progress = {"filename": path.name}
        return {"filename": path.name}

    return {"job_id": jobs.submit(task)}
