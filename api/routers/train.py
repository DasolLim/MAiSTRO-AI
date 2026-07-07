from fastapi import APIRouter
from pydantic import BaseModel

from backend.maistro.train import train_model

from .. import jobs

router = APIRouter(prefix="/train", tags=["train"])


class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 64


@router.post("/start")
def start(req: TrainRequest) -> dict:
    def task(job: jobs.Job):
        def on_epoch_end(epoch: int, total_epochs: int, loss: float):
            job.progress = {"epoch": epoch, "total_epochs": total_epochs, "loss": loss}
            job.log.append(f"Epoch {epoch}/{total_epochs} - loss: {loss:.4f}")

        checkpoint_path = train_model(
            epochs=req.epochs, batch_size=req.batch_size, on_epoch_end=on_epoch_end
        )
        return {"checkpoint_path": checkpoint_path}

    return {"job_id": jobs.submit(task)}
