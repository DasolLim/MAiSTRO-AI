from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.maistro import architectures
from backend.maistro.train import load_history, train_model

from .. import jobs

router = APIRouter(prefix="/train", tags=["train"])


class TrainRequest(BaseModel):
    arch: str = architectures.DEFAULT_ARCHITECTURE
    epochs: int = Field(default=5, ge=1, le=500)
    batch_size: int = Field(default=64, ge=1, le=1024)
    validation_split: float = Field(default=0.1, ge=0.0, lt=0.5)


@router.post("/start")
def start(req: TrainRequest) -> dict:
    if req.arch not in architectures.ARCHITECTURES:
        raise HTTPException(status_code=400, detail=f"unknown architecture {req.arch!r}")

    def task(job: jobs.Job):
        def on_epoch_end(epoch: int, total_epochs: int, loss: float, val_loss: float | None):
            job.progress = {
                "arch": req.arch,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "loss": loss,
                "val_loss": val_loss,
            }
            suffix = f" - val_loss: {val_loss:.4f}" if val_loss is not None else ""
            job.log.append(f"Epoch {epoch}/{total_epochs} - loss: {loss:.4f}{suffix}")

        result = train_model(
            arch=req.arch,
            epochs=req.epochs,
            batch_size=req.batch_size,
            validation_split=req.validation_split,
            on_epoch_end=on_epoch_end,
        )
        return {
            "arch": result.arch,
            "checkpoint_dir": result.checkpoint_dir,
            "n_vocab": result.n_vocab,
            "sequences": result.sequences,
            "history": result.history,
        }

    return {"job_id": jobs.submit(task)}


@router.get("/history/{arch}")
def history(arch: str) -> dict:
    if arch not in architectures.ARCHITECTURES:
        raise HTTPException(status_code=404, detail=f"unknown architecture {arch!r}")
    record = load_history(arch)
    if record is None:
        raise HTTPException(status_code=404, detail=f"{arch} has not been trained yet")
    return record
