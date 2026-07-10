from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.maistro import arena
from backend.maistro.generate import GenerationConfig

from .. import jobs

router = APIRouter(prefix="/arena", tags=["arena"])


class PairRequest(BaseModel):
    n_notes: int = Field(default=160, ge=32, le=600)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    key: str | None = None
    scale: str = "chromatic"
    mood: str | None = None
    tempo_bpm: int = Field(default=96, ge=40, le=208)


class VoteRequest(BaseModel):
    pair_id: str
    outcome: str  # "a" | "b" | "tie"
    note: str | None = None


@router.post("/pair")
def create_pair(req: PairRequest = PairRequest()) -> dict:
    """Kick off generation of one blinded A/B pair. Poll the returned job for the result."""
    if not arena.available_matchups():
        raise HTTPException(
            status_code=409,
            detail="The arena needs two trained architectures. Train a second one first.",
        )

    def task(job: jobs.Job):
        cfg = GenerationConfig(
            n_notes=req.n_notes,
            temperature=req.temperature,
            key=req.key,
            scale=req.scale,
            mood=req.mood,
            tempo_bpm=req.tempo_bpm,
        )
        pair = arena.create_pair(cfg, on_progress=job.log.append)
        job.log.append("Both takes ready. Listen blind, then vote.")
        return arena.pair_payload(pair)

    return {"job_id": jobs.submit(task)}


@router.post("/vote")
def vote(req: VoteRequest) -> dict:
    """Record a vote and reveal which architecture produced which clip."""
    try:
        return arena.record_vote(req.pair_id, req.outcome, req.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/leaderboard")
def leaderboard() -> dict:
    return {
        "leaderboard": arena.leaderboard(),
        "matchups": arena.available_matchups(),
        "recent_votes": arena.vote_history(limit=20),
    }


@router.post("/objective")
def objective(req: PairRequest = PairRequest()) -> dict:
    """Score every trained architecture on the reproducible metrics, no listener required."""

    def task(job: jobs.Job):
        job.log.append("Scoring each architecture over 3 takes…")
        cfg = GenerationConfig(
            n_notes=req.n_notes, temperature=req.temperature, key=req.key, scale=req.scale
        )
        return arena.objective_scores(cfg)

    return {"job_id": jobs.submit(task)}
