from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import jobs
from .routers import arena, audio, dataset, external, generate, train

app = FastAPI(title="MAiSTRO API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    # Vercel preview URLs are generated per-deployment (e.g. maistro-git-main-you.vercel.app),
    # so match the domain pattern rather than hardcoding one.
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(train.router)
app.include_router(generate.router)
app.include_router(audio.router)
app.include_router(arena.router)
app.include_router(external.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": "local"}


@app.get("/capabilities")
def capabilities() -> dict:
    """What this backend can do. The deployed serverless API answers the same shape.

    The frontend reads this once and hides or explains whatever is missing, so a
    visitor to the free-tier deployment never clicks into a feature that cannot work
    there. Running locally, everything is available.
    """
    return {
        "mode": "local",
        "model_loaded": True,
        "features": {
            "generate": True,
            "library": True,
            "arena": True,
            "train": True,
            "dataset": True,
            "musicgen": True,
        },
        "reason": None,
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()
