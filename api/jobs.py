"""In-process background job registry.

Dataset prep, training and generation are long-running, blocking calls (Keras
`.fit()`, fluidsynth rendering), so each is run on a background thread and
tracked here; the frontend polls GET /jobs/{id} for progress instead of
holding an HTTP request open or requiring websockets.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    state: JobState = JobState.PENDING
    progress: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "state": self.state.value,
            "progress": self.progress,
            "log": self.log[-100:],
            "result": self.result,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def _run(job: Job, fn: Callable[[Job], Any]) -> None:
    job.state = JobState.RUNNING
    try:
        job.result = fn(job)
        job.state = JobState.DONE
    except Exception as exc:  # noqa: BLE001 - surfaced to the frontend, not swallowed
        job.error = str(exc)
        job.log.append(traceback.format_exc())
        job.state = JobState.ERROR


def submit(fn: Callable[[Job], Any]) -> str:
    """Run fn(job) on a background thread; fn should mutate job.progress/job.log as it goes."""
    job = Job(id=str(uuid.uuid4()))
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=_run, args=(job, fn), daemon=True).start()
    return job.id


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)
