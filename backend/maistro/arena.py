"""Blind A/B listening test between architectures, scored with Elo.

A single loss curve tells you which model fits the training corpus. It does not
tell you which model people would rather listen to -- categorical cross-entropy
punishes a beautiful passing tone the corpus happened not to contain. So the
arena pairs two architectures on the same generation settings, hides which is
which, and asks a listener to pick.

Elo is the right scorer here because votes arrive as pairwise comparisons rather
than absolute scores, listeners disagree, and new architectures can be added
without re-running every earlier comparison. Ratings and every individual vote
are persisted so the record can be replayed or exported.
"""

from __future__ import annotations

import json
import random
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from . import architectures, config, metrics
from .generate import GenerationConfig, generate, generate_tokens

STARTING_ELO = 1500.0
# K=24 is the standard "provisional" K-factor: fast enough that a leaderboard is
# informative after a few dozen votes, slow enough that one listener cannot
# invert it. Ties count as half a win to each side.
K_FACTOR = 24.0

Outcome = str  # "a" | "b" | "tie"

_MAX_OPEN_PAIRS = 200


@dataclass
class Clip:
    """One side of a pair. `arch` is withheld from the API until the vote lands."""

    clip_id: str
    arch: str
    midi_filename: str
    metrics: dict = field(default_factory=dict)


@dataclass
class Pair:
    pair_id: str
    a: Clip
    b: Clip
    config: dict
    voted: bool = False


_lock = threading.Lock()
_open_pairs: dict[str, Pair] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_store() -> dict:
    if not config.ARENA_STORE.exists():
        return {"ratings": {}, "votes": []}
    return json.loads(config.ARENA_STORE.read_text())


def _save_store(store: dict) -> None:
    config.ARENA_STORE.parent.mkdir(parents=True, exist_ok=True)
    config.ARENA_STORE.write_text(json.dumps(store, indent=2))


def _rating(store: dict, arch: str) -> float:
    return store["ratings"].get(arch, {}).get("elo", STARTING_ELO)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo/logistic model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _apply_elo(store: dict, arch_a: str, arch_b: str, outcome: Outcome) -> None:
    rating_a, rating_b = _rating(store, arch_a), _rating(store, arch_b)
    score_a = {"a": 1.0, "b": 0.0, "tie": 0.5}[outcome]

    expected_a = expected_score(rating_a, rating_b)
    new_a = rating_a + K_FACTOR * (score_a - expected_a)
    new_b = rating_b + K_FACTOR * ((1.0 - score_a) - (1.0 - expected_a))

    for arch, new_rating, score in ((arch_a, new_a, score_a), (arch_b, new_b, 1.0 - score_a)):
        record = store["ratings"].setdefault(
            arch, {"elo": STARTING_ELO, "games": 0, "wins": 0, "losses": 0, "ties": 0}
        )
        record["elo"] = round(new_rating, 1)
        record["games"] += 1
        if score == 1.0:
            record["wins"] += 1
        elif score == 0.0:
            record["losses"] += 1
        else:
            record["ties"] += 1


def available_matchups() -> list[tuple[str, str]]:
    """Every unordered pair of architectures that currently has trained weights."""
    trained = architectures.trained_architectures()
    return [(a, b) for i, a in enumerate(trained) for b in trained[i + 1 :]]


def create_pair(
    cfg: GenerationConfig | None = None,
    matchup: tuple[str, str] | None = None,
    on_progress=None,
) -> Pair:
    """Generate one clip from each of two architectures under identical settings.

    Both clips share an RNG seed and every conditioning knob, so the only variable
    left in the comparison is the architecture itself.
    """
    matchups = available_matchups()
    if not matchups:
        raise RuntimeError(
            "The arena needs at least two trained architectures. Train another with "
            "POST /train/start {\"arch\": \"transformer\"}."
        )

    if matchup is None:
        matchup = random.choice(matchups)
    # Randomise which architecture is presented as "A" so a listener who votes
    # left-first cannot systematically favour one model.
    arch_a, arch_b = random.sample(list(matchup), 2)

    base = cfg or GenerationConfig()
    base = base.resolved()
    if base.seed is None:
        base.seed = int(uuid.uuid4().int % (2**32))

    pair_id = uuid.uuid4().hex[:12]
    config.ARENA_DIR.mkdir(parents=True, exist_ok=True)

    clips: list[Clip] = []
    for side, arch in (("a", arch_a), ("b", arch_b)):
        # The frontend streams this log while the pair renders, so it must never
        # name the architecture — that would unblind the test before the vote.
        if on_progress:
            on_progress(f"Composing take {side.upper()}…")

        side_cfg = GenerationConfig(**{**asdict(base), "arch": arch})
        filename = f"arena/{pair_id}-{side}.mid"
        result = generate(filename, side_cfg, write_sidecar=False)
        clips.append(
            Clip(
                clip_id=f"{pair_id}-{side}",
                arch=arch,
                midi_filename=filename,
                metrics=result.metrics,
            )
        )

    pair = Pair(pair_id=pair_id, a=clips[0], b=clips[1], config=asdict(base))
    with _lock:
        # Votes are persisted to disk; the in-memory map only needs to hold pairs
        # long enough for the listener to finish auditioning them.
        if len(_open_pairs) > _MAX_OPEN_PAIRS:
            for stale_id, stale in list(_open_pairs.items()):
                if stale.voted:
                    _open_pairs.pop(stale_id, None)
        _open_pairs[pair_id] = pair
    return pair


def pair_payload(pair: Pair) -> dict:
    """The blinded view sent to the listener: filenames and nothing identifying.

    `config` carries the shared brief both takes were generated under, minus the
    architecture field — that is the one thing the listener must not see.
    """
    shared_config = {key: value for key, value in pair.config.items() if key != "arch"}
    return {
        "pair_id": pair.pair_id,
        "a": {"clip_id": pair.a.clip_id, "midi_filename": pair.a.midi_filename},
        "b": {"clip_id": pair.b.clip_id, "midi_filename": pair.b.midi_filename},
        "config": shared_config,
    }


def record_vote(pair_id: str, outcome: Outcome, note: str | None = None) -> dict:
    """Score a vote, update Elo, and reveal which architecture was which."""
    if outcome not in ("a", "b", "tie"):
        raise ValueError(f"outcome must be 'a', 'b' or 'tie', got {outcome!r}")

    with _lock:
        pair = _open_pairs.get(pair_id)
        if pair is None:
            raise KeyError(f"unknown or expired pair {pair_id!r}")
        if pair.voted:
            raise ValueError(f"pair {pair_id!r} has already been voted on")
        pair.voted = True

        store = _load_store()
        _apply_elo(store, pair.a.arch, pair.b.arch, outcome)
        store["votes"].append(
            {
                "pair_id": pair_id,
                "arch_a": pair.a.arch,
                "arch_b": pair.b.arch,
                "outcome": outcome,
                "note": note,
                "config": pair.config,
                "metrics": {"a": pair.a.metrics, "b": pair.b.metrics},
                "at": _now(),
            }
        )
        _save_store(store)

    return {
        "pair_id": pair_id,
        "outcome": outcome,
        "reveal": {
            "a": {"arch": pair.a.arch, "metrics": pair.a.metrics},
            "b": {"arch": pair.b.arch, "metrics": pair.b.metrics},
        },
        "leaderboard": leaderboard(),
    }


def leaderboard() -> list[dict]:
    """Architectures ranked by Elo, annotated with their registry metadata."""
    store = _load_store()
    rows = []

    for arch in architectures.ARCHITECTURES:
        spec = architectures.get(arch)
        record = store["ratings"].get(
            arch, {"elo": STARTING_ELO, "games": 0, "wins": 0, "losses": 0, "ties": 0}
        )
        decided = record["wins"] + record["losses"]
        rows.append(
            {
                "arch": arch,
                "label": spec.label,
                "description": spec.description,
                "trained": arch in architectures.trained_architectures(),
                "elo": record["elo"],
                "games": record["games"],
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "win_rate": (record["wins"] / decided) if decided else None,
            }
        )

    return sorted(rows, key=lambda row: (-row["elo"], row["arch"]))


def vote_history(limit: int = 50) -> list[dict]:
    return _load_store()["votes"][-limit:][::-1]


def objective_scores(cfg: GenerationConfig | None = None, samples: int = 3) -> dict:
    """Score every trained architecture on metrics.evaluate, averaged over `samples` takes.

    Complements the human vote: cheap, reproducible, and it runs without a listener.
    """
    base = (cfg or GenerationConfig()).resolved()
    results: dict[str, dict] = {}

    for arch in architectures.trained_architectures():
        runs = [
            metrics.evaluate(
                generate_tokens(
                    GenerationConfig(**{**asdict(base), "arch": arch, "seed": (base.seed or 0) + i})
                )
            )
            for i in range(samples)
        ]

        numeric_keys = {
            key for run in runs for key, value in run.items() if isinstance(value, (int, float))
        }
        results[arch] = {
            key: sum(run[key] for run in runs if run.get(key) is not None) / samples
            for key in numeric_keys
        }

    return results
