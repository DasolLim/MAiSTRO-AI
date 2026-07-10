"""Copy the TensorFlow-free modules into web/maistro/ for the serverless deploy.

Vercel builds from a single Root Directory and will not reach outside it, so the
deployed function needs its own copy of the pure modules. Rather than let two
copies drift, this script is the only thing allowed to write web/maistro/, and
`--check` fails when the copy is stale -- run it in CI, or before deploying.

    python scripts/sync_web_bundle.py           # copy
    python scripts/sync_web_bundle.py --check   # verify, exit 1 if stale

Only modules that import nothing heavier than numpy are eligible. The guard below
enforces that: a stray `import tensorflow` in one of them fails the sync rather
than the deploy.
"""

from __future__ import annotations

import argparse
import ast
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backend" / "maistro"
DEST = ROOT / "web" / "maistro"

# The dependency closure of lite.generate(), and nothing else.
MODULES = [
    "__init__.py",
    "config.py",
    "theory.py",
    "sampling.py",
    "conditioning.py",
    "metrics.py",
    "npmodel.py",
    "midi_writer.py",
    "lite.py",
]

# Anything a 500MB serverless bundle cannot afford.
FORBIDDEN = {"tensorflow", "keras", "keras_self_attention", "music21", "torch", "scipy",
             "transformers", "pretty_midi", "pydub", "soundfile", "pypianoroll", "matplotlib"}


def _imported_roots(path: Path) -> set[str]:
    """Top-level package names imported anywhere in `path`, including inside functions."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def check_pure(name: str) -> list[str]:
    offenders = sorted(_imported_roots(SOURCE / name) & FORBIDDEN)
    return offenders


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of copying")
    args = parser.parse_args()

    problems: list[str] = []
    for name in MODULES:
        source = SOURCE / name
        if not source.exists():
            problems.append(f"missing source module: {source.relative_to(ROOT)}")
            continue

        offenders = check_pure(name)
        if offenders:
            problems.append(f"{name} imports {', '.join(offenders)} — not serverless-safe")

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []

    for name in MODULES:
        source, target = SOURCE / name, DEST / name
        if args.check:
            if not target.exists() or not filecmp.cmp(source, target, shallow=False):
                stale.append(name)
        else:
            shutil.copy2(source, target)

    if args.check:
        if stale:
            print(f"error: web/maistro is stale: {', '.join(stale)}", file=sys.stderr)
            print("run: python scripts/sync_web_bundle.py", file=sys.stderr)
            return 1
        print(f"web/maistro is in sync ({len(MODULES)} modules)")
        return 0

    total = sum((DEST / name).stat().st_size for name in MODULES)
    print(f"synced {len(MODULES)} modules -> {DEST.relative_to(ROOT)} ({total / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
