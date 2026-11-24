"""Small I/O helpers shared by the experiment runners.

Every runner appends one JSON record per finished unit of work and skips units
whose key is already present, so that a job interrupted by the memory watchdog
(scripts/memguard.py) can be restarted without repeating work.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

CODE_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = CODE_ROOT / "output"
DATA_ROOT = CODE_ROOT / "data"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # a partially written last line after a hard kill is skipped
                continue
    return rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    yield from read_jsonl(path)


def done_keys(path: Path, key: str = "key") -> Set[str]:
    return {r[key] for r in read_jsonl(path) if key in r}


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def workers_from_env(default: int) -> int:
    """Worker/batch count, reduced by job_supervisor.py after a memory stop."""
    try:
        return max(1, int(os.environ.get("WORKERS", default)))
    except ValueError:
        return default
