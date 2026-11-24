#!/usr/bin/env python3
"""Run resumable experiment jobs under the memory policy enforced by memguard.py.

Each job is a shell command that must be idempotent/resumable (all experiment
runners in this artifact append per-case JSONL records and skip finished cases).
Before (re)starting a job the supervisor waits until
    system RAM used < system_limit - margin   and   project RSS < project_limit - margin,
then waits a cool-down of 90 s x throttle level ("restart slowly"), and exports
    WORKERS = max(1, base_workers >> level), THROTTLE_LEVEL = level
so that a restarted job uses less memory. A job that is killed (by memguard or
otherwise) is retried up to --max-retries times.

Usage:
  python job_supervisor.py --name e1_qwen7b --workers 4 -- <command ...>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memguard import RUN_DIR, read_throttle, reading  # noqa: E402


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (RUN_DIR / "supervisor.log").open("a") as fh:
        fh.write(line + "\n")


def wait_for_headroom(project_limit: float, system_limit: float, margin: float, marker: str) -> None:
    while True:
        r = reading(marker)
        if (r["system_used_gb"] < system_limit - margin
                and r["project_gb"] < project_limit - margin):
            return
        log(f"waiting for memory headroom: project={r['project_gb']} GB system={r['system_used_gb']} GB")
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=40)
    ap.add_argument("--project-limit-gb", type=float, default=200.0)
    ap.add_argument("--system-limit-gb", type=float, default=490.0)
    ap.add_argument("--margin-gb", type=float, default=50.0)
    ap.add_argument("--marker", default="ManuscriptR2")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("missing command")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    status_file = RUN_DIR / f"job_{args.name}.json"
    for attempt in range(args.max_retries + 1):
        wait_for_headroom(args.project_limit_gb, args.system_limit_gb, args.margin_gb, args.marker)
        level = int(read_throttle().get("level", 0))
        if attempt > 0 or level > 0:
            cool = 90 * max(level, 1)
            log(f"{args.name}: cool-down {cool}s before attempt {attempt} (throttle level {level})")
            time.sleep(cool)
        workers = max(1, args.workers >> level)
        env = dict(os.environ, WORKERS=str(workers), THROTTLE_LEVEL=str(level))
        log(f"{args.name}: start attempt {attempt} workers={workers}: {' '.join(cmd)}")
        status_file.write_text(json.dumps({"state": "running", "attempt": attempt,
                                           "workers": workers, "cmd": cmd}, indent=2))
        rc = subprocess.call(cmd, env=env)
        if rc == 0:
            status_file.write_text(json.dumps({"state": "done", "attempt": attempt}, indent=2))
            log(f"{args.name}: finished OK")
            return
        log(f"{args.name}: exited rc={rc}; will retry")
    status_file.write_text(json.dumps({"state": "failed"}, indent=2))
    log(f"{args.name}: giving up after {args.max_retries} retries")
    sys.exit(1)


if __name__ == "__main__":
    main()
