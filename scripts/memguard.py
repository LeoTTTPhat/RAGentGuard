#!/usr/bin/env python3
"""Memory watchdog for the ManuscriptR2 experiment campaign.

Limits requested by the user:
  * processes of this project must stay below --project-limit-gb (default 200 GB);
  * total RAM in use on the machine must stay below --system-limit-gb (default 490 GB).

When either limit is reached the watchdog stops every project job (SIGTERM, then
SIGKILL after a grace period), raises the throttle level stored in
run/throttle.json and appends an event to run/memguard_events.jsonl.
job_supervisor.py restarts interrupted jobs only after memory has fallen well
below both limits and a cool-down proportional to the throttle level has passed,
and it hands the job fewer workers (WORKERS = base >> level).
After --relax-after-min minutes well below both limits the level is lowered by one
(new jobs only), so the project ramps back up slowly.

A process belongs to the project when its command line contains the project
marker (default "ManuscriptR2"), when its PID is registered in run/pids/, or
when it descends from such a process. Processes whose command line contains
"memguard" or "job_supervisor" are monitored but never killed.

Usage:
  python memguard.py                 # run forever (start in background)
  python memguard.py --once          # print one reading and exit
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

GB = 1024 ** 3
RUN_DIR = Path(__file__).resolve().parents[1] / "run"
EXEMPT_MARKERS = ("memguard", "job_supervisor", "queue_", "serve_mlx_forever", "tail -F", "grep --line-buffered")


def _registered_pids() -> set[int]:
    pids: set[int] = set()
    pid_dir = RUN_DIR / "pids"
    if pid_dir.is_dir():
        for f in pid_dir.glob("*.pid"):
            try:
                pids.add(int(f.read_text().strip()))
            except (ValueError, OSError):
                pass
    return pids


def project_processes(marker: str) -> list[psutil.Process]:
    roots: set[int] = set(_registered_pids())
    me = os.getpid()
    # read each command line separately: a process that exits during the scan can make psutil raise
    # SystemError/OSError from inside process_iter(attrs=...), which once killed this guard
    for p in psutil.process_iter():
        try:
            cmd = " ".join(p.cmdline() or [])
        except (psutil.Error, TypeError, SystemError, OSError):
            continue
        if marker in cmd and p.pid != me:
            roots.add(p.pid)
    procs: dict[int, psutil.Process] = {}
    for pid in roots:
        try:
            root = psutil.Process(pid)
        except psutil.Error:
            continue
        procs[pid] = root
        try:
            for child in root.children(recursive=True):
                procs[child.pid] = child
        except psutil.Error:
            pass
    procs.pop(me, None)
    return list(procs.values())


def _cmd(p: psutil.Process) -> str:
    try:
        return " ".join(p.cmdline())
    except (psutil.Error, SystemError, OSError):
        return "?"


_UNITS = {"B": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def footprints() -> dict:
    """pid -> physical footprint in bytes (macOS `top` MEM column).

    On Apple Silicon, Metal/MPS/MLX buffers are part of a process's physical
    footprint but NOT of its RSS, so RSS alone under-counts GPU jobs badly.
    """
    try:
        out = subprocess.check_output(["top", "-l", "1", "-n", "500", "-o", "mem", "-stats", "pid,mem"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:
        return {}
    fp, started = {}, False
    for line in out.splitlines():
        if line.strip().startswith("PID"):
            started = True
            continue
        if started:
            m = re.match(r"\s*(\d+)\s+([\d.]+)([BKMGT])?[+-]?", line)
            if m:
                fp[int(m.group(1))] = int(float(m.group(2)) * _UNITS.get(m.group(3) or "B", 1))
    return fp


def reading(marker: str) -> dict:
    procs = project_processes(marker)
    fp = footprints()
    rss = 0
    top = []
    servers = []
    for p in procs:
        try:
            r = max(p.memory_info().rss, fp.get(p.pid, 0))
        except (psutil.Error, SystemError, OSError):
            continue
        rss += r
        cmd = _cmd(p)
        top.append((r, p.pid, cmd[:160]))
        if "mlx_server_capped" in cmd:
            servers.append({"gb": round(r / GB, 2), "pid": p.pid})
    top.sort(reverse=True)
    vm = psutil.virtual_memory()
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_gb": round(rss / GB, 2),
        "system_used_gb": round((vm.total - vm.available) / GB, 2),
        "system_total_gb": round(vm.total / GB, 2),
        "n_project_procs": len(procs),
        "top": [{"gb": round(r / GB, 2), "pid": pid, "cmd": c} for r, pid, c in top[:6]],
        "servers": servers,
    }


def _throttle_path() -> Path:
    return RUN_DIR / "throttle.json"


def read_throttle() -> dict:
    try:
        return json.loads(_throttle_path().read_text())
    except (OSError, ValueError):
        return {"level": 0, "events": 0}


def bump_throttle(reason: str) -> int:
    t = read_throttle()
    t["level"] = min(int(t.get("level", 0)) + 1, 4)
    t["events"] = int(t.get("events", 0)) + 1
    t["last_reason"] = reason
    t["last_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _throttle_path().write_text(json.dumps(t, indent=2))
    return t["level"]


def relax_throttle(reason: str) -> int:
    """Lower the throttle level by one (applies to jobs started afterwards)."""
    t = read_throttle()
    t["level"] = max(int(t.get("level", 0)) - 1, 0)
    t["last_relax"] = time.strftime("%Y-%m-%d %H:%M:%S")
    t["last_relax_reason"] = reason
    _throttle_path().write_text(json.dumps(t, indent=2))
    return t["level"]


def stop_jobs(marker: str, grace: float = 20.0) -> list[int]:
    victims = [p for p in project_processes(marker)
               if not any(m in _cmd(p) for m in EXEMPT_MARKERS)]
    for p in victims:
        try:
            p.send_signal(signal.SIGTERM)
        except psutil.Error:
            pass
    gone, alive = psutil.wait_procs(victims, timeout=grace)
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    return [p.pid for p in victims]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-limit-gb", type=float, default=200.0)
    ap.add_argument("--system-limit-gb", type=float, default=490.0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--marker", default="ManuscriptR2")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--server-cap-gb", type=float, default=60.0,
                    help="restart only the MLX server (its keep-alive loop brings it back) when its footprint "
                         "exceeds this; its prompt/KV caches grow over hours of serving")
    ap.add_argument("--relax-after-min", type=float, default=45.0,
                    help="lower the throttle level by one after this long without a breach while the "
                         "project stays below 75%% of its limit and the system below its limit minus 50 GB")
    args = ap.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "pids").mkdir(exist_ok=True)
    if args.once:
        print(json.dumps(reading(args.marker), indent=2))
        return

    (RUN_DIR / "pids" / "memguard.pid").write_text(str(os.getpid()))
    log = RUN_DIR / "memguard_events.jsonl"
    heartbeat = RUN_DIR / "memguard_last.json"
    calm_since = time.time()
    while True:
        try:
            r = reading(args.marker)
        except Exception as exc:  # noqa: BLE001 -- never let a measurement glitch stop the guard
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] reading failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.interval)
            continue
        heartbeat.write_text(json.dumps(r, indent=2))
        for srv in r.get("servers", []):
            if srv["gb"] >= args.server_cap_gb:
                try:
                    os.kill(srv["pid"], signal.SIGTERM)
                except OSError:
                    pass
                event = {**r, "event": "SERVER_RESTART", "reason": f"MLX server {srv['gb']} GB >= {args.server_cap_gb} GB",
                         "pid": srv["pid"]}
                with log.open("a") as fh:
                    fh.write(json.dumps(event) + "\n")
                print(json.dumps(event), flush=True)
        # calm = the headroom job_supervisor.py requires to (re)start a job (limits minus 50 GB), with the
        # project additionally below 75% of its limit
        calm = (r["project_gb"] < min(0.75 * args.project_limit_gb, args.project_limit_gb - 50)
                and r["system_used_gb"] < args.system_limit_gb - 50)
        if not calm:
            calm_since = time.time()
        elif (int(read_throttle().get("level", 0)) > 0
              and time.time() - calm_since >= 60 * args.relax_after_min):
            why = f"{args.relax_after_min:.0f} min with project < 75% of its limit and system < limit - 50 GB"
            level = relax_throttle(why)
            event = {**r, "event": "RELAX", "reason": why, "new_level": level}
            with log.open("a") as fh:
                fh.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            calm_since = time.time()
        reason = None
        if r["project_gb"] >= args.project_limit_gb:
            reason = f"project {r['project_gb']} GB >= {args.project_limit_gb} GB"
        elif r["system_used_gb"] >= args.system_limit_gb:
            reason = f"system {r['system_used_gb']} GB >= {args.system_limit_gb} GB"
        if reason:
            level = bump_throttle(reason)
            killed = stop_jobs(args.marker)
            event = {**r, "event": "STOP", "reason": reason, "new_level": level, "killed": killed}
            with log.open("a") as fh:
                fh.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
            time.sleep(30)  # let memory drain before measuring again
            calm_since = time.time()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
