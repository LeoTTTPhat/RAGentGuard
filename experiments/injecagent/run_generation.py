"""Generate agent outputs for InjecAgent with a local MLX model (resumable).

Example:
  python -m experiments.injecagent.run_generation --model Qwen/Qwen2.5-7B-Instruct \
      --tag qwen2.5-7b --variant none

Output: output/injecagent/<tag>/<variant>/gen.jsonl, one record per
(attack, setting, case index, step). Data-stealing (ds) cases whose first step
succeeds get a second step whose scratchpad contains the attacker tool call and
its simulated response, exactly as in the original benchmark; the simulated
response is looked up in the benchmark's cache (exact parameters, else the
nearest cached parameters of the same tool).
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional, Tuple

from ..common.io import OUTPUT_ROOT, append_jsonl, read_jsonl, workers_from_env
from ..common.mlx_engine import GenRequest, MLXEngine
from . import data
from .parsing import evaluate_output, first_action


def _record(req_meta: Dict, text: str, prompt_tokens: int, gen_tokens: int, seconds: float,
            model: str, variant: str) -> Dict:
    item = req_meta["item"]
    step = req_meta["step"]
    target = item["Attacker Tools"][0] if step == 1 else item["Attacker Tools"][1]
    ev = evaluate_output(text, target, item["Attacker Tools"], item["User Tool"], item["Attacker Instruction"])
    rec = {
        "key": req_meta["key"], "attack": req_meta["attack"], "setting": req_meta["setting"],
        "idx": req_meta["idx"], "step": step, "variant": variant, "model": model,
        "output": text, "eval": ev["eval"], "invalid_reason": ev.get("invalid_reason"),
        "attacker_params": ev.get("Attacker Tool Parameters"),
        "attacker_thought": ev.get("Attacker Tool Thought"),
        "first_action": first_action(text),
        "prompt_tokens": prompt_tokens, "gen_tokens": gen_tokens,
        "batch_seconds": round(seconds, 2),
    }
    if "sim_match" in req_meta:
        rec["sim_match"] = req_meta["sim_match"]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--variant", default="none",
                    choices=["none", "spotlight", "sandwich", "masked", "adaptive", "adaptive_masked", "clean"])
    ap.add_argument("--settings", nargs="+", default=["base", "enhanced"])
    ap.add_argument("--attacks", nargs="+", default=["dh", "ds"])
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of case indices per attack")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--chunk", type=int, default=16, help="case indices per generation batch")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--only-keys", default=None, help="JSON list of step-1 keys to (re)generate")
    ap.add_argument("--trajectory-from", default=None,
                    help="gen.jsonl of the 'none' run; supplies step-1 attacker calls for masked step 2")
    args = ap.parse_args()

    out_path = OUTPUT_ROOT / "injecagent" / args.tag / args.variant / "gen.jsonl"
    existing = {r["key"]: r for r in read_jsonl(out_path)}
    only = set(json.load(open(args.only_keys))) if args.only_keys else None
    traj = {r["key"]: r for r in read_jsonl(args.trajectory_from)} if args.trajectory_from else {}

    engine = MLXEngine(args.model, stop_strings=["Observation:"])
    batch = workers_from_env(args.chunk)
    print(f"[{args.tag}/{args.variant}] model loaded in {engine.load_seconds:.1f}s; chunk={batch}", flush=True)

    cases = {(a, s): data.load_cases(a, s) for a in args.attacks for s in args.settings}
    if args.variant.startswith("adaptive"):
        cases = {k: [data.adaptive_item(it) for it in v] for k, v in cases.items()}
    if args.variant == "clean":  # step 1 only: does the attacker call survive removing the injection?
        cases = {k: [data.clean_item(it) for it in v] for k, v in cases.items()}
    prompt_variant = {"adaptive": "none", "adaptive_masked": "masked", "clean": "none"}.get(args.variant, args.variant)

    # ---------------- step 1 ---------------- #
    for attack in args.attacks:
        n = len(cases[(attack, args.settings[0])])
        indices = data.sample_indices(n, args.frac, args.seed + (0 if attack == "dh" else 1))
        todo: List[Tuple[int, str]] = []
        for idx in indices:
            for setting in args.settings:
                key = f"{attack}|{setting}|{idx}|s1"
                if key in existing or (only is not None and key not in only):
                    continue
                todo.append((idx, setting))
        print(f"[{args.tag}/{args.variant}] {attack} step1 todo={len(todo)}", flush=True)
        by_idx: Dict[int, List[str]] = {}
        for idx, setting in todo:
            by_idx.setdefault(idx, []).append(setting)
        idx_list = sorted(by_idx)
        for start in range(0, len(idx_list), batch):
            reqs = []
            for idx in idx_list[start:start + batch]:
                for setting in by_idx[idx]:
                    item = cases[(attack, setting)][idx]
                    key = f"{attack}|{setting}|{idx}|s1"
                    reqs.append(GenRequest(
                        key=key, messages=data.build_messages(item, prompt_variant, step_2=False),
                        max_tokens=args.max_tokens, share_group=f"{attack}|{idx}|s1",
                        meta={"key": key, "attack": attack, "setting": setting, "idx": idx,
                              "step": 1, "item": item}))
            t0 = time.time()
            results = engine.generate(reqs, completion_batch_size=2 * batch, prefill_batch_size=min(8, 2 * batch))
            dt = time.time() - t0
            recs = [_record(r.meta, r.text, r.prompt_tokens, r.gen_tokens, dt, args.model, args.variant)
                    for r in results]
            append_jsonl(out_path, recs)
            for rec in recs:
                existing[rec["key"]] = rec
            print(f"[{args.tag}/{args.variant}] {attack} step1 {start + len(reqs)//max(1,len(args.settings))}/"
                  f"{len(idx_list)} idx done ({dt:.1f}s, {len(reqs)} prompts)", flush=True)

    # ---------------- step 2 (data stealing) ---------------- #
    if "ds" in args.attacks and args.variant != "clean":
        reqs_all = []
        for setting in args.settings:
            for idx, item in enumerate(cases[("ds", setting)]):
                k1 = f"ds|{setting}|{idx}|s1"
                k2 = f"ds|{setting}|{idx}|s2"
                if k2 in existing:
                    continue
                src: Optional[Dict] = traj.get(k1) if args.variant.endswith("masked") else existing.get(k1)
                if src is None or src.get("eval") != "succ":
                    continue
                if args.variant.endswith("masked") and (only is not None and k1 not in only):
                    continue
                item2 = dict(item)
                item2["Attacker Tool Parameters"] = src["attacker_params"] or "{}"
                item2["Attacker Tool Thought"] = src["attacker_thought"] or ""
                resp, match = data.simulated_response(item["Attacker Tools"][0], item2["Attacker Tool Parameters"])
                item2["Attacker Tool Response"] = resp
                reqs_all.append(GenRequest(
                    key=k2, messages=data.build_messages(item2, prompt_variant, step_2=True),
                    max_tokens=args.max_tokens, share_group=f"ds|{idx}|s2",
                    meta={"key": k2, "attack": "ds", "setting": setting, "idx": idx, "step": 2,
                          "item": item2, "sim_match": match}))
        print(f"[{args.tag}/{args.variant}] ds step2 todo={len(reqs_all)}", flush=True)
        for start in range(0, len(reqs_all), 2 * batch):
            reqs = reqs_all[start:start + 2 * batch]
            t0 = time.time()
            results = engine.generate(reqs, completion_batch_size=2 * batch, prefill_batch_size=min(8, 2 * batch))
            dt = time.time() - t0
            recs = [_record(r.meta, r.text, r.prompt_tokens, r.gen_tokens, dt, args.model, args.variant)
                    for r in results]
            append_jsonl(out_path, recs)
            print(f"[{args.tag}/{args.variant}] ds step2 {start + len(reqs)}/{len(reqs_all)} ({dt:.1f}s)", flush=True)
    print(f"[{args.tag}/{args.variant}] DONE", flush=True)


if __name__ == "__main__":
    main()
