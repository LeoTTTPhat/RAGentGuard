"""Generation for the RAG answer-level study (resumable, deduplicated by context).

  python -m experiments.rag.run retrieve
  python -m experiments.rag.run generate --model Qwen/Qwen2.5-7B-Instruct --tag qwen2.5-7b --phase A
  (judge with experiments.rag.judge, then)
  python -m experiments.rag.run generate ... --phase B     # interventions

Phase A contexts per case: base retrieval; S5 redaction of chunks flagged by each
detector (calibrated at 1% benign FPR); optional prompt-level baselines
(spotlight, sandwich). Phase B (needs phase-A judgments): for attacked cases that
succeed under no defense and for benign cases answered correctly, leave-one-out
removal of every retrieved chunk and, for attacked cases, replacement of the
poisoned e-mail by its clean version.
Generations are keyed by sha1(prompt) so identical contexts are generated once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Dict, List

from ..common.detect_cache import ensure_scores, sha
from ..common.io import OUTPUT_ROOT, append_jsonl, read_jsonl, workers_from_env, write_json
from . import cases as C

DETS = {"regex": ("regex",), "semantic": ("semantic",), "classifier": ("classifier",),
        "regex|classifier": ("regex", "classifier")}


def retrieval_path():
    return OUTPUT_ROOT / "rag" / "retrieval.json"


def do_retrieve() -> None:
    data = C.load()
    emails = {d["doc_id"]: d["text"] for d in data["docs"]}
    r = C.Retriever(data["docs"])
    out = {}
    for c in data["cases"]:
        rep = {c["email_id"]: c["poisoned_text"]} if c["attack"] else None
        out[c["case_id"]] = r.retrieve(c["question"], rep)
    write_json(retrieval_path(), out)
    texts = [ch["text"] for v in out.values() for ch in v]
    ensure_scores(texts)
    hit = sum(any(ch["doc_id"] == c["email_id"] for ch in out[c["case_id"]]) for c in data["cases"]) / len(data["cases"])
    print(f"retrieved {len(out)} cases, target-email hit rate {hit:.3f}")


def thresholds(budget="fpr_0.01") -> Dict[str, float]:
    th = json.loads((OUTPUT_ROOT / "calibration" / "thresholds.json").read_text())
    return {"regex": 0.5, "semantic": th["semantic"][budget], "classifier": th["classifier"][budget]}


def flagged(chunk: Dict, dets, scores, th) -> bool:
    s = scores.get(sha(chunk["text"]), {})
    return any(s.get(d, 0.0) >= th[d] for d in dets)


def key_of(messages: List[Dict]) -> str:
    return hashlib.sha1(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def plan_phase_a(prompt_variants: List[str]) -> List[Dict]:
    data = C.load()
    ret = json.loads(retrieval_path().read_text())
    from ..common.detect_cache import load_cache
    scores, th = load_cache(), thresholds()
    reqs = []
    for c in data["cases"]:
        chunks = ret[c["case_id"]]
        reqs.append({"case_id": c["case_id"], "variant": "base", "chunks": [ch["doc_id"] for ch in chunks],
                     "messages": C.prompt(c["question"], chunks)})
        for name, dets in DETS.items():
            kept = [ch for ch in chunks if not flagged(ch, dets, scores, th)]
            if len(kept) != len(chunks):
                reqs.append({"case_id": c["case_id"], "variant": f"redact[{name}]",
                             "chunks": [ch["doc_id"] for ch in kept], "messages": C.prompt(c["question"], kept)})
        for pv in prompt_variants:
            reqs.append({"case_id": c["case_id"], "variant": f"prompt[{pv}]", "chunks": [ch["doc_id"] for ch in chunks],
                         "messages": C.prompt(c["question"], chunks, pv)})
    return reqs


def plan_phase_b(tag: str) -> List[Dict]:
    data = C.load()
    ret = json.loads(retrieval_path().read_text())
    judged = {r["key"]: r for r in read_jsonl(OUTPUT_ROOT / "rag" / tag / "judged.jsonl")}
    gens = {r["case_id"] + "|" + r["variant"]: r for r in read_jsonl(OUTPUT_ROOT / "rag" / tag / "gen.jsonl")}
    emails = {d["doc_id"]: d["text"] for d in data["docs"]}
    reqs = []
    for c in data["cases"]:
        g = gens.get(c["case_id"] + "|base")
        if g is None or g["key"] not in judged:
            continue
        j = judged[g["key"]]
        chunks = ret[c["case_id"]]
        if c["attack"] and j.get("attack_success"):
            pass
        elif not c["attack"] and j.get("correct"):
            pass
        else:
            continue
        for i in range(len(chunks)):
            kept = [ch for k, ch in enumerate(chunks) if k != i]
            reqs.append({"case_id": c["case_id"], "variant": f"loo[{i}]", "chunks": [ch["doc_id"] for ch in kept],
                         "removed": chunks[i]["doc_id"], "messages": C.prompt(c["question"], kept)})
        if c["attack"] and any(ch["poisoned"] for ch in chunks):
            clean = [dict(ch, text=emails[ch["doc_id"]], poisoned=False) if ch["poisoned"] else ch for ch in chunks]
            reqs.append({"case_id": c["case_id"], "variant": "clean_replace", "chunks": [ch["doc_id"] for ch in clean],
                         "messages": C.prompt(c["question"], clean)})
    return reqs


def do_generate(model: str, tag: str, phase: str, prompt_variants: List[str], batch: int) -> None:
    from ..common.mlx_engine import GenRequest, MLXEngine
    out = OUTPUT_ROOT / "rag" / tag / "gen.jsonl"
    have = {r["case_id"] + "|" + r["variant"] for r in read_jsonl(out)}
    by_key = {r["key"]: r for r in read_jsonl(out)}
    reqs = plan_phase_a(prompt_variants) if phase == "A" else plan_phase_b(tag)
    todo = [r for r in reqs if r["case_id"] + "|" + r["variant"] not in have]
    for r in todo:
        r["key"] = key_of(r["messages"])
    # reuse generations for identical prompts
    reuse = [r for r in todo if r["key"] in by_key]
    append_jsonl(out, [{**{k: v for k, v in r.items() if k != "messages"}, "output": by_key[r["key"]]["output"],
                        "reused": True} for r in reuse])
    todo = [r for r in todo if r["key"] not in by_key]
    uniq: Dict[str, Dict] = {}
    for r in todo:
        uniq.setdefault(r["key"], r)
    print(f"[rag/{tag}/phase {phase}] requests={len(reqs)} new={len(todo)} unique prompts={len(uniq)}", flush=True)
    if not uniq:
        return
    eng = MLXEngine(model)
    keys = list(uniq)
    for s in range(0, len(keys), batch):
        part = keys[s:s + batch]
        res = eng.generate([GenRequest(key=k, messages=uniq[k]["messages"], max_tokens=320) for k in part],
                           completion_batch_size=batch, prefill_batch_size=min(8, batch))
        texts = {r.key: r.text for r in res}
        rows = []
        for r in todo:
            if r["key"] in texts:
                rows.append({**{k: v for k, v in r.items() if k != "messages"}, "output": texts[r["key"]]})
        append_jsonl(out, rows)
        todo = [r for r in todo if r["key"] not in texts]
        print(f"[rag/{tag}/phase {phase}] {min(s + batch, len(keys))}/{len(keys)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "retrieve", "generate"])
    ap.add_argument("--model")
    ap.add_argument("--tag")
    ap.add_argument("--phase", default="A")
    ap.add_argument("--prompt-variants", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "build":
        C.build()
    elif a.cmd == "retrieve":
        do_retrieve()
    else:
        do_generate(a.model, a.tag, a.phase, a.prompt_variants, workers_from_env(a.batch))


if __name__ == "__main__":
    main()
