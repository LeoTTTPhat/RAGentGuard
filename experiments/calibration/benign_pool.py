"""Build the benign text pool used to calibrate every detector under one budget.

Categories (500 texts each unless the source is smaller), split 50/50 into a
calibration ("dev") half and a held-out ("test") half with a fixed seed:
  news          AG News test split
  encyclopedic  Wikitext-2 test paragraphs (>= 200 characters)
  code          CodeSearchNet (python) functions with docstrings
  email         BIPIA email-QA contexts (train + test), attack-free
  risky         benign-but-alarming text: XSTest safe-prompt completions and
                OR-Bench-hard-1k prompts (security, weapons-sounding, medical...)
  tool_output   InjecAgent tool-response templates whose injection slot is
                filled with a benign AG News sentence
Output: data/benign_pool.jsonl  ({id, category, split, text})
"""
from __future__ import annotations

import json
import random
import re

from datasets import load_dataset

from ..common.io import DATA_ROOT, append_jsonl
from ..injecagent import data as ia

SEED = 2026
N = 500


def _take(texts, n, rng):
    texts = [t for t in texts if t and len(t.strip()) >= 40]
    rng.shuffle(texts)
    return texts[:n]


def build():
    rng = random.Random(SEED)
    pool = {}
    pool["news"] = _take([r["text"] for r in load_dataset("ag_news", split="test")], N, rng)
    wiki = [r["text"].strip() for r in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")]
    pool["encyclopedic"] = _take([t for t in wiki if len(t) >= 200 and not t.startswith("=")], N, rng)
    code = load_dataset("code-search-net/code_search_net", "python", split="test")
    pool["code"] = _take([r["whole_func_string"][:3000] for r in code.select(range(5000))
                          if r["func_documentation_string"]], N, rng)
    emails = []
    for split in ("train", "test"):
        with open(DATA_ROOT / "ext/BIPIA/benchmark/email" / f"{split}.jsonl") as fh:
            emails += [json.loads(l)["context"] for l in fh if l.strip()]
    pool["email"] = _take(emails, N, rng)
    xs = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    xs_txt = [r["completion"] for r in xs if not str(r["type"]).startswith("contrast")
              and r["final_label"] == "1_full_compliance"]
    orb = [r["prompt"] for r in load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")]
    pool["risky"] = _take(xs_txt, N // 2, rng) + _take(orb, N - N // 2, rng)
    sentences = [s for t in pool["news"] for s in re.split(r"(?<=[.!?])\s+", t) if len(s) > 40]
    templates = {}
    for attack in ("dh", "ds"):
        for item in ia.load_cases(attack, "base"):
            templates[item["User Tool"]] = item["Tool Response Template"]
    outs = []
    tlist = sorted(templates.items())
    for i in range(N):
        tool, tmpl = tlist[i % len(tlist)]
        outs.append(tmpl.replace("<Attacker Instruction>", rng.choice(sentences).replace("'", "")))
    pool["tool_output"] = outs

    rows = []
    for cat, texts in pool.items():
        idx = list(range(len(texts)))
        rng.shuffle(idx)
        half = set(idx[: len(idx) // 2])
        for i, t in enumerate(texts):
            rows.append({"id": f"{cat}-{i:04d}", "category": cat,
                         "split": "dev" if i in half else "test", "text": t})
    out = DATA_ROOT / "benign_pool.jsonl"
    if out.exists():
        out.unlink()
    append_jsonl(out, rows)
    print({c: len(t) for c, t in pool.items()}, "->", out)


if __name__ == "__main__":
    build()
