"""Keys needing the masked (task-neutral) re-run for the S6 interventional check:
step-1 keys whose no-defense output called the attacker tool (DH and DS); for DS the
masked step 2 is generated from the no-defense step-1 trajectory."""
from __future__ import annotations

import json
import sys

from ..common.io import OUTPUT_ROOT, read_jsonl

tag = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "none"
recs = read_jsonl(OUTPUT_ROOT / "injecagent" / tag / variant / "gen.jsonl")
keys = sorted({r["key"] for r in recs if r["step"] == 1 and r["eval"] == "succ"})
suffix = "" if variant == "none" else f"_{variant}"
out = OUTPUT_ROOT.parent / "run" / f"masked_keys_{tag}{suffix}.json"
out.write_text(json.dumps(keys))
print(tag, len(keys), "->", out)
