"""Matched-budget threshold calibration for every detector.

Thresholds are chosen on the *dev* half of the benign pool only, so that the
fraction of benign dev texts that fire is <= the budget (1% or 5%). The same
routine is applied to RAGentGuard's evidence detectors and to the external
guardrail baselines. Held-out FPR is then measured on the *test* half.
Output: output/calibration/thresholds.json
"""
from __future__ import annotations

import json
import sys

from ..common.detect_cache import ensure_scores, sha
from ..common.io import DATA_ROOT, OUTPUT_ROOT, read_jsonl, write_json
from ..common.stats import wilson

sys.path.insert(0, str(OUTPUT_ROOT.parent / "src"))
from ragentguard.detectors import calibrate_threshold  # noqa: E402

DETS = ("regex", "semantic", "classifier")
BUDGETS = {"fpr_0.01": 0.01, "fpr_0.05": 0.05}


def main():
    pool = read_jsonl(DATA_ROOT / "benign_pool.jsonl")
    extra = sys.argv[1:]
    for path in extra:  # e.g. AgentDojo benign tool outputs
        pool += read_jsonl(path)
    scores = ensure_scores([r["text"] for r in pool])
    out = {}
    for d in DETS:
        dev = [scores[sha(r["text"])][d] for r in pool if r["split"] == "dev"]
        out[d] = {}
        for bname, b in BUDGETS.items():
            thr = 0.5 if d == "regex" else calibrate_threshold(dev, b)
            out[d][bname] = thr
            per_cat = {}
            for cat in sorted({r["category"] for r in pool}):
                test = [scores[sha(r["text"])][d] for r in pool if r["split"] == "test" and r["category"] == cat]
                k = sum(s >= thr for s in test)
                per_cat[cat] = {"fpr": k / max(1, len(test)), "k": k, "n": len(test), "ci": wilson(k, len(test))}
            test_all = [scores[sha(r["text"])][d] for r in pool if r["split"] == "test"]
            k = sum(s >= thr for s in test_all)
            out[d][bname + "_heldout"] = {"fpr": k / max(1, len(test_all)), "k": k, "n": len(test_all),
                                          "ci": wilson(k, len(test_all)), "per_category": per_cat}
        # default thresholds used by the original tools
        for name, thr in (("default_0.5", 0.5), ("default_0.92", 0.92)):
            test_all = [scores[sha(r["text"])][d] for r in pool if r["split"] == "test"]
            k = sum(s >= thr for s in test_all)
            out[d][name + "_heldout"] = {"fpr": k / max(1, len(test_all)), "k": k, "n": len(test_all)}
    write_json(OUTPUT_ROOT / "calibration" / "thresholds.json", out)
    print(json.dumps({d: {b: out[d][b] for b in BUDGETS} for d in DETS}, indent=1))
    for d in DETS:
        for b in BUDGETS:
            print(d, b, "heldout FPR", round(out[d][b + "_heldout"]["fpr"], 4),
                  {c: round(v["fpr"], 3) for c, v in out[d][b + "_heldout"]["per_category"].items()})


if __name__ == "__main__":
    main()
