"""Matched-budget comparison of content detectors (guardrail baselines and RAGentGuard's
evidence detectors) with repeated calibration splits.

Protocol (identical for every detector):
  1. benign texts = benign pool (6 categories) + AgentDojo benign tool outputs;
  2. R = 10 random stratified 50/50 dev/test splits of the benign texts;
  3. on each split, the threshold is the smallest value whose dev FPR <= budget
     (1% or 5%); attacks are never used for tuning;
  4. report held-out benign FPR and TPR on five attack sets, mean and SD across
     splits, and a Wilson 95% CI on pooled counts.
Untuned reference points use each tool's default threshold.

  python -m experiments.calibration.baselines
Output: output/calibration/baselines.json
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from typing import Dict, List

from ..common.detect_cache import ensure_scores, sha
from ..common.io import DATA_ROOT, OUTPUT_ROOT, read_jsonl, write_json
from ..common.stats import wilson
from ..injecagent import data as ia

sys.path.insert(0, str(OUTPUT_ROOT.parent / "src"))
from ragentguard.detectors import calibrate_threshold  # noqa: E402

DETECTORS = ["regex", "keyword", "semantic", "classifier", "testsavant", "deepset", "fmops", "nemo"]
LABELS = {"regex": "StruQ-like regex", "keyword": "Keyword count", "semantic": "Semantic prototypes",
          "classifier": "LLM Guard (ProtectAI DeBERTa-v3 v2)", "testsavant": "TestSavantAI tiny",
          "deepset": "deepset DeBERTa-v3", "fmops": "fmops DistilBERT", "nemo": "NeMo Guardrails injection rules"}
DEFAULTS = {"regex": 0.5, "keyword": 1.0, "semantic": 0.30, "classifier": 0.92, "testsavant": 0.5,
            "deepset": 0.5, "fmops": 0.5, "nemo": 15}  # nemo: bitmask of rule families (15 = stock rail)


def nemo_subset(dev_scores, budget):
    """Most permissive rule-family subset (bitmask) whose dev FPR <= budget."""
    best, best_key = 0, (-1, 0.0)
    n = max(1, len(dev_scores))
    for mask in range(16):
        fpr = sum((int(sc) & mask) != 0 for sc in dev_scores) / n
        if fpr <= budget:
            key = (bin(mask).count("1"), -fpr)
            if key > best_key:
                best, best_key = mask, key
    return best
R = 10


def attack_sets() -> Dict[str, List[str]]:
    sets = {}
    for setting in ("base", "enhanced"):
        sets[f"InjecAgent-{setting}"] = [c["Tool Response"] for a in ("dh", "ds") for c in ia.load_cases(a, setting)]
    rag = json.loads((OUTPUT_ROOT / "rag" / "cases.json").read_text())
    sets["BIPIA e-mail"] = [c["poisoned_text"] for c in rag["cases"] if c["attack"]]
    gt = read_jsonl(OUTPUT_ROOT / "agentdojo" / "gt_texts.jsonl")
    sets["AgentDojo tool output"] = [r["text"] for r in gt if r["kind"] == "attack"]
    sys.path.insert(0, str(OUTPUT_ROOT.parent / "src"))
    from ragentguard.attacks.corpus import (adaptive_redteam_attack_set, human_redteam_attack_set,
                                            independent_annotated_redteam_attack_set)
    sets["R1 in-study (280)"] = [d.text for fn in (human_redteam_attack_set, adaptive_redteam_attack_set,
                                                   independent_annotated_redteam_attack_set) for d in fn()]
    return sets


def main():
    benign = [{"text": r["text"], "category": r["category"]} for r in read_jsonl(DATA_ROOT / "benign_pool.jsonl")]
    benign += [{"text": r["text"], "category": "agent_tool_output"}
               for r in read_jsonl(OUTPUT_ROOT / "agentdojo" / "gt_texts.jsonl") if r["kind"] == "benign"]
    attacks = attack_sets()
    texts = [b["text"] for b in benign] + [t for v in attacks.values() for t in v]
    scores = ensure_scores(texts, DETECTORS)
    cats = sorted({b["category"] for b in benign})
    by_cat = {c: [b for b in benign if b["category"] == c] for c in cats}
    out: Dict[str, Dict] = {"_meta": {"n_benign": len(benign), "categories": {c: len(v) for c, v in by_cat.items()},
                                      "attack_sets": {k: len(v) for k, v in attacks.items()}, "R": R}}
    for det in DETECTORS:
        if det == "nemo":
            S = lambda t: scores[sha(t)]["nemo"]
            fire = lambda sc, thr: (int(sc) & int(thr)) != 0
        else:
            S = lambda t: scores[sha(t)][det]
            fire = lambda sc, thr: sc >= thr
        res = {"label": LABELS[det]}
        for bname, budget in (("fpr_0.01", 0.01), ("fpr_0.05", 0.05), ("default", None)):
            reps = []
            for r in range(R):
                rng = random.Random(1000 + r)
                dev, test = [], []
                for c, items in by_cat.items():
                    idx = list(range(len(items)))
                    rng.shuffle(idx)
                    h = len(idx) // 2
                    dev += [items[i] for i in idx[:h]]
                    test += [items[i] for i in idx[h:]]
                if budget is None:
                    thr = DEFAULTS[det]
                elif det == "regex":
                    thr = 0.5
                elif det == "nemo":
                    thr = nemo_subset([S(b["text"]) for b in dev], budget)
                else:
                    thr = calibrate_threshold([S(b["text"]) for b in dev], budget)
                rec = {"threshold": thr,
                       "fpr": sum(fire(S(b["text"]), thr) for b in test) / len(test),
                       "fpr_cat": {c: sum(fire(S(b["text"]), thr) for b in test if b["category"] == c) /
                                   max(1, sum(1 for b in test if b["category"] == c)) for c in cats},
                       "tpr": {k: sum(fire(S(t), thr) for t in v) / len(v) for k, v in attacks.items()},
                       "k_fp": sum(fire(S(b["text"]), thr) for b in test), "n_test": len(test),
                       "k_tp": {k: sum(fire(S(t), thr) for t in v) for k, v in attacks.items()}}
                reps.append(rec)
                if budget is None:
                    break
            agg = {
                "threshold_mean": statistics.mean(x["threshold"] for x in reps),
                "fpr_mean": statistics.mean(x["fpr"] for x in reps),
                "fpr_sd": statistics.pstdev(x["fpr"] for x in reps),
                "fpr_ci_pooled": wilson(sum(x["k_fp"] for x in reps), sum(x["n_test"] for x in reps)),
                "fpr_cat_mean": {c: statistics.mean(x["fpr_cat"][c] for x in reps) for c in cats},
                "tpr_mean": {k: statistics.mean(x["tpr"][k] for x in reps) for k in attacks},
                "tpr_sd": {k: statistics.pstdev(x["tpr"][k] for x in reps) for k in attacks},
                "tpr_ci_pooled": {k: wilson(sum(x["k_tp"][k] for x in reps), len(attacks[k]) * len(reps))
                                  for k in attacks},
                "n_reps": len(reps),
            }
            res[bname] = agg
        out[det] = res
        print(f"{LABELS[det]:40s}", " | ".join(
            f"{b}: FPR {out[det][b]['fpr_mean']:.3f} " + " ".join(f"{k.split()[0][:10]}={v:.2f}" for k, v in out[det][b]['tpr_mean'].items())
            for b in ("fpr_0.01", "fpr_0.05", "default")), flush=True)
    write_json(OUTPUT_ROOT / "calibration" / "baselines.json", out)


if __name__ == "__main__":
    main()
