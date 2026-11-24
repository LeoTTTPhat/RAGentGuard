"""Score texts with every content-evidence detector once and cache the scores.

Scores are keyed by the SHA-1 of the text and stored in
output/detector_scores.jsonl ({"h": sha1, "regex": s, "semantic": s,
"classifier": s}). Evaluation scripts only read this cache, so the detectors
(which share the GPU with generation jobs) run once per unique text.
"""
from __future__ import annotations

import hashlib
import sys
from typing import Dict, Iterable, List, Sequence

from .io import OUTPUT_ROOT, append_jsonl, read_jsonl

CACHE = OUTPUT_ROOT / "detector_scores.jsonl"
DETECTORS = ("regex", "semantic", "classifier")


def sha(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def load_cache() -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in read_jsonl(CACHE):
        out.setdefault(r["h"], {}).update({k: v for k, v in r.items() if k != "h"})
    return out


_DET = {}


def _detector(name: str):
    sys.path.insert(0, str(OUTPUT_ROOT.parent / "src"))
    from ragentguard.detectors import ClassifierDetector, KeywordDetector, RegexDetector, SemanticDetector
    if name not in _DET:
        factories = {
            "regex": RegexDetector, "semantic": SemanticDetector, "keyword": KeywordDetector,
            "classifier": ClassifierDetector,  # protectai/deberta-v3-base-prompt-injection-v2 (LLM Guard model)
            "testsavant": lambda: ClassifierDetector("testsavantai/prompt-injection-defender-tiny-v0", name="testsavant"),
            "deepset": lambda: ClassifierDetector("deepset/deberta-v3-base-injection", name="deepset"),
            "fmops": lambda: ClassifierDetector("fmops/distilbert-prompt-injection", injection_label="LABEL_1",
                                                name="fmops"),
            "nemo": lambda: __import__("ragentguard.detectors", fromlist=["NemoYaraDetector"]).NemoYaraDetector(
                str(OUTPUT_ROOT.parent / "data" / "ext" / "nemo_rules")),
        }
        _DET[name] = factories[name]()
    return _DET[name]


def ensure_scores(texts: Iterable[str], detectors: Sequence[str] = DETECTORS,
                  batch: int = 256) -> Dict[str, Dict[str, float]]:
    cache = load_cache()
    uniq: Dict[str, str] = {}
    for t in texts:
        uniq.setdefault(sha(t), t)
    for name in detectors:
        todo = [(h, t) for h, t in uniq.items() if name not in cache.get(h, {})]
        if not todo:
            continue
        det = _detector(name)
        print(f"[detect_cache] scoring {len(todo)} texts with {name}", flush=True)
        for s in range(0, len(todo), batch):
            part = todo[s:s + batch]
            scores = det.score_many([t for _, t in part])
            rows = [{"h": h, name: round(float(sc), 6)} for (h, _), sc in zip(part, scores)]
            append_jsonl(CACHE, rows)
            for r in rows:
                cache.setdefault(r["h"], {})[name] = r[name]
    return cache


def scores_for(texts: Sequence[str], cache: Dict[str, Dict[str, float]], name: str) -> List[float]:
    return [cache[sha(t)][name] for t in texts]
