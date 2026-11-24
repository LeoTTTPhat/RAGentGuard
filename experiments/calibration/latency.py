"""Monitor overhead (P5): per-item latency of each detector and of attribution + gate.

Measured on CPU with batch size 1 on 300 real tool outputs (AgentDojo ground-truth
outputs and InjecAgent tool responses). Output: output/latency.json
"""
from __future__ import annotations

import random
import statistics
import sys
import time

from ..common.io import OUTPUT_ROOT, read_jsonl, write_json
from ..injecagent import data as ia

sys.path.insert(0, str(OUTPUT_ROOT.parent / "src"))
from ragentguard.core.config import PolicyConfig  # noqa: E402
from ragentguard.core.policy import ToolGatePolicy  # noqa: E402
from ragentguard.core.provenance import TrustLevel  # noqa: E402
from ragentguard.core.segments import Segment, SegmentKind  # noqa: E402
from ragentguard.detectors import ClassifierDetector, RegexDetector, SemanticDetector  # noqa: E402
from ragentguard.pipeline.attribution import attribute_tool_call  # noqa: E402


def stats(ms):
    ms = sorted(ms)
    return {"n": len(ms), "mean": statistics.mean(ms), "p50": ms[len(ms) // 2], "p95": ms[int(0.95 * (len(ms) - 1))]}


def main():
    rng = random.Random(7)
    gt = [r["text"] for r in read_jsonl(OUTPUT_ROOT / "agentdojo" / "gt_texts.jsonl")]
    ia_txt = [c["Tool Response"] for c in ia.load_cases("dh", "base")]
    texts = rng.sample(gt, min(150, len(gt))) + rng.sample(ia_txt, 150)
    out = {"chars_mean": statistics.mean(len(t) for t in texts)}
    for name, det in (("regex", RegexDetector()), ("semantic", SemanticDetector(device="cpu")),
                      ("classifier", ClassifierDetector(device="cpu"))):
        det.score(texts[0])  # warm-up
        ms = []
        for t in texts:
            t0 = time.perf_counter()
            det.score(t)
            ms.append((time.perf_counter() - t0) * 1000)
        out[name] = stats(ms)
    pol = ToolGatePolicy(PolicyConfig())
    ms = []
    for i, t in enumerate(texts):
        segs = [Segment("You are a helpful assistant.", SegmentKind.SYSTEM, TrustLevel.SYSTEM),
                Segment("Please pay my bill and summarise my mail.", SegmentKind.USER, TrustLevel.HIGH),
                Segment(t, SegmentKind.TOOL_OUTPUT, TrustLevel.UNTRUSTED),
                Segment(texts[(i + 1) % len(texts)], SegmentKind.TOOL_OUTPUT, TrustLevel.UNTRUSTED)]
        args = {"recipient": "US133000000121212121212", "amount": 100.0, "subject": t[:60]}
        t0 = time.perf_counter()
        att = attribute_tool_call("send_money", args, segs)
        pol.decide("send_money", args, att, segs)
        ms.append((time.perf_counter() - t0) * 1000)
    out["attribution_plus_gate"] = stats(ms)
    write_json(OUTPUT_ROOT / "latency.json", out)
    print(out)


if __name__ == "__main__":
    main()
