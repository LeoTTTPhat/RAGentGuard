"""Score the benign pool and the attack corpora with all detectors (cached)."""
from __future__ import annotations

import json
import sys

from ..common.detect_cache import ensure_scores
from ..common.io import DATA_ROOT, OUTPUT_ROOT, read_jsonl
from ..injecagent import data as ia


def attack_texts():
    out = []
    for attack in ("dh", "ds"):
        for setting in ("base", "enhanced"):
            out += [c["Tool Response"] for c in ia.load_cases(attack, setting)]
    raw = json.loads((ia.ROOT / "data/attacker_simulated_responses.json").read_text())
    out += list(raw.values())
    sys.path.insert(0, str(OUTPUT_ROOT.parent / "src"))
    from ragentguard.attacks.corpus import (adaptive_redteam_attack_set, human_redteam_attack_set,
                                            independent_annotated_redteam_attack_set)
    for fn in (human_redteam_attack_set, adaptive_redteam_attack_set, independent_annotated_redteam_attack_set):
        out += [d.text for d in fn()]
    return out


def main():
    benign = [r["text"] for r in read_jsonl(DATA_ROOT / "benign_pool.jsonl")]
    texts = benign + attack_texts()
    extra = sys.argv[1:]  # optional extra jsonl files with a "text" field
    for path in extra:
        texts += [r["text"] for r in read_jsonl(path)]
    ensure_scores(texts)
    print("scored", len(texts), "texts")


if __name__ == "__main__":
    main()
