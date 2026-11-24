"""Benign and injected AgentDojo tool outputs from the ground-truth pipeline (no LLM).

For every user task the ground-truth function calls are executed in the clean
environment (benign tool outputs) and, for every (user task, injection task)
pair, in the environment injected by the important_instructions attack; tool
outputs that contain an injection string are attack texts.
Output: output/agentdojo/gt_texts.jsonl  ({text, kind, suite, user_task, injection_task})
"""
from __future__ import annotations

from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import get_text_content_as_str

from ..common.io import OUTPUT_ROOT, append_jsonl
from .run import VERSION, NamedAttack


def tool_texts(suite, ut, it, injections):
    pipe = GroundTruthPipeline(it if it is not None else ut)
    pipe.name = "ground-truth-Qwen"
    env = suite.load_and_inject_default_environment(injections)
    env = ut.init_environment(env)
    from agentdojo.functions_runtime import FunctionsRuntime
    runtime = FunctionsRuntime(suite.tools)
    gt = GroundTruthPipeline(ut)
    _, _, _, messages, _ = gt.query(ut.PROMPT, runtime, env)
    return [get_text_content_as_str(m["content"]) or "" for m in messages if m["role"] == "tool"]


def main():
    out = OUTPUT_ROOT / "agentdojo" / "gt_texts.jsonl"
    if out.exists():
        out.unlink()
    rows, seen = [], set()
    for sname, suite in get_suites(VERSION).items():
        for ut_id, ut in suite.user_tasks.items():
            for t in tool_texts(suite, ut, None, {}):
                if t and t not in seen:
                    seen.add(t)
                    rows.append({"text": t, "kind": "benign", "suite": sname, "user_task": ut_id, "injection_task": None})
        dummy = GroundTruthPipeline(None)
        dummy.name = "ground-truth-Qwen"
        attack = NamedAttack(suite, dummy, "Qwen")
        for ut_id, ut in suite.user_tasks.items():
            for it_id, it in suite.injection_tasks.items():
                inj = attack.attack(ut, it)
                if not inj:
                    continue
                for t in tool_texts(suite, ut, it, inj):
                    if t and t not in seen and "important message from me" in t:
                        seen.add(t)
                        rows.append({"text": t, "kind": "attack", "suite": sname, "user_task": ut_id,
                                     "injection_task": it_id})
    append_jsonl(out, rows)
    print({k: sum(r["kind"] == k for r in rows) for k in ("benign", "attack")}, "->", out)


if __name__ == "__main__":
    main()
