# RAGentGuard

Cross-stage provenance and sink-level containment of content-induced faults in RAG agents.

Release v0.2.2 accompanies the manuscript *Containing Content-Induced Faults in Retrieval-Augmented
Generation Agents through Cross-Stage Provenance* (under review). It contains the library, every
experiment script, and the raw outputs from which the manuscript's tables and numbers are generated.

RAGentGuard is a runtime monitor for retrieval-augmented generation agents. Up to v0.1.0 it was
called *RAGuard*; we renamed it to avoid confusion with the unrelated RAGuard of Walker et al.
(IMBSA 2025). The monitor

1. labels every piece of content with an integrity level when it enters the agent (documents,
   web pages, e-mails, tool results fed back into the loop, memory items, messages from other
   agents) and gives derived content the least trusted label of its sources;
2. scores content with any pluggable detector (structural rules, embedding prototypes, or a
   fine-tuned classifier) and stores the result as *evidence* on the segment;
3. attributes each proposed tool call (argument values and the choice of action) and each
   answer's actionable spans to the segments they came from (stage S6), and
4. gates side-effecting tool calls at dispatch (stage S7) using that attribution: block when an
   attributed untrusted source carries evidence; resolve evidence-free tainted calls with an
   interventional check (re-run the step without the user's task; MELON-style) or a review policy.

```
S1 Ingestion -> S2 Chunk/Embed -> S3 Store/Memory -> S4 Retrieval -> S5 Context (evidence)
      ^                ^                                                   |
      |                +---- memory write-back (label = meet of sources) --+-- S6 Generation (attribution)
      +------------------- tool output re-enters as a new labelled segment --- S7 Tool dispatch (gate)
```

## Changes since v0.1.0

* S6 is part of the execution flow: `RAGentGuard.process_generation()`
  (`src/ragentguard/ragentguard.py`) runs attribution on every generation, records it in the audit
  trail, and passes argument/action provenance to the S7 gate (`ToolGatePolicy`,
  `src/ragentguard/core/policy.py`), which raises `ToolCallBlockedError` on a block. Unit tests in
  `tests/test_r2_integration.py` show that changing only the provenance of an argument changes the
  dispatch decision.
* Cyclic workflows are supported: `src/ragentguard/agent/session.py` (`AgentSession`) applies the stages per
  event (tool feedback, memory write-back/read, inter-agent messages); `core/segments.py` implements
  the label meet and evidence inheritance (no laundering through loops, tested).
* Detectors are pluggable and calibrated under one budget: `src/ragentguard/detectors.py` (rules, keyword count,
  embedding prototypes, fine-tuned classifiers, NeMo Guardrails' injection rules).
* The new evaluation (`experiments/`) scores actions that the models emit. The v0.1.0 harness
  (`src/ragentguard/evaluation/end_to_end.py`) derived tool calls from a deterministic parser over the
  retrieved context, so its results are withdrawn; it is kept, with its outputs in
  `output/r1_experiments/`, only to reproduce the earlier numbers and is not used by the new study.

## Quick check (no GPU, no downloads)

```bash
./reproduce.sh    # unit tests, then every table and number of the manuscript from output/
```

It needs Python 3.10 or later with `pytest` (set `PYTHON=...` to choose the interpreter). The
library and `paper/make_results.py` use only the standard library.

## Install (experiments)

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch transformers sentence-transformers faiss-cpu mlx mlx-lm openai agentdojo \
               datasets scikit-learn scipy pandas numpy statsmodels psutil yara-python pytest
PYTHONPATH=src python -m pytest -q tests
```

`requirements-experiments.txt` lists the exact package versions we used.

Benchmarks: `git clone https://github.com/uiuc-kang-lab/InjecAgent data/ext/InjecAgent` and
`git clone https://github.com/microsoft/BIPIA data/ext/BIPIA` (both MIT); AgentDojo is installed
from PyPI (v1.2.1 suites). The NeMo Guardrails detector reads the four YARA rule files (`code`, `sqli`,
`template`, `xss`) of NeMo Guardrails' injection-detection rail (Apache-2.0) from
`data/ext/nemo_rules/`. Models are local (Apple Silicon/MLX): Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, Qwen2.5-32B/72B-Instruct (4-bit MLX community builds).

## Reproducing the experiments

All runners are resumable (one JSON line per finished unit; re-running a command continues where it
stopped). We launched them through `scripts/job_supervisor.py` with `scripts/memguard.py` running: the
guard stops all jobs if this project's memory footprint reaches 200 GB or the machine reaches 490 GB
in use, and the supervisor restarts them slowly with fewer workers (`WORKERS` environment variable).
Both find the project's processes by a path marker (`--marker`, default `ManuscriptR2`, the folder name
we used; pass the name of your checkout's folder). Long jobs were started detached from the terminal
with `scripts/detach.sh`. Our machine-specific job queues are not included; the table lists every
command they ran.

| Study | Generate | Evaluate |
|---|---|---|
| Detector calibration (8 detectors, 10 splits) | `python -m experiments.calibration.benign_pool`; `python -m experiments.agentdojo_exp.gt_texts`; `python -m experiments.calibration.score_all` | `python -m experiments.calibration.calibrate`; `python -m experiments.calibration.baselines`; `python -m experiments.calibration.latency` |
| InjecAgent (1,054 cases x 2 settings) | `python -m experiments.injecagent.run_generation --model <hf-id> --tag <tag> --variant none [--frac 0.5 / 0.2]`; masked re-runs: `python -m experiments.injecagent.masked_keys <tag>` then `--variant masked --only-keys run/masked_keys_<tag>.json --trajectory-from output/injecagent/<tag>/none/gen.jsonl`; attribution-aware attacker: `--variant adaptive --settings base` (+ `adaptive_masked`); interventional test of attribution (injected text removed): `--variant clean --only-keys run/masked_keys_<tag>.json` | `python -m experiments.injecagent.evaluate --tags qwen2.5-7b llama3.1-8b qwen2.5-32b qwen2.5-72b` |
| AgentDojo v1.2.1 (97 user tasks + 25 attacked pairs per suite) | `scripts/serve_mlx_forever.sh Qwen/Qwen2.5-7B-Instruct 8090 16`; `python -m experiments.agentdojo_exp.run --port 8090 --model-id Qwen/Qwen2.5-7B-Instruct --model-name Qwen --tag qwen2.5-7b --per-suite 25 --configs none --shadow`; then `--rerun-from-shadow --configs rg_attr_verify rg_attr_block rg_maxtaint s5_filter pi_detector rg_attr_verify_t05` | `python -m experiments.agentdojo_exp.evaluate --tag qwen2.5-7b` |
| RAG answer-level (BIPIA) + interventions | `python -m experiments.rag.run build`; `python -m experiments.rag.run retrieve`; `python -m experiments.rag.run generate --model <hf-id> --tag <tag> --phase A` (phase B = leave-one-out / clean replacement); `python -m experiments.rag.judge --tag <tag>` | `python -m experiments.rag.evaluate --tags qwen2.5-7b qwen2.5-32b` |
| Cross-session memory poisoning | `python -m experiments.memory.run --model <hf-id> --tag <tag> [--channel message]` | `python -m experiments.memory.evaluate --tags qwen2.5-7b qwen2.5-32b [--channel message]` |

Models: `Qwen/Qwen2.5-7B-Instruct` (tag `qwen2.5-7b`), `meta-llama/Llama-3.1-8B-Instruct` (`llama3.1-8b`),
`mlx-community/Qwen2.5-32B-Instruct-4bit` (`qwen2.5-32b`, InjecAgent `--frac 0.5`) and
`mlx-community/Qwen2.5-72B-Instruct-4bit` (`qwen2.5-72b`, `--frac 0.2`; the 20% sample is a subset of the
50% sample because both use the same seed). The manuscript's tables, figures and every number in its
text are generated from `output/` by `paper/make_results.py`, which writes them to `paper/generated/`.

## Layout

```
src/ragentguard/
  core/        provenance.py, segments.py (labels, meet), policy.py (rules + ToolGatePolicy), config.py
  pipeline/    ingestion.py, retrieval.py, attribution.py (S6: chunk, argument, action, answer provenance)
  monitors/    context_monitor.py, tool_monitor.py
  agent/       session.py (AgentSession: cyclic workflows)
  detectors.py regex / keyword / semantic / classifier detectors and calibrate_threshold()
  audit/       reporter.py (violations + S6/S7 stage events)
experiments/   injecagent/, agentdojo_exp/, rag/, memory/, calibration/, common/
scripts/       memguard.py, job_supervisor.py, detach.sh, serve_mlx_forever.sh (AgentDojo model server);
               llm_guard_baseline.py, nemo_guardrails_baseline.py (v0.1.0 baselines)
paper/         make_results.py (every table and number of the manuscript, from output/)
data/          benign_pool.jsonl (benign calibration texts); ext/ (benchmarks, cloned separately)
output/        raw outputs of every study; r1_experiments/ holds the withdrawn v0.1.0 outputs
```

## Licence and data

MIT (see `LICENSE`). Benchmarks keep their own licences (InjecAgent, BIPIA, AgentDojo: MIT; the
NeMo Guardrails rules: Apache-2.0). `data/benign_pool.jsonl` and the stored outputs contain text drawn
from AG News, WikiText-2, CodeSearchNet, BIPIA, XSTest, OR-Bench and InjecAgent
(see `experiments/calibration/benign_pool.py`); that text remains under the terms of its source.
