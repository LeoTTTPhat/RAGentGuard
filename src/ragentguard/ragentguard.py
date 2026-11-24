"""
RAGentGuard: main orchestrator.

Provides a single high-level class that wires all seven pipeline stages together.
Use this for end-to-end auditing of a RAG-agent query.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .agent.session import AgentSession
from .audit.reporter import AuditLogger, AuditReporter
from .core.config import RAGentGuardConfig
from .core.policy import ToolGateDecision, ToolGatePolicy, scan_for_injection_patterns
from .core.provenance import PipelineStage, PolicyViolation, TaintVector, TrustLevel
from .core.segments import Segment, SegmentKind, TrustMapping
from .monitors.context_monitor import ContextPolicyMonitor, PolicyBlockedError
from .monitors.tool_monitor import ToolCallMonitor, ToolCallBlockedError
from .pipeline.attribution import AttributionResult, GenerationAttributor, attribute_tool_call
from .pipeline.ingestion import DocumentIngestor
from .pipeline.retrieval import TaintPropagator


@dataclass
class GenerationAudit:
    """Output of `RAGentGuard.process_generation` (S6 -> S7)."""
    attribution: AttributionResult
    tool_decision: Optional[ToolGateDecision] = None


class RAGentGuard:
    """
    End-to-end RAG-agent security auditor.

    Minimal example (no external dependencies):
        guard = RAGentGuard()
        tagged = guard.ingest("adversarial document text", "upload://doc.pdf")
        chunks = guard.retrieve([tagged[0]])     # simulate retrieval
        context, taint = guard.assemble(chunks)
        try:
            guard.check_context(context, chunks, query="user query")
        except PolicyBlockedError as e:
            print("Blocked:", e)
        guard.save_report("./report.html")
    """

    def __init__(self, config: Optional[RAGentGuardConfig] = None):
        self.config = config or RAGentGuardConfig.default()

        self.ingestor = DocumentIngestor(
            trusted_source_ids=self.config.trusted_source_ids,
            provenance_signing_key=self.config.provenance_signing_key,
        )
        self.propagator = TaintPropagator(self.config)
        self.context_monitor = ContextPolicyMonitor(self.config)
        self.tool_monitor = ToolCallMonitor(self.config)
        self.attributor = GenerationAttributor(self.config.attribution)
        self.tool_gate = ToolGatePolicy(self.config.policy)
        self.reporter = AuditReporter()

        self._audit_logger: Optional[AuditLogger] = None
        if self.config.audit_log_path:
            self._audit_logger = AuditLogger(self.config.audit_log_path)

    # ------------------------------------------------------------------ #
    # Stage 1: Ingest                                                      #
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        text: str,
        source_path: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Tag and chunk a document. Returns list of tagged chunk dicts."""
        return self.ingestor.ingest(text, source_path, metadata)

    def ingest_batch(
        self, documents: List[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        return self.ingestor.ingest_batch(documents)

    # ------------------------------------------------------------------ #
    # Stage 2: Retrieve + taint propagation                               #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        tagged_chunks: List[Dict[str, Any]],
    ) -> List[Tuple[str, TaintVector]]:
        """
        Simulate retrieval: given already-tagged chunks, propagate taint.
        In a real deployment, wrap your vector DB retriever instead.
        """
        raw = [{"text": c["text"], "metadata": c["metadata"]} for c in tagged_chunks]
        return self.propagator.propagate(raw)

    # ------------------------------------------------------------------ #
    # Stage 2→3: Assemble context                                         #
    # ------------------------------------------------------------------ #

    def assemble(
        self,
        chunk_taints: List[Tuple[str, TaintVector]],
        separator: str = "\n\n",
    ) -> Tuple[str, TaintVector]:
        """Concatenate chunks and merge taint vectors."""
        return self.propagator.assemble_context(chunk_taints, separator)

    # ------------------------------------------------------------------ #
    # Stage 3: Context policy check                                       #
    # ------------------------------------------------------------------ #

    def check_context(
        self,
        assembled_context: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        query: str = "",
    ) -> Optional[PolicyViolation]:
        """
        Evaluate context before LLM call.
        Raises PolicyBlockedError if blocking_mode=True and threshold exceeded.
        """
        violation = self.context_monitor.check(assembled_context, chunk_taints, query)
        if violation:
            self._record_violation(violation)
            if violation.blocked:
                raise PolicyBlockedError(violation)
        return violation

    # ------------------------------------------------------------------ #
    # Stage 4: Generation attribution                                      #
    # ------------------------------------------------------------------ #

    def attribute(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any = None,
        tokenizer: Any = None,
    ) -> AttributionResult:
        """Map generated text back to source documents (chunk-level S6)."""
        return self.attributor.attribute(generated_text, chunk_taints, model, tokenizer)

    def _segments_from_chunks(
        self, chunk_taints: List[Tuple[str, TaintVector]], query: str = ""
    ) -> List[Segment]:
        segs: List[Segment] = []
        if query:
            segs.append(Segment(query, SegmentKind.USER, TrustLevel.HIGH, "user"))
        for text, tv in chunk_taints:
            trust = min((t.trust_level for t in tv.contributing_tags), key=float,
                        default=TrustLevel.UNTRUSTED)
            patterns = tv.injection_patterns_found or scan_for_injection_patterns(text)
            src = tv.contributing_tags[0].source_path if tv.contributing_tags else ""
            seg = Segment(text, SegmentKind.RETRIEVED, trust, src)
            seg.flagged = bool(patterns)
            seg.evidence = {"regex": 1.0 if patterns else 0.0}
            segs.append(seg)
        return segs

    def process_generation(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        query: str = "",
        tool_call: Optional[Dict[str, Any]] = None,
        model: Any = None,
        tokenizer: Any = None,
        verifier: Any = None,
    ) -> GenerationAudit:
        """
        Stage 6 in the execution flow: attribute the generation, record the
        attribution in the audit trail, and -- if the model emitted a tool call --
        hand the argument/action provenance to the Stage-7 gate.

        Raises ToolCallBlockedError when the gate blocks and blocking_mode=True.
        """
        att = self.attribute(generated_text, chunk_taints, model, tokenizer)
        self.reporter.record_event({"stage": "S6", "event": "generation_attribution", **att.to_dict()})
        if not tool_call:
            return GenerationAudit(att)
        segments = self._segments_from_chunks(chunk_taints, query)
        name, args = tool_call.get("tool", ""), tool_call.get("args", {})
        pc = self.config.policy
        call_att = attribute_tool_call(
            name, args, segments,
            TrustMapping(untrusted_below=pc.untrusted_below, zero_trust=pc.zero_trust),
            fuzzy_min=pc.attribution_fuzzy_min, action_min=pc.action_attribution_min)
        self.reporter.record_event({"stage": "S6", "event": "tool_attribution", **call_att.to_dict()})
        decision = self.tool_gate.decide(name, args, call_att, segments, verifier)
        self.reporter.record_event({"stage": "S7", "event": "tool_decision", "tool": name,
                                    "action": decision.action, "reasons": decision.reasons})
        if decision.action in ("block", "review"):
            taint = att.updated_taint or TaintVector()
            violation = PolicyViolation(
                stage=PipelineStage.TOOL_DISPATCH,
                violation_type=f"tool_call_{decision.action} (tool={name}; {'; '.join(decision.reasons)})",
                taint_vector=taint.advance_stage(PipelineStage.TOOL_DISPATCH),
                context_fraction=call_att.attributed_taint,
                blocked=decision.blocked and pc.blocking_mode,
                query=query,
                tool_call={"tool": name, "args": args, "attribution": call_att.to_dict()},
            )
            self._record_violation(violation)
            if violation.blocked:
                raise ToolCallBlockedError(violation)
        return GenerationAudit(att, decision)

    def session(self, system_prompt: str = "", user_query: str = "",
                detectors: Sequence[Any] = (), **kwargs) -> AgentSession:
        """Monitor for an iterative agent episode (tool feedback, memory, agents)."""
        return AgentSession(self.config, detectors, system_prompt, user_query,
                            reporter=self.reporter, **kwargs)

    # ------------------------------------------------------------------ #
    # Stage 5: Tool-call taint check                                      #
    # ------------------------------------------------------------------ #

    def check_tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        taint: TaintVector,
        query: str = "",
    ) -> Optional[PolicyViolation]:
        """
        Evaluate a tool call before execution.
        Raises ToolCallBlockedError if blocking_mode=True and threshold exceeded.
        """
        violation = self.tool_monitor.check(tool_name, tool_args, taint, query)
        if violation:
            self._record_violation(violation)
            if violation.blocked:
                raise ToolCallBlockedError(violation)
        return violation

    # ------------------------------------------------------------------ #
    # Stage 6: Reporting                                                   #
    # ------------------------------------------------------------------ #

    def save_report(
        self,
        html_path: Optional[str] = None,
        json_path: Optional[str] = None,
    ) -> None:
        """Write HTML and/or JSON audit reports."""
        html_path = html_path or self.config.audit_report_path
        if html_path:
            self.reporter.write_html(html_path)
        if json_path:
            self.reporter.write_json(json_path)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _record_violation(self, violation: PolicyViolation) -> None:
        self.reporter.record(violation)
        if self._audit_logger:
            self._audit_logger.log(violation)

    @property
    def violations(self) -> List[PolicyViolation]:
        return self.reporter._violations

    def reset_session(self) -> None:
        """Clear accumulated violations for a new session."""
        self.context_monitor.reset()
        self.tool_monitor.reset()
        self.reporter.reset()
