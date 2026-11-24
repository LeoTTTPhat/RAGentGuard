"""
Runtime monitor for one (possibly cyclic) agent episode.

The seven stages of the paper are realised per *event* rather than once per
query, so the same machinery covers linear RAG and iterative agents:

  S1/S1'  every new piece of content (retrieved chunk, tool output fed back
          into the loop, memory item, message from another agent) becomes a
          provenance-tagged Segment with a trust label;
  S5      content-evidence detectors score each non-principal segment
          (warn/redact; evidence is stored on the segment);
  S6      generated tool calls / answers are attributed to segments
          (argument, action and answer provenance);
  S7      the ToolGatePolicy uses the S6 output to allow / review / block.

Derived content (memory notes, summaries, agent messages) is labelled with the
lattice join of the segments it was derived from, so a poisoned memory item
keeps its untrusted label in later sessions (persisted via `Segment.to_dict`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..core.config import RAGentGuardConfig
from ..core.policy import ToolGateDecision, ToolGatePolicy
from ..core.provenance import TrustLevel
from ..core.segments import Segment, SegmentKind, TrustMapping, derive_segment
from ..pipeline.attribution import AnswerAttribution, attribute_answer, attribute_tool_call


@dataclass
class AnswerDecision:
    action: str                     # release | warn | withhold
    reasons: List[str]
    attribution: AnswerAttribution


class AgentSession:
    def __init__(self, config: Optional[RAGentGuardConfig] = None, detectors: Sequence[Any] = (),
                 system_prompt: str = "", user_query: str = "",
                 trust_map: Optional[TrustMapping] = None,
                 privilege_overrides: Optional[Dict[str, str]] = None,
                 verifier: Optional[Callable[..., bool]] = None, reporter: Any = None):
        self.config = config or RAGentGuardConfig.default()
        pc = self.config.policy
        self.trust = trust_map or TrustMapping(untrusted_below=pc.untrusted_below, zero_trust=pc.zero_trust)
        self.detectors = list(detectors)
        self.policy = ToolGatePolicy(pc, privilege_overrides)
        self.verifier = verifier
        self.reporter = reporter
        self.segments: List[Segment] = []
        self.events: List[Dict[str, Any]] = []
        self.step = 0
        if system_prompt:
            self.add_segment(system_prompt, SegmentKind.SYSTEM, TrustLevel.SYSTEM, "system")
        if user_query:
            self.add_segment(user_query, SegmentKind.USER, TrustLevel.HIGH, "user")

    # ----------------------------------------------------------- S1 / S5 #
    def _scan(self, seg: Segment) -> None:
        if seg.kind in (SegmentKind.SYSTEM, SegmentKind.USER) or not self.detectors:
            return
        for det in self.detectors:
            sc = float(det.score(seg.text))
            seg.evidence[det.name] = round(sc, 4)
            if sc >= det.threshold:
                seg.flagged = True

    def add_segment(self, text: str, kind: SegmentKind, trust: TrustLevel, source: str = "",
                    parents: Sequence[Segment] = (), evidence: Optional[Dict[str, float]] = None,
                    flagged: Optional[bool] = None) -> Segment:
        seg = Segment(text=text, kind=kind, trust=trust, source=source,
                      parents=[p.segment_id for p in parents], step=self.step,
                      inherited_flag=any(p.has_evidence for p in parents))
        if evidence is not None:
            seg.evidence = dict(evidence)
            seg.flagged = bool(flagged)
        else:
            self._scan(seg)
        self.segments.append(seg)
        self._log({"stage": "S1", "event": "segment", **seg.to_dict()})
        return seg

    def add_retrieved(self, chunks: Sequence[Tuple[str, TrustLevel, str]]) -> List[Segment]:
        return [self.add_segment(t, SegmentKind.RETRIEVED, tr, src) for t, tr, src in chunks]

    def observe_tool_output(self, tool_name: str, text: str,
                            trust: TrustLevel = TrustLevel.UNTRUSTED,
                            evidence: Optional[Dict[str, float]] = None,
                            flagged: Optional[bool] = None) -> Segment:
        """S1' -- a tool result re-enters the loop as a new tagged segment."""
        self.step += 1
        return self.add_segment(text, SegmentKind.TOOL_OUTPUT, trust, tool_name,
                                evidence=evidence, flagged=flagged)

    def add_agent_message(self, text: str, sender: str, parents: Sequence[Segment]) -> Segment:
        """A message from another agent inherits the join of what that agent read."""
        seg = derive_segment(text, SegmentKind.AGENT, parents, source=sender, step=self.step)
        self._scan(seg)
        self.segments.append(seg)
        self._log({"stage": "S1", "event": "agent_message", **seg.to_dict()})
        return seg

    def write_memory(self, text: str, derived_from: Sequence[Segment],
                     floor: TrustLevel = TrustLevel.MEDIUM) -> Segment:
        """Memory write-back: the stored item is labelled with the join of its sources."""
        seg = derive_segment(text, SegmentKind.MEMORY, derived_from, source="memory",
                             step=self.step, floor=floor)
        self._scan(seg)
        self._log({"stage": "S3", "event": "memory_write", **seg.to_dict()})
        return seg

    def load_memory(self, seg: Segment) -> Segment:
        """Re-surface a persisted memory item (label and evidence preserved)."""
        self.segments.append(seg)
        self._log({"stage": "S4", "event": "memory_read", **seg.to_dict()})
        return seg

    def redact_flagged(self) -> List[Segment]:
        """S5 'redact_chunk' action: segments whose own text carries evidence."""
        return [s for s in self.segments if s.flagged and s.kind not in (SegmentKind.SYSTEM, SegmentKind.USER)]

    # ---------------------------------------------------------- S6 / S7 #
    def attribute(self, tool_name: str, args: Any):
        pc = self.config.policy
        return attribute_tool_call(tool_name, args, self.segments, self.trust,
                                   fuzzy_min=pc.attribution_fuzzy_min,
                                   action_min=pc.action_attribution_min)

    def gate_tool_call(self, tool_name: str, args: Any) -> ToolGateDecision:
        att = self.attribute(tool_name, args)
        self._log({"stage": "S6", "event": "tool_attribution", **att.to_dict()})
        decision = self.policy.decide(tool_name, args, att, self.segments, self.verifier)
        self._log({"stage": "S7", "event": "tool_decision", "tool": tool_name,
                   "action": decision.action, "reasons": decision.reasons,
                   "privilege": decision.privilege, "verified": decision.verified})
        return decision

    def release_answer(self, answer: str) -> AnswerDecision:
        att = attribute_answer(answer, self.segments, self.trust)
        by_id = {s.segment_id: s for s in self.segments}
        reasons: List[str] = []
        action = "release"
        evid_spans = [sp for sp in att.tainted_spans if any(by_id[h].has_evidence for h in sp.sources)]
        if evid_spans:
            action, reasons = "withhold", ["actionable span from evidence-bearing untrusted segment"]
        elif att.tainted_spans:
            action, reasons = "warn", ["actionable span supported only by untrusted segments"]
        self._log({"stage": "S6", "event": "answer_attribution", "action": action, **att.to_dict()})
        return AnswerDecision(action, reasons, att)

    def _log(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        if self.reporter is not None and event.get("stage") in ("S6", "S7"):
            self.reporter.record_event(event)
