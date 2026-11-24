"""
Provenance-tagged segments for cyclic agent workflows.

A *segment* is any unit of content that becomes visible to the language model:
the system prompt, the user request, a retrieved chunk, a tool output that is
fed back into the loop, a memory item written in an earlier session, or a
message from another agent. Every segment carries an integrity label drawn
from the trust lattice (TrustLevel) and the ids of the segments it was derived
from.

Derived segments (memory notes, agent messages, summaries) receive the
lattice join of their parents, i.e. the *least* trusted parent label, and
inherit any content-evidence flag raised on a parent. Because the lattice is
finite and the join is monotone, repeated loop iterations can only keep or
lower a label: content cannot be "laundered" to a higher trust tier by passing
it through memory, a tool, or another agent.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence

from .provenance import ProvenanceTag, TrustLevel


class SegmentKind(str, Enum):
    SYSTEM = "system"
    USER = "user"
    RETRIEVED = "retrieved"
    TOOL_OUTPUT = "tool_output"
    MEMORY = "memory"
    AGENT = "agent"          # message produced by this or another agent


#: Default numeric trust values (the lattice order is what matters; the
#: numbers are used only through thresholds, see `TrustMapping`).
DEFAULT_TRUST_VALUES: Dict[TrustLevel, float] = {
    TrustLevel.UNTRUSTED: 0.0,
    TrustLevel.LOW: 0.25,
    TrustLevel.MEDIUM: 0.5,
    TrustLevel.HIGH: 0.75,
    TrustLevel.SYSTEM: 1.0,
}


@dataclass
class TrustMapping:
    """Maps lattice tiers to numbers and decides which tiers count as untrusted.

    `untrusted_below` is the threshold t* such that a segment is untrusted when
    value(tier) < t*. With the default values and t* = 0.5, UNTRUSTED and LOW
    are untrusted. Sensitivity analyses vary both the values and t*.
    `zero_trust=True` treats every segment that is not SYSTEM or USER as
    untrusted, which is the robust setting when a "trusted" source may be
    compromised.
    """
    values: Dict[TrustLevel, float] = field(default_factory=lambda: dict(DEFAULT_TRUST_VALUES))
    untrusted_below: float = 0.5
    zero_trust: bool = False

    def value(self, tier: TrustLevel) -> float:
        return float(self.values.get(tier, float(tier)))

    def is_untrusted(self, seg: "Segment") -> bool:
        if seg.kind in (SegmentKind.SYSTEM, SegmentKind.USER):
            return False
        if self.zero_trust:
            return True
        return self.value(seg.trust) < self.untrusted_below

    def untrust(self, seg: "Segment") -> float:
        if seg.kind in (SegmentKind.SYSTEM, SegmentKind.USER):
            return 0.0
        if self.zero_trust:
            return 1.0
        return 1.0 - self.value(seg.trust)


@dataclass
class Segment:
    text: str
    kind: SegmentKind
    trust: TrustLevel = TrustLevel.UNTRUSTED
    source: str = ""                         # URI, tool name, memory key, agent id
    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parents: List[str] = field(default_factory=list)
    step: int = 0                            # loop iteration that produced it
    evidence: Dict[str, float] = field(default_factory=dict)   # detector -> score
    flagged: bool = False                    # some detector fired on this text
    inherited_flag: bool = False             # a parent was flagged
    tag: Optional[ProvenanceTag] = None

    @property
    def has_evidence(self) -> bool:
        return self.flagged or self.inherited_flag

    def to_dict(self) -> Dict:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind.value,
            "trust": float(self.trust),
            "source": self.source,
            "parents": list(self.parents),
            "step": self.step,
            "evidence": dict(self.evidence),
            "flagged": self.flagged,
            "inherited_flag": self.inherited_flag,
            "chars": len(self.text),
        }


def join_trust(levels: Iterable[TrustLevel]) -> TrustLevel:
    """Lattice join for integrity: the least trusted level wins."""
    levels = list(levels)
    if not levels:
        return TrustLevel.UNTRUSTED
    return min(levels, key=float)


def derive_segment(text: str, kind: SegmentKind, parents: Sequence[Segment], source: str = "",
                   step: int = 0, floor: Optional[TrustLevel] = None) -> Segment:
    """Create a derived segment whose label is the join of its parents.

    `floor` optionally caps the label (e.g. memory written by the agent can never
    be more trusted than MEDIUM even when all parents are trusted).
    """
    levels = [p.trust for p in parents]
    if floor is not None:
        levels.append(floor)
    return Segment(
        text=text,
        kind=kind,
        trust=join_trust(levels) if levels else TrustLevel.UNTRUSTED,
        source=source,
        parents=[p.segment_id for p in parents],
        step=step,
        inherited_flag=any(p.has_evidence for p in parents),
    )
