"""
Cross-stage policy model for RAGentGuard.

Two-dimensional risk model:
  injection_risk  — content-level (pattern matching).  PRIMARY blocking signal.
  trust_risk      — source-level (untrusted fraction).  WARNING signal by default.

This separation is critical for a low false-positive rate: real RAG systems
routinely retrieve from unverified external sources, so blocking on
trust_risk alone would yield FPR ≈ 100%.  Blocking on injection_risk alone
keeps FPR near zero on typical benign corpora while catching known attack patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import PolicyConfig
from .provenance import (
    AttackCategory,
    PipelineStage,
    PolicyViolation,
    TaintVector,
    TrustLevel,
)


# Structural injection patterns (StruQ-inspired).
# Each entry: (pattern_name, compiled_regex)
INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Role-switching phrases
    ("role_switch", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|system|instructions?)\b",
        re.IGNORECASE,
    )),
    # Instruction injection verbs in data zones
    ("instruction_verb", re.compile(
        r"^\s*(you (must|should|shall|need to)|do not|never|always|stop|respond only|act as)\b",
        re.IGNORECASE | re.MULTILINE,
    )),
    # Base64 payload embedded in text
    ("base64_payload", re.compile(
        r"(?:[A-Za-z0-9+/]{40,}={0,2})"
    )),
    # Prompt delimiters / separator injection
    ("prompt_delimiter", re.compile(
        r"(```|<\|im_start\||<\|im_end\||<\|system\||<\|user\||<\|assistant\||###\s*(System|Human|Assistant))",
        re.IGNORECASE,
    )),
    # Exfiltration attempt patterns
    ("exfiltration_url", re.compile(
        r"(https?://[^\s]*attacker[^\s]*|https?://[^\s]*\?[^\s=]+=\{.*\}|"
        r"(fetch|curl|wget|POST|GET)\b[^\n]*(attacker|\{.*\})|"
        r"[A-Za-z0-9._%+-]+@attacker\.[A-Za-z0-9.-]+)",
        re.IGNORECASE,
    )),
    # Memory/tool hijack phrases
    ("tool_hijack", re.compile(
        r"\b(call|invoke|execute|run|trigger)\b.{0,30}\b(tool|function|api|command|bash|shell|python)\b",
        re.IGNORECASE,
    )),
    # Judge manipulation phrases
    ("judge_manipulation", re.compile(
        r"\b(rate|score|evaluate|grade|judge)\b.{0,30}\b(10|perfect|excellent|100|highest|best)\b",
        re.IGNORECASE,
    )),
]


def _compute_context_untrust_fraction(
    chunks: List[Tuple[str, TaintVector]],
) -> float:
    """Return fraction of total text length from sources with trust < MEDIUM."""
    if not chunks:
        return 0.0
    total_len = sum(len(text) for text, _ in chunks)
    if total_len == 0:
        return 0.0
    untrusted_len = sum(
        len(text)
        for text, tv in chunks
        if any(t.trust_level < TrustLevel.MEDIUM for t in tv.contributing_tags)
    )
    return untrusted_len / total_len


def scan_for_injection_patterns(text: str) -> List[str]:
    """Return names of all injection patterns found in text."""
    return [name for name, pattern in INJECTION_PATTERNS if pattern.search(text)]


def _classify_attack_category(patterns: List[str], taint: TaintVector) -> AttackCategory:
    if "judge_manipulation" in patterns:
        return AttackCategory.JUDGE_MANIPULATION
    if "tool_hijack" in patterns or "exfiltration_url" in patterns:
        return AttackCategory.CROSS_TOOL_TAINT
    if "role_switch" in patterns or "instruction_verb" in patterns or "prompt_delimiter" in patterns:
        return AttackCategory.RETRIEVAL_INJECTION
    if taint.taint_score > 0 and not patterns:
        return AttackCategory.MEMORY_POISONING
    return AttackCategory.UNKNOWN


@dataclass
class ContextRiskAssessment:
    """Result of a context-level risk evaluation."""
    injection_risk: bool        # injection patterns detected (content-level)
    trust_risk: bool            # high untrusted-source fraction (source-level)
    injection_patterns: List[str]
    untrust_fraction: float
    violation: Optional[PolicyViolation]

    @property
    def has_violation(self) -> bool:
        return self.violation is not None


class PolicyEngine:
    """
    Central policy decision point.
    Implements the two-dimensional risk model described in the module docstring.
    """

    def __init__(self, config: PolicyConfig):
        self.config = config

    def evaluate_context(
        self,
        assembled_context: str,
        chunks: List[Tuple[str, TaintVector]],
        query: str = "",
    ) -> Optional[PolicyViolation]:
        """
        Stage 3 check: evaluate assembled context before LLM submission.

        Blocking logic:
          - injection_risk alone  → BLOCK (if blocking_mode=True)
          - trust_risk alone      → FLAG only (unless trust_only_blocking=True)
          - both risks            → BLOCK (if blocking_mode=True)
          - neither               → PASS
        """
        assessment = self.assess_context(assembled_context, chunks, query)
        return assessment.violation

    def assess_context(
        self,
        assembled_context: str,
        chunks: List[Tuple[str, TaintVector]],
        query: str = "",
    ) -> ContextRiskAssessment:
        """Full risk assessment with both risk dimensions exposed."""
        untrust_fraction = _compute_context_untrust_fraction(chunks)
        injection_patterns = scan_for_injection_patterns(assembled_context)

        injection_risk = len(injection_patterns) >= self.config.injection_pattern_min_matches
        trust_risk = untrust_fraction >= self.config.context_untrust_threshold

        # Neither risk dimension triggered → clean pass
        if not injection_risk and not trust_risk:
            return ContextRiskAssessment(
                injection_risk=False, trust_risk=False,
                injection_patterns=[], untrust_fraction=untrust_fraction,
                violation=None,
            )

        # Merge taint vectors
        merged_taint = TaintVector()
        for _, tv in chunks:
            merged_taint = merged_taint.merge(tv)
        merged_taint = merged_taint.advance_stage(PipelineStage.CONTEXT_ASSEMBLY)
        merged_taint.injection_patterns_found = injection_patterns

        # Determine whether to block
        # injection_risk → block; trust_risk alone → warn unless trust_only_blocking
        should_block = self.config.blocking_mode and (
            injection_risk
            or (trust_risk and self.config.trust_only_blocking)
        )

        attack_cat = _classify_attack_category(injection_patterns, merged_taint)
        merged_taint.detected_attack_category = attack_cat

        risk_labels = []
        if injection_risk:
            risk_labels.append(f"injection({injection_patterns})")
        if trust_risk:
            risk_labels.append(f"trust_fraction={untrust_fraction:.2f}")

        violation = PolicyViolation(
            stage=PipelineStage.CONTEXT_ASSEMBLY,
            violation_type=f"context_risk: {', '.join(risk_labels)}",
            taint_vector=merged_taint,
            context_fraction=untrust_fraction,
            blocked=should_block,
            query=query,
            raw_context_snippet=assembled_context[:512],
        )

        return ContextRiskAssessment(
            injection_risk=injection_risk,
            trust_risk=trust_risk,
            injection_patterns=injection_patterns,
            untrust_fraction=untrust_fraction,
            violation=violation,
        )

    def evaluate_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        taint: TaintVector,
        query: str = "",
    ) -> Optional[PolicyViolation]:
        """
        Stage 5 check: evaluate tool call taint before execution.

        Tool calls require BOTH high taint score AND injection patterns to block,
        preventing spurious blocks on legitimate tool calls from low-trust corpora.
        """
        taint = taint.advance_stage(PipelineStage.TOOL_DISPATCH)

        arg_text = str(tool_args)
        arg_patterns = scan_for_injection_patterns(arg_text)
        all_patterns = list(set(taint.injection_patterns_found + arg_patterns))
        taint.injection_patterns_found = all_patterns

        # Block: high taint score AND (injection patterns in args OR in taint history)
        high_taint = taint.taint_score >= self.config.tool_call_block_threshold
        has_injection = len(all_patterns) > 0

        if high_taint and has_injection:
            blocked = self.config.blocking_mode
            taint.detected_attack_category = _classify_attack_category(all_patterns, taint)
            return PolicyViolation(
                stage=PipelineStage.TOOL_DISPATCH,
                violation_type=f"tool_call_blocked (tool={tool_name}, score={taint.taint_score:.2f}, patterns={all_patterns})",
                taint_vector=taint,
                context_fraction=taint.taint_score,
                blocked=blocked,
                query=query,
                tool_call={"tool": tool_name, "args": tool_args},
            )

        # Review: high taint score alone (no injection pattern confirmed yet)
        if taint.taint_score >= self.config.tool_call_review_threshold:
            return PolicyViolation(
                stage=PipelineStage.TOOL_DISPATCH,
                violation_type=f"tool_call_review (tool={tool_name}, score={taint.taint_score:.2f})",
                taint_vector=taint,
                context_fraction=taint.taint_score,
                blocked=False,
                query=query,
                tool_call={"tool": tool_name, "args": tool_args},
            )

        return None


# ====================================================================== #
# S7 tool-dispatch gate driven by S6 attribution                          #
# ====================================================================== #

_READ_VERBS = (
    "get", "search", "read", "view", "list", "fetch", "find", "check", "query", "lookup",
    "retrieve", "navigate", "browse", "download", "access", "show", "describe", "count",
)


def tool_privilege(tool_name: str, overrides: Optional[dict] = None) -> str:
    """'read' for retrieval-only tools, 'write' for side-effecting/egress tools.

    The default heuristic classifies by the verb that starts the operation name
    (after any toolkit prefix); deployments should pass explicit overrides.
    """
    if overrides and tool_name in overrides:
        return overrides[tool_name]
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", tool_name or "")
    words = [w.lower() for w in parts] or [(tool_name or "").lower()]
    # snake_case names
    if "_" in (tool_name or ""):
        words = [w for w in (tool_name or "").lower().split("_") if w]
    for w in words:
        if w in _READ_VERBS:
            return "read"
        if w in {"send", "grant", "unlock", "pay", "transfer", "deposit", "place", "withdraw", "create",
                 "update", "move", "manage", "delete", "schedule", "go", "auto", "disable", "share",
                 "leave", "execute", "give", "post", "reply", "forward", "invite", "add", "remove",
                 "set", "change", "book", "reserve", "cancel", "upload", "write", "append", "generate",
                 "follow", "like", "purchase", "buy", "order", "submit", "edit", "rename", "copy",
                 "modify", "approve", "enable", "lock", "trigger", "run", "call", "dispatch"}:
            return "write"
    return "write"


@dataclass
class ToolGateDecision:
    action: str                      # allow | review | block
    reasons: List[str]
    privilege: str
    attribution: Optional[dict] = None
    verified: Optional[bool] = None  # result of the interventional check, if run

    @property
    def blocked(self) -> bool:
        return self.action == "block"


class ToolGatePolicy:
    """S7 decision rule. Inputs: tool name/arguments, the S6 ToolCallAttribution,
    and the visible segments (for the session-level and args-only modes).

    attribution mode (default):
      read-only tool                                   -> allow
      tainted value/action with evidence on a source    -> block
      evidence on a supporting segment labelled trusted -> block (defence in depth)
      tainted without evidence                          -> review_action
                                                           (allow | block | verify)
      otherwise                                         -> allow
    """

    def __init__(self, config: PolicyConfig, privilege_overrides: Optional[dict] = None,
                 args_detector=None):
        self.config = config
        self.overrides = privilege_overrides or {}
        self.args_detector = args_detector

    def decide(self, tool_name: str, tool_args, attribution=None, segments=None,
               verifier=None) -> ToolGateDecision:
        cfg = self.config
        priv = tool_privilege(tool_name, self.overrides)
        att = attribution.to_dict() if attribution is not None else None
        mode = cfg.tool_gate_mode
        if mode == "none":
            return ToolGateDecision("allow", ["gate disabled"], priv, att)
        if mode == "args_only":
            text = str(tool_args)
            fired = (self.args_detector.fires(text) if self.args_detector is not None
                     else bool(scan_for_injection_patterns(text)))
            if priv == "write" and fired:
                return ToolGateDecision("block", ["injection evidence in arguments"], priv, att)
            return ToolGateDecision("allow", [], priv, att)
        if mode == "max_taint":
            segs = segments or []
            session_taint = attribution.session_taint if attribution is not None else 0.0
            evidence = any(getattr(s, "has_evidence", False) for s in segs)
            text = str(tool_args)
            evidence = evidence or bool(scan_for_injection_patterns(text))
            if session_taint >= cfg.tool_call_block_threshold and evidence:
                return ToolGateDecision("block", [f"session taint {session_taint:.2f} with evidence"], priv, att)
            if session_taint >= cfg.tool_call_review_threshold:
                return ToolGateDecision("review", [f"session taint {session_taint:.2f}"], priv, att)
            return ToolGateDecision("allow", [], priv, att)
        # ---------------- attribution mode ---------------- #
        if priv == "read":
            return ToolGateDecision("allow", ["read-only tool"], priv, att)
        if attribution is None:
            return ToolGateDecision("review", ["no attribution available"], priv, att)
        reasons = []
        if attribution.tainted and attribution.evidence_sources:
            reasons.append("tainted flow from segment with injection evidence")
            return ToolGateDecision("block", reasons, priv, att)
        if attribution.evidence_sources and cfg.block_evidence_regardless_of_trust:
            reasons.append("argument/action supported by evidence-bearing segment")
            return ToolGateDecision("block", reasons, priv, att)
        if attribution.tainted:
            reasons.append("tainted value or action without content evidence")
            if cfg.review_action == "block":
                return ToolGateDecision("block", reasons, priv, att)
            if cfg.review_action == "verify" and verifier is not None:
                confirmed = bool(verifier(tool_name, tool_args, attribution))
                reasons.append("interventional check " + ("confirmed" if confirmed else "cleared"))
                return ToolGateDecision("block" if confirmed else "allow", reasons, priv, att, confirmed)
            return ToolGateDecision("review", reasons, priv, att)
        return ToolGateDecision("allow", reasons, priv, att)
