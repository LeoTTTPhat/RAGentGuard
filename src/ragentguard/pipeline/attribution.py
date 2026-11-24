"""
Stage 4: Generation-Time Attribution.

Maps each generated token/span back to the context segments it was most
influenced by.  Three methods are available:

  - "attention_rollout"  : uses attention weights from the last decoder layer
                           (requires transformers model with output_attentions=True)
  - "gradient"           : uses gradient × input (requires autograd access)
  - "overlap"            : lightweight string-overlap baseline (no model access)

For API-only (closed-weight) language models, only "overlap" is available.
The overlap method is used as the research baseline when comparing with TracLLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.config import AttributionConfig
from ..core.provenance import PipelineStage, ProvenanceTag, TaintVector, TrustLevel


@dataclass
class SpanAttribution:
    """Attribution score for one context span."""
    source_id: str
    chunk_id: str
    source_path: str
    trust_level: TrustLevel
    attribution_score: float          # 0.0 – 1.0
    matched_ngrams: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "trust_level": float(self.trust_level),
            "attribution_score": self.attribution_score,
            "matched_ngrams": self.matched_ngrams[:5],  # limit for report size
        }


@dataclass
class AttributionResult:
    """Full attribution analysis for one generation output."""
    generated_text: str
    attributions: List[SpanAttribution] = field(default_factory=list)
    method_used: str = "overlap"
    updated_taint: Optional[TaintVector] = None  # taint refined by attribution

    @property
    def top_source(self) -> Optional[SpanAttribution]:
        if not self.attributions:
            return None
        return max(self.attributions, key=lambda a: a.attribution_score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_text_preview": self.generated_text[:256],
            "method": self.method_used,
            "attributions": [a.to_dict() for a in self.attributions],
            "top_source_id": self.top_source.source_id if self.top_source else None,
        }


def _ngrams(text: str, n: int) -> List[str]:
    tokens = re.findall(r"\w+", text.lower())
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _overlap_score(generated: str, context_chunk: str, n: int = 3) -> Tuple[float, List[str]]:
    """
    Character n-gram overlap between generated text and a context chunk.
    Returns (score, matched_ngrams).
    """
    gen_ngrams = set(_ngrams(generated, n))
    ctx_ngrams = set(_ngrams(context_chunk, n))
    if not gen_ngrams:
        return 0.0, []
    matched = gen_ngrams & ctx_ngrams
    score = len(matched) / len(gen_ngrams)
    return score, list(matched)


class GenerationAttributor:
    """
    Maps generated output to source documents via one of three methods.
    Updates the TaintVector with attribution-weighted taint scores.
    """

    def __init__(self, config: AttributionConfig):
        self.config = config

    def attribute(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any = None,          # transformers model if method != "overlap"
        tokenizer: Any = None,
    ) -> AttributionResult:
        """
        Attribute generated_text to source chunks.

        Args:
            generated_text: LLM output to attribute
            chunk_taints: list of (chunk_text, TaintVector) from retrieval stage
            model: optional transformers model (for attention_rollout / gradient)
            tokenizer: optional tokenizer

        Returns AttributionResult with per-source attribution scores.
        """
        if self.config.method == "attention_rollout" and model is not None:
            return self._attention_rollout(generated_text, chunk_taints, model, tokenizer)
        if self.config.method == "gradient" and model is not None:
            return self._gradient_attribution(generated_text, chunk_taints, model, tokenizer)
        return self._overlap_attribution(generated_text, chunk_taints)

    # ------------------------------------------------------------------ #
    # Overlap baseline (always available, no model required)              #
    # ------------------------------------------------------------------ #

    def _overlap_attribution(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
    ) -> AttributionResult:
        attributions: List[SpanAttribution] = []

        for chunk_text, tv in chunk_taints:
            score, matched = _overlap_score(generated_text, chunk_text, n=3)
            if score < self.config.min_attribution_score:
                continue
            for tag in tv.contributing_tags:
                attributions.append(SpanAttribution(
                    source_id=tag.source_id,
                    chunk_id=tag.chunk_id,
                    source_path=tag.source_path,
                    trust_level=tag.trust_level,
                    attribution_score=score,
                    matched_ngrams=matched,
                ))

        # Keep top-k by score
        attributions.sort(key=lambda a: a.attribution_score, reverse=True)
        attributions = attributions[: self.config.top_k_sources]

        updated_taint = self._build_attributed_taint(attributions)
        return AttributionResult(
            generated_text=generated_text,
            attributions=attributions,
            method_used="overlap",
            updated_taint=updated_taint,
        )

    # ------------------------------------------------------------------ #
    # Attention rollout (requires transformers model)                      #
    # ------------------------------------------------------------------ #

    def _attention_rollout(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any,
        tokenizer: Any,
    ) -> AttributionResult:
        """
        Attention rollout over the decoder's self-attention layers.
        Falls back to overlap if torch is not available.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            return self._overlap_attribution(generated_text, chunk_taints)

        try:
            import torch

            context_text = "\n\n".join(t for t, _ in chunk_taints)
            prompt = f"{context_text}\n\nGenerated: {generated_text}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Average attention across heads and layers
            attn_matrices = outputs.attentions  # tuple of (batch, heads, seq, seq)
            # Simple rollout: product of mean-head attention across layers
            rollout = torch.eye(attn_matrices[0].shape[-1])
            for layer_attn in attn_matrices:
                mean_attn = layer_attn[0].mean(dim=0)  # (seq, seq)
                rollout = rollout @ mean_attn

            # Map attention mass to each context chunk via token positions
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            attributions = self._map_attn_to_chunks(
                rollout, tokens, chunk_taints, generated_text, tokenizer
            )
            attributions.sort(key=lambda a: a.attribution_score, reverse=True)
            attributions = attributions[: self.config.top_k_sources]

            return AttributionResult(
                generated_text=generated_text,
                attributions=attributions,
                method_used="attention_rollout",
                updated_taint=self._build_attributed_taint(attributions),
            )
        except Exception:
            # Any error → fall back to overlap
            return self._overlap_attribution(generated_text, chunk_taints)

    def _map_attn_to_chunks(
        self,
        rollout: Any,
        tokens: List[str],
        chunk_taints: List[Tuple[str, TaintVector]],
        generated_text: str,
        tokenizer: Any,
    ) -> List[SpanAttribution]:
        """Map rollout attention to source chunks by approximate token alignment."""
        import torch

        # Find generated text tokens (last part of sequence)
        gen_tokens = tokenizer(generated_text, add_special_tokens=False)["input_ids"]
        total_len = rollout.shape[-1]
        gen_start = max(0, total_len - len(gen_tokens))

        attributions: List[SpanAttribution] = []
        context_start = 0

        for chunk_text, tv in chunk_taints:
            chunk_tokens = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
            chunk_end = min(context_start + len(chunk_tokens), gen_start)

            if chunk_end <= context_start:
                context_start += len(chunk_tokens)
                continue

            # Attention mass from generated tokens → this chunk's tokens
            gen_attn = rollout[gen_start:, context_start:chunk_end]
            score = float(gen_attn.mean().item()) if gen_attn.numel() > 0 else 0.0

            for tag in tv.contributing_tags:
                attributions.append(SpanAttribution(
                    source_id=tag.source_id,
                    chunk_id=tag.chunk_id,
                    source_path=tag.source_path,
                    trust_level=tag.trust_level,
                    attribution_score=score,
                ))

            context_start += len(chunk_tokens)

        return attributions

    # ------------------------------------------------------------------ #
    # Gradient attribution (requires autograd)                             #
    # ------------------------------------------------------------------ #

    def _gradient_attribution(
        self,
        generated_text: str,
        chunk_taints: List[Tuple[str, TaintVector]],
        model: Any,
        tokenizer: Any,
    ) -> AttributionResult:
        """
        Gradient × input attribution.
        Falls back to overlap if torch not available.
        """
        try:
            import torch

            context_text = "\n\n".join(t for t, _ in chunk_taints)
            full_text = f"{context_text}\n\nAnswer: {generated_text}"

            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            input_embeds = model.get_input_embeddings()(inputs["input_ids"])
            input_embeds = input_embeds.requires_grad_(True)

            outputs = model(inputs_embeds=input_embeds, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()

            grads = input_embeds.grad  # (1, seq, hidden)
            scores = (grads * input_embeds).sum(dim=-1).abs()[0]  # (seq,)

            attributions = self._map_gradient_to_chunks(
                scores, chunk_taints, context_text, generated_text, tokenizer
            )
            attributions.sort(key=lambda a: a.attribution_score, reverse=True)
            attributions = attributions[: self.config.top_k_sources]

            return AttributionResult(
                generated_text=generated_text,
                attributions=attributions,
                method_used="gradient",
                updated_taint=self._build_attributed_taint(attributions),
            )
        except Exception:
            return self._overlap_attribution(generated_text, chunk_taints)

    def _map_gradient_to_chunks(
        self,
        token_scores: Any,
        chunk_taints: List[Tuple[str, TaintVector]],
        context_text: str,
        generated_text: str,
        tokenizer: Any,
    ) -> List[SpanAttribution]:
        """Average gradient score over each chunk's token span."""
        attributions: List[SpanAttribution] = []
        offset = 0

        for chunk_text, tv in chunk_taints:
            chunk_token_ids = tokenizer(chunk_text, add_special_tokens=False)["input_ids"]
            end = min(offset + len(chunk_token_ids), len(token_scores))
            chunk_scores = token_scores[offset:end]
            score = float(chunk_scores.mean().item()) if len(chunk_scores) > 0 else 0.0
            # Normalise to [0, 1]
            max_score = float(token_scores.max().item()) or 1.0
            score = score / max_score

            for tag in tv.contributing_tags:
                attributions.append(SpanAttribution(
                    source_id=tag.source_id,
                    chunk_id=tag.chunk_id,
                    source_path=tag.source_path,
                    trust_level=tag.trust_level,
                    attribution_score=score,
                ))
            offset += len(chunk_token_ids)

        return attributions

    # ------------------------------------------------------------------ #
    # Helper                                                               #
    # ------------------------------------------------------------------ #

    def _build_attributed_taint(
        self, attributions: List[SpanAttribution]
    ) -> TaintVector:
        """Build a TaintVector weighted by attribution scores."""
        if not attributions:
            return TaintVector()

        tags: List[ProvenanceTag] = []
        total_score = sum(a.attribution_score for a in attributions) or 1.0
        weighted_untrust = 0.0

        for attr in attributions:
            tag = ProvenanceTag(
                source_id=attr.source_id,
                chunk_id=attr.chunk_id,
                trust_level=attr.trust_level,
                source_path=attr.source_path,
            )
            tags.append(tag)
            weighted_untrust += attr.attribution_score * (1.0 - float(attr.trust_level))

        taint_score = weighted_untrust / total_score

        return TaintVector(
            contributing_tags=tags,
            taint_score=min(1.0, taint_score),
            propagation_path=[
                PipelineStage.INGESTION,
                PipelineStage.RETRIEVAL,
                PipelineStage.CONTEXT_ASSEMBLY,
                PipelineStage.GENERATION,
            ],
        )


# ====================================================================== #
# Argument- and answer-level provenance (S6) -- consumed by S7            #
# ====================================================================== #
#
# The functions below give S6 an explicit contract:
#   input : a generated tool call (name + arguments) or a generated answer, and
#           the provenance-tagged segments visible to the model when it was
#           generated (system prompt, user request, retrieved chunks, tool
#           outputs, memory items);
#   output: for every informative argument value / actionable answer span, the
#           set of segments it can be traced to and whether any *trusted*
#           segment supports it; an action-level attribution score per segment;
#           and an attribution-weighted taint score.
# S7 (ToolGatePolicy) uses this output to decide allow / review / block, so the
# attribution result changes what the pipeline does next (it is not a report-
# only side channel).

import json as _json
from typing import Iterable as _Iterable

from ..core.segments import Segment as _Segment, SegmentKind as _SegKind, TrustMapping as _TrustMapping

_WORD = re.compile(r"[a-z0-9][a-z0-9._%+@/:-]*[a-z0-9]|[a-z0-9]", re.IGNORECASE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|_|-")
_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "for", "in", "on", "at", "by", "with", "from",
    "is", "are", "be", "my", "your", "me", "you", "it", "this", "that", "please", "true", "false",
    "none", "null", "get", "set", "new", "all", "id", "ids", "name", "type", "manager", "info",
}
_URL = re.compile(r"(?:https?://|www\.)[^\s'\"<>)]+", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_LONGNUM = re.compile(r"\b(?:[A-Z]{2}\d{2}[A-Z0-9]{8,30}|\d[\d\- ]{5,}\d)\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


def _flatten(value: Any, path: str = "") -> List[Tuple[str, str]]:
    """Leaf values of a (possibly nested) argument structure as strings."""
    out: List[Tuple[str, str]] = []
    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_flatten(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            out.extend(_flatten(v, f"{path}[{i}]"))
    elif value is None or isinstance(value, bool):
        pass
    else:
        out.append((path, str(value)))
    return out


def parse_arguments(raw: str) -> Any:
    """Tool arguments as emitted by a model: strict JSON, lenient JSON (raw newlines),
    a Python literal (single quotes), or -- as a last resort -- "key": "value" pairs."""
    import ast as _ast
    for loader in (_json.loads, lambda x: _json.loads(x, strict=False), _ast.literal_eval):
        try:
            out = loader(raw)
            if isinstance(out, (dict, list)):
                return out
        except Exception:
            continue
    pairs = re.findall(r"[\"']([A-Za-z_]\w*)[\"']\s*:\s*[\"']((?:[^\"'\\]|\\.)*)[\"']", raw or "")
    return {k: v for k, v in pairs} if pairs else {"_raw": raw}


def informative(value: str) -> bool:
    """Values that can carry provenance: identifiers, addresses, amounts, text."""
    v = value.strip()
    if len(v) >= 4 and re.search(r"[A-Za-z@/]", v):
        return True
    digits = re.sub(r"\D", "", v)
    return len(digits) >= 3


@dataclass
class ValueProvenance:
    path: str
    value: str
    sources: List[str]              # segment ids that contain the value
    trusted_support: bool           # some trusted segment contains it
    match: str                      # exact | fuzzy | none
    tainted: bool                   # supported only by untrusted segments

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "value": self.value[:120], "sources": self.sources,
                "trusted_support": self.trusted_support, "match": self.match, "tainted": self.tainted}


@dataclass
class ToolCallAttribution:
    tool: str
    values: List[ValueProvenance]
    action_scores: Dict[str, float]          # segment id -> lexical support of the call
    action_source: Optional[str]             # best-supporting segment id
    action_tainted: bool                     # call better explained by untrusted content
    attributed_taint: float                  # worst untrust among attributed sources
    session_taint: float                     # max untrust over all visible segments
    evidence_sources: List[str]              # attributed segments carrying content evidence

    @property
    def tainted_values(self) -> List[ValueProvenance]:
        return [v for v in self.values if v.tainted]

    @property
    def tainted(self) -> bool:
        return bool(self.tainted_values) or self.action_tainted

    @property
    def source_ids(self) -> List[str]:
        ids: List[str] = []
        for v in self.tainted_values:
            ids.extend(v.sources)
        if self.action_tainted and self.action_source:
            ids.append(self.action_source)
        return list(dict.fromkeys(ids))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "values": [v.to_dict() for v in self.values],
            "action_source": self.action_source,
            "action_tainted": self.action_tainted,
            "attributed_taint": round(self.attributed_taint, 4),
            "session_taint": round(self.session_taint, 4),
            "evidence_sources": self.evidence_sources,
            "tainted": self.tainted,
        }


def _value_sources(value: str, segments: Sequence[_Segment], fuzzy_min: float, ngram: int = 4
                   ) -> Tuple[List[str], str]:
    v = _norm(value)
    if not v:
        return [], "none"
    if len(v) <= 64:
        hits = [s.segment_id for s in segments if v in _norm(s.text)]
        return hits, ("exact" if hits else "none")
    toks = _words(v)
    grams = {" ".join(toks[i:i + ngram]) for i in range(max(1, len(toks) - ngram + 1))}
    hits = []
    for s in segments:
        st = _words(s.text)
        sg = {" ".join(st[i:i + ngram]) for i in range(max(1, len(st) - ngram + 1))}
        if grams and len(grams & sg) / len(grams) >= fuzzy_min:
            hits.append(s.segment_id)
    return hits, ("fuzzy" if hits else "none")


def _call_tokens(tool: str, values: _Iterable[str]) -> List[str]:
    parts = [p.lower() for p in _CAMEL.split(tool or "") if p]
    toks = [t for t in parts if t not in _STOP and len(t) > 2]
    for v in values:
        toks.extend(t for t in _words(v) if t not in _STOP and len(t) > 2)
    return list(dict.fromkeys(toks))


def attribute_tool_call(tool: str, args: Any, segments: Sequence[_Segment],
                        trust: Optional[_TrustMapping] = None, fuzzy_min: float = 0.5,
                        action_min: float = 0.34, action_margin: float = 0.0) -> ToolCallAttribution:
    """S6 attribution of a tool call to the visible segments.

    * Argument provenance: every informative leaf value is matched against the
      segments (normalised exact substring for values up to 64 characters,
      word-4-gram containment >= `fuzzy_min` for longer free text). A value is
      *tainted* when it is found only in untrusted segments.
    * Action provenance: the share of the call's content tokens (tool-name parts
      and argument tokens) present in each segment. The call is action-tainted
      when its best-supporting segment is untrusted, that support is at least
      `action_min`, and it exceeds the best trusted support by `action_margin`.
    """
    trust = trust or _TrustMapping()
    if isinstance(args, str):
        args = parse_arguments(args)
    by_id = {s.segment_id: s for s in segments}
    values: List[ValueProvenance] = []
    for path, val in _flatten(args):
        if not informative(val):
            continue
        hits, match = _value_sources(val, segments, fuzzy_min)
        trusted = any(not trust.is_untrusted(by_id[h]) for h in hits)
        values.append(ValueProvenance(path, val, hits, trusted, match,
                                      tainted=bool(hits) and not trusted))
    toks = _call_tokens(tool, [v.value for v in values])
    action_scores: Dict[str, float] = {}
    if toks:
        for s in segments:
            sw = set(_words(s.text))
            action_scores[s.segment_id] = sum(1 for t in toks if t in sw) / len(toks)
    best_untrusted = max((sc for sid, sc in action_scores.items() if trust.is_untrusted(by_id[sid])), default=0.0)
    best_trusted = max((sc for sid, sc in action_scores.items() if not trust.is_untrusted(by_id[sid])), default=0.0)
    action_source = max(action_scores, key=action_scores.get) if action_scores else None
    action_tainted = (best_untrusted >= action_min and best_untrusted > best_trusted + action_margin)
    if action_tainted:
        action_source = max((sid for sid in action_scores if trust.is_untrusted(by_id[sid])),
                            key=action_scores.get)
    attributed = [by_id[h] for v in values if v.tainted for h in v.sources]
    if action_tainted and action_source:
        attributed.append(by_id[action_source])
    attributed_taint = max((trust.untrust(s) for s in attributed), default=0.0)
    session_taint = max((trust.untrust(s) for s in segments), default=0.0)
    uniq = list({s.segment_id: s for s in attributed}.values())
    evidence = [s.segment_id for s in uniq if s.has_evidence]
    # defence in depth: evidence-bearing segments that support any value or the
    # action are reported even when they are labelled trusted (compromised source)
    supporting = {h for v in values for h in v.sources}
    if action_source and action_scores.get(action_source, 0.0) >= action_min:
        supporting.add(action_source)
    for sid in supporting:
        s = by_id[sid]
        if s.has_evidence and s.kind not in (_SegKind.SYSTEM, _SegKind.USER) and sid not in evidence:
            evidence.append(sid)
    return ToolCallAttribution(tool, values, action_scores, action_source, action_tainted,
                               attributed_taint, session_taint, evidence)


@dataclass
class AnswerAttribution:
    spans: List[ValueProvenance]             # actionable spans (URLs, e-mails, long numbers)
    chunk_scores: Dict[str, float]           # segment id -> word-3-gram overlap share
    top_segment: Optional[str]
    attributed_taint: float

    @property
    def tainted_spans(self) -> List[ValueProvenance]:
        return [s for s in self.spans if s.tainted]

    def to_dict(self) -> Dict[str, Any]:
        return {"spans": [s.to_dict() for s in self.spans], "chunk_scores": self.chunk_scores,
                "top_segment": self.top_segment, "attributed_taint": round(self.attributed_taint, 4)}


def attribute_answer(answer: str, segments: Sequence[_Segment], trust: Optional[_TrustMapping] = None,
                     n: int = 3) -> AnswerAttribution:
    """S6 attribution of a generated answer.

    Actionable spans (URLs, e-mail addresses, account-like numbers) are traced
    to the segments that contain them; the whole answer is also attributed to
    segments by word-n-gram overlap (the low-cost method usable with
    closed-weight APIs).
    """
    trust = trust or _TrustMapping()
    by_id = {s.segment_id: s for s in segments}
    spans: List[ValueProvenance] = []
    seen = set()
    for rx in (_URL, _EMAIL, _LONGNUM):
        for m in rx.findall(answer or ""):
            val = m.strip(".,;:)")
            if val.lower() in seen:
                continue
            seen.add(val.lower())
            hits, match = _value_sources(val, segments, 0.5)
            trusted = any(not trust.is_untrusted(by_id[h]) for h in hits)
            spans.append(ValueProvenance("answer", val, hits, trusted, match, bool(hits) and not trusted))
    scores: Dict[str, float] = {}
    for s in segments:
        if s.kind in (_SegKind.SYSTEM,):
            continue
        sc, _ = _overlap_score(answer or "", s.text, n=n)
        scores[s.segment_id] = round(sc, 4)
    top = max(scores, key=scores.get) if scores else None
    total = sum(scores.values()) or 1.0
    taint = sum(sc * trust.untrust(by_id[sid]) for sid, sc in scores.items()) / total
    return AnswerAttribution(spans, scores, top, taint)
