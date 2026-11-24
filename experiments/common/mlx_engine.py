"""Batched greedy generation with MLX on Apple Silicon.

`MLXEngine.generate(requests)` takes a list of `GenRequest` objects (chat
messages or raw token ids). Requests that carry the same `share_group` are
expected to share a long common token prefix (e.g. the base and enhanced
variants of one InjecAgent case); the prefix KV cache is computed once and
copied for every member of the group, which removes most of the duplicated
prefill work. Decoding is greedy (temperature 0).
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


@dataclass
class GenRequest:
    key: str
    messages: Optional[List[Dict[str, str]]] = None
    tokens: Optional[List[int]] = None
    max_tokens: int = 256
    share_group: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenResult:
    key: str
    text: str
    prompt_tokens: int
    gen_tokens: int
    meta: Dict[str, Any]


def _lcp(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    m = min(len(a), len(b))
    while n < m and a[n] == b[n]:
        n += 1
    return n


def limit_mlx_memory(cache_gb: float = 4.0, memory_gb: Optional[float] = None) -> None:
    """Keep the process footprint small: MLX otherwise keeps freed Metal buffers
    in a cache that can grow to tens of GB (observed ~95 GB for a 7B batch job)."""
    import os
    cache_gb = float(os.environ.get("MLX_CACHE_GB", cache_gb))
    mx.set_cache_limit(int(cache_gb * 1024 ** 3))
    mem = os.environ.get("MLX_MEMORY_GB", memory_gb)
    if mem:
        mx.set_memory_limit(int(float(mem) * 1024 ** 3))


class MLXEngine:
    def __init__(self, model_id: str, stop_strings: Sequence[str] = (), max_kv_size: Optional[int] = None):
        limit_mlx_memory()
        t0 = time.time()
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        self.load_seconds = time.time() - t0
        self.sampler = make_sampler(temp=0.0)
        self.max_kv_size = max_kv_size
        self.stop_strings = list(stop_strings)
        stop_tokens: List[List[int]] = [[t] for t in self.tokenizer.eos_token_ids]
        for s in self.stop_strings:
            for variant in (s, "\n" + s, " " + s):
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if ids:
                    stop_tokens.append(ids)
        self.stop_tokens = stop_tokens

    # ------------------------------------------------------------------ #
    def encode_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        return list(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True))

    def _truncate(self, text: str) -> str:
        cut = len(text)
        for s in self.stop_strings:
            i = text.find(s)
            if i != -1:
                cut = min(cut, i)
        return text[:cut]

    def _prefix_cache(self, prefix: List[int]):
        cache = make_prompt_cache(self.model, self.max_kv_size)
        step = 2048
        for i in range(0, len(prefix), step):
            self.model(mx.array(prefix[i:i + step])[None], cache=cache)
            mx.eval([c.state for c in cache])
        return cache

    def generate(self, requests: List[GenRequest], completion_batch_size: int = 16,
                 prefill_batch_size: int = 8) -> List[GenResult]:
        toks: Dict[str, List[int]] = {}
        for r in requests:
            toks[r.key] = list(r.tokens) if r.tokens is not None else self.encode_chat(r.messages or [])

        # group requests that share a prefix
        groups: Dict[str, List[GenRequest]] = {}
        singles: List[GenRequest] = []
        for r in requests:
            if r.share_group:
                groups.setdefault(r.share_group, []).append(r)
            else:
                singles.append(r)

        prompts: List[List[int]] = []
        caches: List[Any] = []
        order: List[GenRequest] = []
        for members in groups.values():
            if len(members) == 1:
                singles.append(members[0])
                continue
            seqs = [toks[m.key] for m in members]
            n = min(_lcp(seqs[0], s) for s in seqs[1:])
            n = min(n, min(len(s) for s in seqs) - 1)
            if n < 64:
                singles.extend(members)
                continue
            base_cache = self._prefix_cache(seqs[0][:n])
            for j, m in enumerate(members):
                prompts.append(toks[m.key][n:])
                caches.append(base_cache if j == 0 else copy.deepcopy(base_cache))
                order.append(m)
        for m in singles:
            prompts.append(toks[m.key])
            caches.append(None)
            order.append(m)

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.stop_tokens,
            sampler=self.sampler,
            completion_batch_size=completion_batch_size,
            prefill_batch_size=prefill_batch_size,
            max_kv_size=self.max_kv_size,
        )
        uids = gen.insert(prompts, [m.max_tokens for m in order], caches=caches)
        out: Dict[int, List[int]] = {u: [] for u in uids}
        try:
            while responses := gen.next_generated():
                for resp in responses:
                    if resp.finish_reason != "stop":
                        out[resp.uid].append(resp.token)
        finally:
            gen.close()
        results = []
        for u, m in zip(uids, order):
            text = self._truncate(self.tokenizer.decode(out[u]))
            results.append(GenResult(m.key, text, len(toks[m.key]), len(out[u]), m.meta))
        mx.clear_cache()
        return results
