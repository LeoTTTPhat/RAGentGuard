"""Statistics helpers: percentile bootstrap CIs, exact McNemar, Wilson intervals."""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Optional[Tuple[float, float]]:
    vals = list(values)
    n = len(vals)
    if n == 0:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(lo, 6), round(hi, 6))


def wilson(k: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 6), round(min(1.0, c + h), 6))


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Optional[float]:
    """Two-sided exact McNemar test for paired binary outcomes."""
    if len(a) != len(b):
        return None
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)
