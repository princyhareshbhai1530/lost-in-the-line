"""Paired cluster bootstrap. The cluster is the sentence (fleurs_id): every language reads the same 40 sentences,
so resampling sentences keeps paired comparisons (auto vs hint, T1 vs T2) and cross-language pooling honest.

Within one language each sentence has one clip, so this equals a clip bootstrap there; pooled over languages,
all clips of a chosen sentence come along together. 2,000 reps, seed 42 (config.py).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

import config


def cluster_boot(rows: list[dict], metric: Callable[[list[dict]], float | None],
                 reps: int = config.REPS, seed: int = config.SEED) -> dict:
    """Estimate and 95% percentile interval of metric(rows), resampling fleurs_id clusters with replacement."""
    est = metric(rows)
    if est is None or not rows:
        return {"est": None, "lo": None, "hi": None, "n": len(rows)}
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r["fleurs_id"])].append(r)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(ids), len(ids), replace=True)
        sample = [r for k in pick for r in groups[ids[k]]]
        v = metric(sample)
        if v is not None:
            vals.append(v)
    if not vals:
        return {"est": est, "lo": None, "hi": None, "n": len(rows)}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"est": float(est), "lo": float(lo), "hi": float(hi), "n": len(rows), "n_clusters": len(ids)}


def mean_of(key: str):
    def f(rows):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    return f


def share(pred):
    def f(rows):
        return float(np.mean([1.0 if pred(r) else 0.0 for r in rows])) if rows else None
    return f


def ratio(num_key: str, den_key: str):
    def f(rows):
        d = sum(r.get(den_key) or 0 for r in rows)
        return (sum(r.get(num_key) or 0 for r in rows) / d) if d else None
    return f


def rnd(b: dict, k: int = 4) -> dict:
    return {kk: (round(v, k) if isinstance(v, float) else v) for kk, v in b.items()}
