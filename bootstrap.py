from __future__ import annotations

from typing import Iterable

import numpy as np


def bootstrap_q10_lrv(
    vals_zu: Iterable[int],
    vals_ab: Iterable[int],
    B: int = 1000,
    n_sim: int = 5000,
    q: int = 10,
    alpha: float = 0.05,
    add_one: bool = True,
    seed: int | None = None,
) -> dict[str, float | np.ndarray]:
    rng = np.random.default_rng(seed)

    vals_zu = np.asarray(list(vals_zu), dtype=float)
    vals_ab = np.asarray(list(vals_ab), dtype=float)

    n_zu = len(vals_zu)
    n_ab = len(vals_ab)

    q10s = np.empty(B)

    for b in range(B):
        zu_b = rng.choice(vals_zu, size=n_zu, replace=True)
        ab_b = rng.choice(vals_ab, size=n_ab, replace=True)

        zu_sim = rng.choice(zu_b, size=n_sim, replace=True)
        ab_sim = rng.choice(ab_b, size=n_sim, replace=True)

        if add_one:
            ab_sim = ab_sim + 1.0

        lrv = np.log10(zu_sim / ab_sim)
        q10s[b] = np.percentile(lrv, q)

    return {
        "L_alpha": float(np.percentile(q10s, 100 * alpha)),
        "median": float(np.percentile(q10s, 50)),
        "upper_(1-alpha)": float(np.percentile(q10s, 100 * (1 - alpha))),
        "mean": float(np.mean(q10s)),
        "std_dev": float(np.std(q10s)),
        "q10_samples": q10s,
    }
