from __future__ import annotations

import os
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/pytensor,compiledir=/tmp/pytensor/compiledir")

import pymc as pm


def pymc_q10_lrv(
    vals_zu: Iterable[int],
    vals_ab: Iterable[int],
    draws: int = 600,
    warmup: int = 300,
    chains: int = 2,
    n_sim: int = 5000,
    q: int = 10,
    alpha: float = 0.05,
    add_one: bool = True,
    seed: int | None = None,
) -> dict[str, float | np.ndarray]:
    vals_zu_arr = np.asarray(list(vals_zu), dtype=float)
    vals_ab_arr = np.asarray(list(vals_ab), dtype=int)

    if np.any(vals_zu_arr <= 0):
        raise ValueError("PyMC Bayesian analysis requires vals_zu to be greater than 0.")

    log_zu = np.log(vals_zu_arr)

    with pm.Model() as model:
        meanlog_in = pm.Normal("meanlog_in", mu=float(np.mean(log_zu)), sigma=5.0)
        sdlog_in = pm.HalfNormal("sdlog_in", sigma=2.0)
        pm.Normal("log_zulauf_obs", mu=meanlog_in, sigma=sdlog_in, observed=log_zu)

        log_mu_out = pm.Normal("log_mu_out", mu=np.log(float(np.mean(vals_ab_arr) + 0.5)), sigma=5.0)
        mu_out = pm.Deterministic("mu_out", pm.math.exp(log_mu_out))
        alpha_out = pm.HalfNormal("alpha_out", sigma=10.0)
        pm.NegativeBinomial("ablauf_obs", mu=mu_out, alpha=alpha_out, observed=vals_ab_arr)

        trace = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
            target_accept=0.9,
        )

    posterior = trace.posterior
    meanlog_draws = posterior["meanlog_in"].values.reshape(-1)
    sdlog_draws = posterior["sdlog_in"].values.reshape(-1)
    mu_draws = posterior["mu_out"].values.reshape(-1)
    alpha_draws = posterior["alpha_out"].values.reshape(-1)

    rng = np.random.default_rng(seed)
    posterior_draw_count = min(len(meanlog_draws), len(sdlog_draws), len(mu_draws), len(alpha_draws))
    q10_samples = np.empty(posterior_draw_count)

    for i in range(posterior_draw_count):
        in_sim = rng.lognormal(mean=meanlog_draws[i], sigma=sdlog_draws[i], size=n_sim)
        prob = alpha_draws[i] / (alpha_draws[i] + mu_draws[i])
        out_sim = rng.negative_binomial(alpha_draws[i], prob, size=n_sim).astype(float)
        if add_one:
            out_sim = out_sim + 1.0

        lrv = np.log10(in_sim / out_sim)
        q10_samples[i] = np.percentile(lrv, q)

    return {
        "L_alpha": float(np.percentile(q10_samples, 100 * alpha)),
        "median": float(np.percentile(q10_samples, 50)),
        "upper_(1-alpha)": float(np.percentile(q10_samples, 100 * (1 - alpha))),
        "mean": float(np.mean(q10_samples)),
        "std_dev": float(np.std(q10_samples)),
        "q10_samples": q10_samples,
    }
