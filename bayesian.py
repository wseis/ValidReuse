from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _sample_lognormal_posterior(vals_zu: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    log_vals = np.log(vals_zu)
    n = len(log_vals)
    mean_y = float(np.mean(log_vals))
    ssq = float(np.sum((log_vals - mean_y) ** 2))

    mu0 = mean_y
    kappa0 = 0.25
    alpha0 = 2.5
    beta0 = max(np.var(log_vals, ddof=1), 1e-6)

    kappa_n = kappa0 + n
    alpha_n = alpha0 + 0.5 * n
    beta_n = beta0 + 0.5 * ssq + (kappa0 * n * (mean_y - mu0) ** 2) / (2.0 * kappa_n)
    mu_n = (kappa0 * mu0 + n * mean_y) / kappa_n

    sigma2_draws = 1.0 / rng.gamma(shape=alpha_n, scale=1.0 / beta_n, size=draws)
    meanlog_draws = rng.normal(loc=mu_n, scale=np.sqrt(sigma2_draws / kappa_n))
    sdlog_draws = np.sqrt(sigma2_draws)
    return meanlog_draws, sdlog_draws


def _negbin_log_posterior(log_mu: float, log_size: float, vals_ab: np.ndarray) -> float:
    mu = np.exp(log_mu)
    size = np.exp(log_size)

    if not np.isfinite(mu) or not np.isfinite(size):
        return -np.inf

    lgamma = np.vectorize(math.lgamma)

    log_likelihood = np.sum(
        lgamma(vals_ab + size)
        - math.lgamma(size)
        - lgamma(vals_ab + 1.0)
        + size * (log_size - np.log(size + mu))
        + vals_ab * (log_mu - np.log(size + mu))
    )

    mean_anchor = np.log(np.mean(vals_ab) + 0.5)
    log_mu_prior = -0.5 * ((log_mu - mean_anchor) / 3.0) ** 2
    log_size_prior = -0.5 * (log_size / 2.5) ** 2
    return float(log_likelihood + log_mu_prior + log_size_prior)


def _sample_negative_binomial_posterior(
    vals_ab: np.ndarray,
    draws: int,
    warmup: int,
    chains: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    kept_log_mu: list[float] = []
    kept_log_size: list[float] = []
    total_steps = draws + warmup

    for chain in range(chains):
        chain_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        current = np.array([np.log(np.mean(vals_ab) + 0.5), 0.0], dtype=float)
        current_lp = _negbin_log_posterior(current[0], current[1], vals_ab)
        proposal_scale = np.array([0.12, 0.10], dtype=float)
        accepted_window = 0

        for step in range(total_steps):
            proposal = current + chain_rng.normal(loc=0.0, scale=proposal_scale, size=2)
            proposal_lp = _negbin_log_posterior(proposal[0], proposal[1], vals_ab)

            if np.log(chain_rng.random()) < proposal_lp - current_lp:
                current = proposal
                current_lp = proposal_lp
                accepted_window += 1

            if step < warmup and (step + 1) % 50 == 0:
                acceptance_rate = accepted_window / 50.0
                if acceptance_rate < 0.20:
                    proposal_scale *= 0.8
                elif acceptance_rate > 0.40:
                    proposal_scale *= 1.2
                accepted_window = 0

            if step >= warmup:
                kept_log_mu.append(current[0])
                kept_log_size.append(current[1])

    return np.exp(np.asarray(kept_log_mu)), np.exp(np.asarray(kept_log_size))


def bayesian_q10_lrv(
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
    rng = np.random.default_rng(seed)
    vals_zu_arr = np.asarray(list(vals_zu), dtype=float)
    vals_ab_arr = np.asarray(list(vals_ab), dtype=float)

    if np.any(vals_zu_arr <= 0):
        raise ValueError("Bayesian analysis requires vals_zu to be greater than 0.")

    q_prob = q / 100.0
    total_draws = draws * chains

    meanlog_draws, sdlog_draws = _sample_lognormal_posterior(vals_zu_arr, total_draws, rng)
    mu_draws, size_draws = _sample_negative_binomial_posterior(vals_ab_arr, draws, warmup, chains, rng)
    posterior_draw_count = min(len(meanlog_draws), len(mu_draws))
    q10_samples = np.empty(posterior_draw_count)

    for i in range(posterior_draw_count):
        in_sim = rng.lognormal(mean=meanlog_draws[i], sigma=sdlog_draws[i], size=n_sim)
        out_sim = rng.negative_binomial(size_draws[i], size_draws[i] / (size_draws[i] + mu_draws[i]), size=n_sim)
        out_sim = out_sim.astype(float)
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
