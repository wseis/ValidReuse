from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bayesian import bayesian_q10_lrv
from bootstrap import bootstrap_q10_lrv
from pymc_bayesian import pymc_q10_lrv

logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)


@dataclass
class ScenarioSummary:
    scenario: str
    replicate: int
    method: str
    lower_bound: float
    median: float
    mean: float
    std_dev: float
    draws: int


def generate_balanced_baseline(rng: np.random.Generator, sample_size: int) -> tuple[list[int], list[int]]:
    vals_zu = np.rint(rng.lognormal(mean=18.0, sigma=0.45, size=sample_size)).astype(int)
    vals_ab = rng.negative_binomial(n=7, p=0.52, size=sample_size).astype(int)
    return vals_zu.tolist(), vals_ab.tolist()


def generate_many_zero_ab(rng: np.random.Generator, sample_size: int) -> tuple[list[int], list[int]]:
    vals_zu = np.rint(rng.lognormal(mean=17.8, sigma=0.5, size=sample_size)).astype(int)
    zero_mask = rng.random(sample_size) < 0.6
    positive_part = rng.poisson(lam=3.0, size=sample_size)
    vals_ab = np.where(zero_mask, 0, positive_part).astype(int)
    return vals_zu.tolist(), vals_ab.tolist()


def generate_wide_spread_zu(rng: np.random.Generator, sample_size: int) -> tuple[list[int], list[int]]:
    vals_zu = np.rint(rng.lognormal(mean=17.7, sigma=1.1, size=sample_size)).astype(int)
    vals_ab = rng.negative_binomial(n=8, p=0.55, size=sample_size).astype(int)
    return vals_zu.tolist(), vals_ab.tolist()


def generate_high_outlier_ab(rng: np.random.Generator, sample_size: int) -> tuple[list[int], list[int]]:
    vals_zu = np.rint(rng.lognormal(mean=17.9, sigma=0.45, size=sample_size)).astype(int)
    vals_ab = rng.negative_binomial(n=9, p=0.58, size=sample_size).astype(int)
    outlier_idx = rng.integers(0, sample_size)
    vals_ab[outlier_idx] = int(vals_ab[outlier_idx] + rng.integers(60, 140))
    return vals_zu.tolist(), vals_ab.tolist()


def generate_low_signal(rng: np.random.Generator, sample_size: int) -> tuple[list[int], list[int]]:
    vals_zu = np.rint(rng.lognormal(mean=16.1, sigma=0.28, size=sample_size)).astype(int)
    vals_ab = rng.negative_binomial(n=12, p=0.2, size=sample_size).astype(int)
    return vals_zu.tolist(), vals_ab.tolist()


def generate_bimodal_ab(rng: np.random.Generator, sample_size: int) -> tuple[list[int], list[int]]:
    vals_zu = np.rint(rng.lognormal(mean=18.0, sigma=0.42, size=sample_size)).astype(int)
    cluster_mask = rng.random(sample_size) < 0.5
    vals_ab = np.where(
        cluster_mask,
        rng.poisson(lam=1.0, size=sample_size),
        rng.poisson(lam=18.0, size=sample_size),
    ).astype(int)
    return vals_zu.tolist(), vals_ab.tolist()


SCENARIOS = {
    "balanced_baseline": generate_balanced_baseline,
    "many_zero_ab": generate_many_zero_ab,
    "wide_spread_zu": generate_wide_spread_zu,
    "high_outlier_ab": generate_high_outlier_ab,
    "low_signal": generate_low_signal,
    "bimodal_ab": generate_bimodal_ab,
}

SCENARIO_DESCRIPTIONS = {
    "balanced_baseline": "Moderately spread positive vals_zu with ordinary count-like vals_ab and no special pathologies.",
    "many_zero_ab": "vals_ab contains many zeros, stressing zero inflation and denominator floor effects after the +1 adjustment.",
    "wide_spread_zu": "vals_zu has very large multiplicative spread, creating heavy skew and occasional extreme inflow values.",
    "high_outlier_ab": "vals_ab is mostly ordinary counts but includes one very large outlier, testing robustness to extreme output observations.",
    "low_signal": "vals_zu is lower and tighter while vals_ab is relatively larger, producing weaker reduction and less separation.",
    "bimodal_ab": "vals_ab mixes a near-zero cluster with a much higher cluster, creating a two-mode denominator distribution.",
}


def run_edge_case_comparison(
    sample_size: int = 12,
    replicates_per_scenario: int = 5,
    bootstrap_samples: int = 1000,
    posterior_draws_per_chain: int = 1000,
    warmup_per_chain: int = 1000,
    chains: int = 2,
    n_sim: int = 4000,
    q: int = 10,
    alpha: float = 0.05,
    seed: int = 321,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    summaries: list[ScenarioSummary] = []
    datasets: list[dict[str, object]] = []

    for scenario_name, generator in SCENARIOS.items():
        for replicate in range(1, replicates_per_scenario + 1):
            vals_zu, vals_ab = generator(rng, sample_size)
            datasets.append(
                {
                    "scenario": scenario_name,
                    "replicate": replicate,
                    "vals_zu": vals_zu,
                    "vals_ab": vals_ab,
                }
            )

            shared_seed = seed + replicate * 100 + len(summaries)
            method_outputs = [
                (
                    "Bootstrap",
                    bootstrap_q10_lrv(
                        vals_zu=vals_zu,
                        vals_ab=vals_ab,
                        B=bootstrap_samples,
                        n_sim=n_sim,
                        q=q,
                        alpha=alpha,
                        add_one=True,
                        seed=shared_seed,
                    ),
                ),
                (
                    "Bayesian approximation",
                    bayesian_q10_lrv(
                        vals_zu=vals_zu,
                        vals_ab=vals_ab,
                        draws=posterior_draws_per_chain,
                        warmup=warmup_per_chain,
                        chains=chains,
                        n_sim=n_sim,
                        q=q,
                        alpha=alpha,
                        add_one=True,
                        seed=shared_seed,
                    ),
                ),
                (
                    "Bayesian (PyMC)",
                    pymc_q10_lrv(
                        vals_zu=vals_zu,
                        vals_ab=vals_ab,
                        draws=posterior_draws_per_chain,
                        warmup=warmup_per_chain,
                        chains=chains,
                        n_sim=n_sim,
                        q=q,
                        alpha=alpha,
                        add_one=True,
                        seed=shared_seed,
                    ),
                ),
            ]

            for method_name, result in method_outputs:
                summaries.append(
                    ScenarioSummary(
                        scenario=scenario_name,
                        replicate=replicate,
                        method=method_name,
                        lower_bound=float(result["L_alpha"]),
                        median=float(result["median"]),
                        mean=float(result["mean"]),
                        std_dev=float(result["std_dev"]),
                        draws=len(result["q10_samples"]),
                    )
                )

    summary_df = pd.DataFrame(asdict(row) for row in summaries)

    aggregate_by_scenario = (
        summary_df.groupby(["scenario", "method"])[["lower_bound", "median", "mean", "std_dev"]]
        .agg(["mean", "std"])
        .round(6)
    )

    lb_pivot = summary_df.pivot_table(
        index=["scenario", "replicate"],
        columns="method",
        values="lower_bound",
    ).reset_index()
    lb_pivot["approx_minus_pymc"] = lb_pivot["Bayesian approximation"] - lb_pivot["Bayesian (PyMC)"]
    lb_pivot["bootstrap_minus_pymc"] = lb_pivot["Bootstrap"] - lb_pivot["Bayesian (PyMC)"]
    lb_pivot["bootstrap_minus_approx"] = lb_pivot["Bootstrap"] - lb_pivot["Bayesian approximation"]

    gap_summary = (
        lb_pivot.groupby("scenario")[["approx_minus_pymc", "bootstrap_minus_pymc", "bootstrap_minus_approx"]]
        .agg(["mean", "std", "max"])
        .round(6)
    )

    return {
        "scenario_descriptions": SCENARIO_DESCRIPTIONS,
        "datasets": datasets,
        "summary_rows": [asdict(row) for row in summaries],
        "aggregate_by_scenario": json.loads(aggregate_by_scenario.to_json()),
        "lower_bound_gaps": lb_pivot.to_dict(orient="records"),
        "gap_summary": json.loads(gap_summary.to_json()),
    }


if __name__ == "__main__":
    result = run_edge_case_comparison()
    output_path = Path("/Users/wseis/Projects/ValidReuse/edge_case_results.json")
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote edge-case comparison results to {output_path}")
