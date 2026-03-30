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


EXAMPLE_VALS_ZU = np.array(
    [
        90464631,
        264236258,
        228301585,
        43838430,
        94591158,
        94953995,
        42461661,
        56769779,
        75191460,
        9914308,
        81957051,
        20862649,
        45134087,
        36250848,
        20299364,
        37936171,
        23526431,
        31465798,
        16831883,
        37140659,
    ],
    dtype=int,
)

EXAMPLE_VALS_AB = np.array(
    [3, 9, 8, 2, 6, 7, 4, 6, 10, 2, 18, 5, 12, 10, 6, 14, 10, 23, 14, 107],
    dtype=int,
)


@dataclass
class MethodSummary:
    dataset_id: int
    method: str
    lower_bound: float
    median: float
    mean: float
    std_dev: float
    draws: int


def generate_dataset(sample_size: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    indices = rng.choice(len(EXAMPLE_VALS_ZU), size=sample_size, replace=True)
    vals_zu = EXAMPLE_VALS_ZU[indices].tolist()
    vals_ab = EXAMPLE_VALS_AB[indices].tolist()
    return vals_zu, vals_ab


def run_comparison(
    dataset_count: int = 10,
    sample_size: int = 12,
    bootstrap_samples: int = 1000,
    posterior_draws_per_chain: int = 1000,
    warmup_per_chain: int = 1000,
    chains: int = 2,
    n_sim: int = 4000,
    q: int = 10,
    alpha: float = 0.05,
    seed: int = 123,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    summaries: list[MethodSummary] = []
    dataset_rows: list[dict[str, object]] = []

    for dataset_id in range(1, dataset_count + 1):
        vals_zu, vals_ab = generate_dataset(sample_size=sample_size, rng=rng)
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "vals_zu": vals_zu,
                "vals_ab": vals_ab,
            }
        )

        bootstrap_result = bootstrap_q10_lrv(
            vals_zu=vals_zu,
            vals_ab=vals_ab,
            B=bootstrap_samples,
            n_sim=n_sim,
            q=q,
            alpha=alpha,
            add_one=True,
            seed=seed + dataset_id,
        )
        approx_result = bayesian_q10_lrv(
            vals_zu=vals_zu,
            vals_ab=vals_ab,
            draws=posterior_draws_per_chain,
            warmup=warmup_per_chain,
            chains=chains,
            n_sim=n_sim,
            q=q,
            alpha=alpha,
            add_one=True,
            seed=seed + dataset_id,
        )
        pymc_result = pymc_q10_lrv(
            vals_zu=vals_zu,
            vals_ab=vals_ab,
            draws=posterior_draws_per_chain,
            warmup=warmup_per_chain,
            chains=chains,
            n_sim=n_sim,
            q=q,
            alpha=alpha,
            add_one=True,
            seed=seed + dataset_id,
        )

        for method_name, result in [
            ("Bootstrap", bootstrap_result),
            ("Bayesian approximation", approx_result),
            ("Bayesian (PyMC)", pymc_result),
        ]:
            summaries.append(
                MethodSummary(
                    dataset_id=dataset_id,
                    method=method_name,
                    lower_bound=float(result["L_alpha"]),
                    median=float(result["median"]),
                    mean=float(result["mean"]),
                    std_dev=float(result["std_dev"]),
                    draws=len(result["q10_samples"]),
                )
            )

    summary_df = pd.DataFrame(asdict(item) for item in summaries)
    pivot_lower = summary_df.pivot(index="dataset_id", columns="method", values="lower_bound").reset_index()
    pivot_lower["approx_minus_pymc"] = (
        pivot_lower["Bayesian approximation"] - pivot_lower["Bayesian (PyMC)"]
    )
    pivot_lower["bootstrap_minus_pymc"] = pivot_lower["Bootstrap"] - pivot_lower["Bayesian (PyMC)"]

    aggregate_df = (
        summary_df.groupby("method")[["lower_bound", "median", "mean", "std_dev"]]
        .agg(["mean", "std"])
        .round(6)
    )

    return {
        "datasets": dataset_rows,
        "summary_rows": [asdict(item) for item in summaries],
        "lower_bound_comparison": pivot_lower.to_dict(orient="records"),
        "aggregate_summary": json.loads(aggregate_df.to_json()),
    }


if __name__ == "__main__":
    result = run_comparison()
    output_path = Path(__file__).parent / "comparison_results_10x12.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote comparison results to {output_path}")
