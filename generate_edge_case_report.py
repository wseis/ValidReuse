from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "edge_case_results.json"
REPORT_PATH = ROOT / "edge_case_report.md"
FIG_DIR = ROOT / "report_figures"

SCENARIO_ORDER = [
    "balanced_baseline",
    "many_zero_ab",
    "wide_spread_zu",
    "high_outlier_ab",
    "low_signal",
    "bimodal_ab",
]

METHOD_ORDER = [
    "Bootstrap",
    "Bayesian approximation",
    "Bayesian (PyMC)",
]

METHOD_COLORS = {
    "Bootstrap": "#2563eb",
    "Bayesian approximation": "#ea580c",
    "Bayesian (PyMC)": "#059669",
}


def load_frames() -> tuple[dict[str, str], pd.DataFrame]:
    data = json.loads(DATA_PATH.read_text())
    descriptions = data["scenario_descriptions"]
    summary_df = pd.DataFrame(data["summary_rows"])
    return descriptions, summary_df


def build_mean_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    table = (
        summary_df.groupby(["scenario", "method"])[["lower_bound", "median", "mean", "std_dev"]]
        .mean()
        .reset_index()
    )
    table["scenario"] = pd.Categorical(table["scenario"], categories=SCENARIO_ORDER, ordered=True)
    table["method"] = pd.Categorical(table["method"], categories=METHOD_ORDER, ordered=True)
    return table.sort_values(["scenario", "method"]).reset_index(drop=True)


def build_gap_table(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = (
        summary_df.pivot_table(index=["scenario", "replicate"], columns="method", values=metric)
        .reset_index()
    )
    pivot["approx_minus_pymc"] = pivot["Bayesian approximation"] - pivot["Bayesian (PyMC)"]
    pivot["bootstrap_minus_pymc"] = pivot["Bootstrap"] - pivot["Bayesian (PyMC)"]
    pivot["bootstrap_minus_approx"] = pivot["Bootstrap"] - pivot["Bayesian approximation"]
    gap = (
        pivot.groupby("scenario")[["approx_minus_pymc", "bootstrap_minus_pymc", "bootstrap_minus_approx"]]
        .mean()
        .reset_index()
    )
    gap["scenario"] = pd.Categorical(gap["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return gap.sort_values("scenario").reset_index(drop=True)


def save_grouped_bar(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.23
    x = range(len(SCENARIO_ORDER))

    for idx, method in enumerate(METHOD_ORDER):
        subset = table[table["method"] == method].set_index("scenario").loc[SCENARIO_ORDER]
        positions = [value + (idx - 1) * width for value in x]
        ax.bar(
            positions,
            subset[metric],
            width=width,
            label=method,
            color=METHOD_COLORS[method],
            alpha=0.9,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(SCENARIO_ORDER, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_gap_heatmap(gap_df: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    heatmap_values = gap_df[["approx_minus_pymc", "bootstrap_minus_pymc", "bootstrap_minus_approx"]].to_numpy()
    image = ax.imshow(heatmap_values, cmap="coolwarm", aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["Approx - PyMC", "Bootstrap - PyMC", "Bootstrap - Approx"])
    ax.set_yticks(range(len(gap_df)))
    ax.set_yticklabels(gap_df["scenario"])
    ax.set_title(title)

    for row in range(heatmap_values.shape[0]):
        for col in range(heatmap_values.shape[1]):
            ax.text(col, row, f"{heatmap_values[row, col]:.3f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, shrink=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, rename_map: dict[str, str]) -> str:
    display_df = df.rename(columns=rename_map).copy()
    numeric_cols = display_df.select_dtypes(include="number").columns
    for column in numeric_cols:
        display_df[column] = display_df[column].map(lambda value: f"{value:.3f}")

    headers = list(display_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")

    return "\n".join(lines)


def build_report() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    descriptions, summary_df = load_frames()

    mean_table = build_mean_table(summary_df)
    lower_gap_table = build_gap_table(summary_df, "lower_bound")
    median_gap_table = build_gap_table(summary_df, "median")

    save_grouped_bar(
        mean_table,
        metric="lower_bound",
        ylabel="Mean L_alpha",
        title="Mean Lower Bound by Scenario and Method",
        filename="lower_bound_means.png",
    )
    save_grouped_bar(
        mean_table,
        metric="median",
        ylabel="Mean Median",
        title="Mean Median by Scenario and Method",
        filename="median_means.png",
    )
    save_gap_heatmap(
        lower_gap_table,
        title="Lower-Bound Gap Heatmap",
        filename="lower_bound_gap_heatmap.png",
    )
    save_gap_heatmap(
        median_gap_table,
        title="Median Gap Heatmap",
        filename="median_gap_heatmap.png",
    )

    scenario_lines = "\n".join(
        f"- `{scenario}`: {descriptions[scenario]}" for scenario in SCENARIO_ORDER
    )

    report = f"""# Edge-Case Method Comparison Report

This report summarizes the synthetic edge-case study comparing:

- `Bootstrap`
- `Bayesian approximation`
- `Bayesian (PyMC)`

Each scenario contains 5 synthetic datasets of sample size 12. Reported method statistics are scenario-level means across those 5 replicates.

## Scenario Guide

{scenario_lines}

## Method Statistics

{markdown_table(
        mean_table,
        {
            "scenario": "Scenario",
            "method": "Method",
            "lower_bound": "Mean L_alpha",
            "median": "Mean Median",
            "mean": "Mean Mean",
            "std_dev": "Mean SD",
        },
    )}

## Lower-Bound Gap Table

{markdown_table(
        lower_gap_table,
        {
            "scenario": "Scenario",
            "approx_minus_pymc": "Approx - PyMC",
            "bootstrap_minus_pymc": "Bootstrap - PyMC",
            "bootstrap_minus_approx": "Bootstrap - Approx",
        },
    )}

## Median Gap Table

{markdown_table(
        median_gap_table,
        {
            "scenario": "Scenario",
            "approx_minus_pymc": "Approx - PyMC",
            "bootstrap_minus_pymc": "Bootstrap - PyMC",
            "bootstrap_minus_approx": "Bootstrap - Approx",
        },
    )}

## Visualizations

### Mean Lower Bound by Scenario

![Mean lower bound comparison](report_figures/lower_bound_means.png)

### Mean Median by Scenario

![Mean median comparison](report_figures/median_means.png)

### Lower-Bound Gap Heatmap

![Lower-bound gap heatmap](report_figures/lower_bound_gap_heatmap.png)

### Median Gap Heatmap

![Median gap heatmap](report_figures/median_gap_heatmap.png)
"""

    REPORT_PATH.write_text(report)


if __name__ == "__main__":
    build_report()
    print(f"Wrote report to {REPORT_PATH}")
