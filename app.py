from __future__ import annotations

import io
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import warnings

from bayesian import bayesian_q10_lrv
from bootstrap import bootstrap_q10_lrv
from pymc_bayesian import PYMC_AVAILABLE, pymc_q10_lrv, pymc_q10_lrv_batch


METHOD_OPTIONS = {
    "Empirischer Bootstrap": "bootstrap",
    "Bayessche Approximation": "bayesian",
}

if PYMC_AVAILABLE:
    METHOD_OPTIONS["Bayessch (PyMC)"] = "pymc"

PARAMETERS = [
    {
        "id": "ecoli",
        "name": "E. coli",
        "display_name": "*E. coli*",
        "zulauf_col": "ecoli_zulauf",
        "ablauf_col": "ecoli_ablauf",
        "target": 5.0,
    },
    {
        "id": "cperfringens",
        "name": "Sporen C. perfringens",
        "display_name": "Sporen *C. perfringens*",
        "zulauf_col": "cperfringens_zulauf",
        "ablauf_col": "cperfringens_ablauf",
        "target": 4.0,
    },
    {
        "id": "somatische_coliphagen",
        "name": "Somatische Coliphagen",
        "display_name": "Somatische Coliphagen",
        "zulauf_col": "somatische_zulauf",
        "ablauf_col": "somatische_ablauf",
        "target": 6.0,
    },
    {
        "id": "fspezifische_coliphagen",
        "name": "F-spezifische Coliphagen",
        "display_name": "F-spezifische Coliphagen",
        "zulauf_col": "fspezifische_zulauf",
        "ablauf_col": "fspezifische_ablauf",
        "target": 6.0,
    },
]


def clean_integer_series(series: pd.Series, column_name: str) -> list[int]:
    values: list[int] = []

    for index, raw_value in series.items():
        if pd.isna(raw_value) or raw_value == "":
            continue

        numeric_value = float(raw_value)
        if not numeric_value.is_integer():
            raise ValueError(f"{column_name} Zeile {index + 1} muss eine ganze Zahl sein.")

        int_value = int(numeric_value)
        if int_value < 0:
            raise ValueError(f"{column_name} Zeile {index + 1} muss 0 oder groesser sein.")

        values.append(int_value)

    return values


def build_default_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecoli_zulauf": [8400000, 9100000, 8750000, 9600000, 8900000, 9300000, 9050000, 9800000, 9200000, 9500000, 8850000, 9700000, 8990000, 9400000, 9120000, 9680000, None, None, None, None],
            "ecoli_ablauf": [18, 12, 15, 10, 14, 11, 13, 9, 16, 10, 15, 8, 14, 12, 11, 9, None, None, None, None],
            "cperfringens_zulauf": [62000, 58000, 64000, 60500, 59000, 63000, 61500, 60000, 65000, 62500, 59800, 61200, 63500, 60700, 62100, 64300, None, None, None, None],
            "cperfringens_ablauf": [140, 120, 135, 110, 150, 125, 130, 118, 145, 128, 132, 115, 138, 122, 127, 134, None, None, None, None],
            "somatische_zulauf": [14500000, 15200000, 14850000, 15500000, 14900000, 15100000, 14650000, 15350000, 14750000, 15400000, 15050000, 15600000, 14800000, 15250000, 14950000, 15550000, None, None, None, None],
            "somatische_ablauf": [6, 4, 5, 3, 4, 5, 4, 3, 5, 4, 6, 3, 4, 5, 4, 3, None, None, None, None],
            "fspezifische_zulauf": [11200000, 11800000, 11550000, 12100000, 11700000, 11950000, 11400000, 12200000, 11600000, 12050000, 11350000, 12300000, 11750000, 11850000, 11500000, 12150000, None, None, None, None],
            "fspezifische_ablauf": [5, 4, 6, 3, 5, 4, 5, 3, 4, 5, 6, 3, 5, 4, 4, 3, None, None, None, None],
        }
    )


def build_empty_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ecoli_zulauf": [None] * 20,
            "ecoli_ablauf": [None] * 20,
            "cperfringens_zulauf": [None] * 20,
            "cperfringens_ablauf": [None] * 20,
            "somatische_zulauf": [None] * 20,
            "somatische_ablauf": [None] * 20,
            "fspezifische_zulauf": [None] * 20,
            "fspezifische_ablauf": [None] * 20,
        }
    )


def build_parameter_table(input_df: pd.DataFrame, parameter: dict[str, object]) -> pd.DataFrame:
    return input_df[[parameter["zulauf_col"], parameter["ablauf_col"]]].rename(
        columns={
            parameter["zulauf_col"]: "Zulaufwerte",
            parameter["ablauf_col"]: "Ablaufwerte",
        }
    )


def parameter_table_has_values(parameter_df: pd.DataFrame) -> bool:
    return parameter_df.notna().any().any()


def initialize_input_state(force_defaults: bool = False) -> None:
    seed_mode = st.session_state.get("input_seed_mode", "default")
    default_df = build_default_table() if seed_mode == "default" else build_empty_table()

    if force_defaults or "input_df" not in st.session_state:
        st.session_state["input_df"] = default_df.copy()

    should_seed_defaults = force_defaults
    if not force_defaults:
        existing_tables: list[pd.DataFrame] = []
        for parameter in PARAMETERS:
            data_key = f"input_data_{parameter['id']}"
            if data_key in st.session_state and isinstance(st.session_state[data_key], pd.DataFrame):
                existing_tables.append(st.session_state[data_key])

        if not existing_tables:
            should_seed_defaults = True
        else:
            should_seed_defaults = not any(parameter_table_has_values(table) for table in existing_tables)

    for parameter in PARAMETERS:
        data_key = f"input_data_{parameter['id']}"
        editor_key = f"editor_v{st.session_state.get('editor_version', 2)}_{parameter['id']}"
        parameter_default_df = build_parameter_table(default_df, parameter)

        if should_seed_defaults or data_key not in st.session_state:
            st.session_state[data_key] = parameter_default_df.copy()
            st.session_state.pop(editor_key, None)


def clear_input_state() -> None:
    empty_df = build_empty_table()
    st.session_state["input_df"] = empty_df.copy()
    for parameter in PARAMETERS:
        data_key = f"input_data_{parameter['id']}"
        editor_key = f"editor_v{st.session_state.get('editor_version', 2)}_{parameter['id']}"
        st.session_state[data_key] = build_parameter_table(empty_df, parameter)
        st.session_state.pop(editor_key, None)


def initialize_target_state() -> None:
    if "target_values" not in st.session_state:
        st.session_state["target_values"] = {parameter["id"]: parameter["target"] for parameter in PARAMETERS}
    if "target_table" not in st.session_state:
        st.session_state["target_table"] = pd.DataFrame(
            {
                "Parameter": [parameter["name"] for parameter in PARAMETERS],
                "Validierungszielwert": [float(st.session_state["target_values"][parameter["id"]]) for parameter in PARAMETERS],
            }
        )


def sync_target_state_from_table() -> None:
    target_table = st.session_state["target_table"]
    for index, parameter in enumerate(PARAMETERS):
        value = float(target_table.iloc[index]["Validierungszielwert"])
        st.session_state["target_values"][parameter["id"]] = round(value, 1)


def load_example_data() -> None:
    st.session_state["editor_version"] = st.session_state.get("editor_version", 2) + 1
    st.session_state["input_seed_mode"] = "default"
    initialize_input_state(force_defaults=True)
    st.session_state.pop("analysis_results", None)


def remove_example_data() -> None:
    st.session_state["editor_version"] = st.session_state.get("editor_version", 2) + 1
    st.session_state["input_seed_mode"] = "empty"
    clear_input_state()
    st.session_state.pop("analysis_results", None)
    st.session_state.pop("report_pdf", None)


def sync_input_tables_from_parameter_state() -> None:
    combined_df = build_default_table()
    for parameter in PARAMETERS:
        data_key = f"input_data_{parameter['id']}"
        if data_key not in st.session_state:
            st.session_state[data_key] = build_parameter_table(combined_df, parameter)
        parameter_df = st.session_state[data_key].rename(
            columns={
                "Zulaufwerte": parameter["zulauf_col"],
                "Ablaufwerte": parameter["ablauf_col"],
            }
        )
        combined_df.loc[:, parameter["zulauf_col"]] = parameter_df[parameter["zulauf_col"]]
        combined_df.loc[:, parameter["ablauf_col"]] = parameter_df[parameter["ablauf_col"]]
    st.session_state["input_df"] = combined_df


def build_histogram_chart(chart_df: pd.DataFrame, summary_df: pd.DataFrame, q: int) -> plt.Figure:
    colors = {
        "Empirischer Bootstrap": "#2563eb",
        "Bayessche Approximation": "#ea580c",
        "Bayessch (PyMC)": "#059669",
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    all_samples = chart_df["q10_sample"].to_numpy(dtype=float)
    if all_samples.size == 0:
        return fig

    bins = np.histogram_bin_edges(all_samples, bins=32)

    for method_name, method_df in chart_df.groupby("Method"):
        ax.hist(
            method_df["q10_sample"].to_numpy(dtype=float),
            bins=bins,
            density=True,
            alpha=0.4,
            label=method_name,
            color=colors.get(method_name, "#2563eb"),
            edgecolor="white",
            linewidth=0.7,
        )

    for _, row in summary_df.iterrows():
        ax.axvline(
            float(row["Lower Bound"]),
            color=colors.get(row["Method"], "#2563eb"),
            linestyle="--",
            linewidth=2,
        )

    ax.axvline(float(summary_df["Zielwert"].iloc[0]), color="#7c3aed", linewidth=3, label="Zielwert")
    ax.set_title(f"q{q}-Verteilung der Simulationswerte", fontsize=16)
    ax.set_xlabel("q-Perzentil der Logreduktion", fontsize=14)
    ax.set_ylabel("Dichte", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_facecolor((251 / 255, 252 / 255, 255 / 255, 0.95))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=11)
    fig.patch.set_alpha(0)
    fig.tight_layout()
    return fig


def execute_analysis_task(task: dict[str, object]) -> tuple[str, str, dict[str, float | np.ndarray]]:
    result = run_method(
        method_key=task["method_key"],
        vals_zu=task["vals_zu"],
        vals_ab=task["vals_ab"],
        q=task["q"],
        alpha=task["alpha"],
        n_sim=task["n_sim"],
        seed_value=task["seed_value"],
        B=task["B"],
        add_one=task["add_one"],
        posterior_draws=task["posterior_draws"],
        warmup=task["warmup"],
        chains=task["chains"],
        pymc_draws=task["pymc_draws"],
        pymc_warmup=task["pymc_warmup"],
        pymc_chains=task["pymc_chains"],
        add_one_bayes=task["add_one_bayes"],
        add_one_pymc=task["add_one_pymc"],
    )
    return task["parameter_id"], task["label"], result


def build_validation_report_pdf(
    summary_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    q: int,
    selected_methods: list[str],
) -> bytes:
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        both_pass = int(
            summary_df[
                (summary_df["Lower Bound >= Zielwert"] == "Ja") & (summary_df["Median >= Zielwert"] == "Ja")
            ].shape[0]
        )
        total_rows = int(summary_df.shape[0])
        parameter_count = int(summary_df["Parameter"].nunique())

        summary_page, summary_ax = plt.subplots(figsize=(11.69, 8.27))
        summary_ax.axis("off")
        summary_page.patch.set_facecolor("white")
        summary_ax.add_patch(
            plt.Rectangle((0.015, 0.90), 0.97, 0.09, color="#e8efff", transform=summary_ax.transAxes, zorder=0)
        )
        summary_ax.text(0.03, 0.955, "Reuse-Validierungsreport", fontsize=22, fontweight="bold", va="top", color="#132238")
        summary_ax.text(
            0.03,
            0.90,
            f"Erstellt am: {datetime.now().strftime('%d.%m.%Y %H:%M')}   |   Methoden: {', '.join(selected_methods)}",
            fontsize=10,
            color="#475569",
            va="top",
        )
        summary_ax.text(
            0.03,
            0.84,
            "Kurzfazit",
            fontsize=14,
            fontweight="bold",
            color="#132238",
            va="top",
        )

        metric_boxes = [
            ("Parameter", str(parameter_count), "#dbeafe"),
            ("Berechnete Kombinationen", str(total_rows), "#e0f2fe"),
            ("Beide Kriterien erfuellt", str(both_pass), "#dcfce7" if both_pass > 0 else "#fee2e2"),
        ]
        x_positions = [0.03, 0.27, 0.56]
        box_widths = [0.18, 0.24, 0.24]
        for (label, value, color), x_pos, width in zip(metric_boxes, x_positions, box_widths):
            summary_ax.add_patch(
                plt.Rectangle((x_pos, 0.69), width, 0.09, color=color, transform=summary_ax.transAxes, ec="none")
            )
            summary_ax.text(x_pos + 0.015, 0.745, label, fontsize=9.5, color="#475569", va="center")
            summary_ax.text(x_pos + 0.015, 0.708, value, fontsize=17, fontweight="bold", color="#132238", va="center")

        display_df = summary_df.copy()
        for column in ["Zielwert", "Lower Bound", "Median", "Obergrenze", "Mittelwert", "Standardabweichung"]:
            display_df[column] = display_df[column].map(lambda value: f"{value:.1f}" if column == "Zielwert" else f"{value:.4f}")
        display_df["Parameter"] = display_df["Parameter"].map(lambda value: textwrap.fill(value, width=18))
        display_df["Methode"] = display_df["Methode"].map(lambda value: textwrap.fill(value, width=16))

        display_df = display_df[
            [
                "Parameter",
                "Methode",
                "Zielwert",
                "Lower Bound",
                "Median",
                "Lower Bound >= Zielwert",
                "Median >= Zielwert",
            ]
        ]

        wrapped_col_labels = [textwrap.fill(str(label), width=16) for label in display_df.columns]

        table = summary_ax.table(
            cellText=display_df.values,
            colLabels=wrapped_col_labels,
            loc="upper left",
            cellLoc="center",
            bbox=[0.03, 0.05, 0.94, 0.57],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d7deed")
            if row == 0:
                cell.set_facecolor("#e8efff")
                cell.set_text_props(weight="bold", color="#132238")
            else:
                cell.set_facecolor("#fbfcff" if row % 2 else "#f5f8ff")
                if col in (5, 6):
                    passed = cell.get_text().get_text() == "Ja"
                    cell.set_facecolor("#dcfce7" if passed else "#fee2e2")

        pdf.savefig(summary_page, bbox_inches="tight")
        plt.close(summary_page)

        for parameter_name in summary_df["Parameter"].drop_duplicates():
            parameter_chart_df = distribution_df[distribution_df["Parameter"] == parameter_name]
            parameter_result_df = summary_df[summary_df["Parameter"] == parameter_name].copy()
            parameter_summary_df = parameter_result_df[
                ["Methode", "Lower Bound", "Zielwert"]
            ].rename(columns={"Methode": "Method"})
            figure = build_histogram_chart(parameter_chart_df, parameter_summary_df, q)
            figure.set_size_inches(11.69, 8.27)
            figure.subplots_adjust(top=0.66, bottom=0.31)
            figure.suptitle(parameter_name, fontsize=19, fontweight="bold", y=0.975)
            figure.text(
                0.125,
                0.92,
                f"q{q}-Verteilung der Simulationswerte",
                fontsize=15,
                fontweight="bold",
                color="#132238",
                va="top",
            )
            figure.text(
                0.125,
                0.885,
                f"Zielwert: {parameter_result_df['Zielwert'].iloc[0]:.1f}   |   Methoden: {', '.join(parameter_result_df['Methode'].tolist())}",
                fontsize=11,
                color="#475569",
                va="top",
            )
            mini_rows = [
                f"{row['Methode']}: Lower Bound {row['Lower Bound']:.4f}, Median {row['Median']:.4f}, "
                f"LB-Ziel {'erfuellt' if row['Lower Bound >= Zielwert'] == 'Ja' else 'nicht erfuellt'}, "
                f"Median-Ziel {'erfuellt' if row['Median >= Zielwert'] == 'Ja' else 'nicht erfuellt'}"
                for _, row in parameter_result_df.iterrows()
            ]
            figure.axes[0].text(
                0.0,
                -0.30,
                "\n".join(mini_rows),
                transform=figure.axes[0].transAxes,
                fontsize=10.5,
                color="#334155",
                va="top",
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    buffer.seek(0)
    return buffer.getvalue()


def run_method(
    method_key: str,
    vals_zu: list[int],
    vals_ab: list[int],
    q: int,
    alpha: float,
    n_sim: int,
    seed_value: int,
    B: int,
    add_one: bool,
    posterior_draws: int,
    warmup: int,
    chains: int,
    pymc_draws: int,
    pymc_warmup: int,
    pymc_chains: int,
    add_one_bayes: bool,
    add_one_pymc: bool,
) -> dict[str, float | list[float]]:
    if method_key == "bootstrap":
        return bootstrap_q10_lrv(
            vals_zu=vals_zu,
            vals_ab=vals_ab,
            B=B,
            n_sim=n_sim,
            q=q,
            alpha=alpha,
            add_one=add_one,
            seed=seed_value,
        )
    if method_key == "bayesian":
        return bayesian_q10_lrv(
            vals_zu=vals_zu,
            vals_ab=vals_ab,
            draws=posterior_draws,
            warmup=warmup,
            chains=chains,
            n_sim=n_sim,
            q=q,
            alpha=alpha,
            add_one=add_one_bayes,
            seed=seed_value,
        )
    return pymc_q10_lrv(
        vals_zu=vals_zu,
        vals_ab=vals_ab,
        draws=pymc_draws,
        warmup=pymc_warmup,
        chains=pymc_chains,
        n_sim=n_sim,
        q=q,
        alpha=alpha,
        add_one=add_one_pymc,
        seed=seed_value,
    )


st.set_page_config(page_title="KA-Validierungsrechner", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(143, 179, 255, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(67, 97, 238, 0.18), transparent 25%),
            linear-gradient(180deg, rgb(246, 246, 254) 0%, rgb(246, 246, 254) 100%);
        color: #1f2937;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 28px;
        padding: 1.75rem;
        box-shadow: 0 20px 50px rgba(148, 163, 184, 0.18);
        backdrop-filter: blur(12px);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        letter-spacing: -0.03em;
        color: #132238;
    }
    .hero p {
        margin-top: 0.75rem;
        margin-bottom: 0;
        font-size: 1rem;
        color: #475569;
        max-width: 68rem;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 24px;
        padding: 1.2rem;
        box-shadow: 0 18px 40px rgba(148, 163, 184, 0.12);
    }
    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.82));
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 16px 38px rgba(148, 163, 184, 0.16);
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
    }
    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataEditor"] > div {
        background: rgba(251, 252, 255, 0.96);
        border-radius: 18px;
    }
    [data-testid="stDataFrame"] [role="grid"],
    [data-testid="stDataEditor"] [role="grid"] {
        background: rgba(251, 252, 255, 0.98);
    }
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataEditor"] [role="columnheader"] {
        background: rgba(228, 236, 255, 0.92);
        color: #132238;
    }
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataEditor"] [role="gridcell"] {
        background: rgba(251, 252, 255, 0.9);
    }
    [data-testid="stExpander"] {
        background: rgba(251, 252, 255, 0.94);
        border: 1px solid rgba(37, 99, 235, 0.08);
        border-radius: 18px;
    }
    [data-testid="stExpander"] summary {
        background: rgba(228, 236, 255, 0.95);
        border-radius: 16px;
        color: #132238;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>KA-Validierungsrechner</h1>
        <p>
            Berechnet wird die Logreduktion mikrobiologischer Daten waehrend der Abwasserreinigung
            fuer <strong><em>E. coli</em></strong>, <strong>Sporen <em>C. perfringens</em></strong>,
            <strong>Somatische Coliphagen</strong> und <strong>F-spezifische Coliphagen</strong>.
            Fuer jeden Parameter koennen Median und Lower Bound direkt mit den jeweiligen Validierungszielwerten verglichen werden.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dateneingabe")
    st.caption("Bitte fuer jeden Parameter separate Zulauf- und Ablaufwerte als ganze Zahlen eingeben. Leere Zeilen werden ignoriert. Parameter ohne Daten werden spaeter automatisch uebersprungen.")

    controls_col1, controls_col2 = st.columns(2)
    with controls_col1:
        st.button("Beispieldaten laden", use_container_width=True, on_click=load_example_data)
    with controls_col2:
        st.button("Beispieldaten entfernen", use_container_width=True, on_click=remove_example_data)

    initialize_input_state()
    initialize_target_state()

    tab_labels = [parameter["name"] for parameter in PARAMETERS]
    tabs = st.tabs(tab_labels)

    for tab, parameter in zip(tabs, PARAMETERS):
        with tab:
            st.markdown(f"**{parameter['display_name']}**", unsafe_allow_html=False)
            edited_parameter_df = st.data_editor(
                st.session_state[f"input_data_{parameter['id']}"],
                num_rows="fixed",
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Zulaufwerte": st.column_config.NumberColumn("Zulaufwerte", step=1, min_value=0, required=False, format="%d"),
                    "Ablaufwerte": st.column_config.NumberColumn("Ablaufwerte", step=1, min_value=0, required=False, format="%d"),
                },
                key=f"editor_v{st.session_state.get('editor_version', 2)}_{parameter['id']}",
            )
            st.session_state[f"input_data_{parameter['id']}"] = edited_parameter_df

    sync_input_tables_from_parameter_state()
    input_df = st.session_state["input_df"]
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Methoden und Einstellungen")
    default_methods = ["Empirischer Bootstrap"]
    if PYMC_AVAILABLE:
        default_methods.append("Bayessch (PyMC)")
    else:
        default_methods.append("Bayessche Approximation")
    selected_labels = st.multiselect(
        "Methoden",
        options=list(METHOD_OPTIONS.keys()),
        default=default_methods,
        help="Waehlen Sie eine oder mehrere Methoden fuer alle Parameter aus.",
    )

    shared_col1, shared_col2 = st.columns(2)
    with shared_col1:
        q = st.slider("Perzentil (q)", min_value=1, max_value=50, value=10, step=1)
        alpha = st.slider("Alpha", min_value=0.05, max_value=0.50, value=0.05, step=0.05)
    with shared_col2:
        n_sim = st.slider("Praediktive Ziehungen", min_value=1000, max_value=20000, value=5000, step=1000)
        seed_value = st.number_input("Zufalls-Seed", min_value=0, value=42, step=1)

    boot_expander, bayes_expander, pymc_expander = st.columns(3)
    with boot_expander:
        with st.expander("Bootstrap", expanded=True):
            B = st.slider("Bootstrap-Stichproben (B)", min_value=250, max_value=5000, value=1000, step=250)
            add_one = st.toggle("1 zu Ablaufwerten addieren", value=True, key="bootstrap_add_one")

    with bayes_expander:
        with st.expander("Approximation", expanded=True):
            posterior_draws = st.slider("Posterior-Ziehungen pro Kette", min_value=200, max_value=2000, value=600, step=100)
            warmup = st.slider("Warmup-Schritte pro Kette", min_value=100, max_value=1500, value=300, step=100)
            chains = st.slider("Ketten", min_value=1, max_value=4, value=2, step=1)
            add_one_bayes = st.toggle("1 in Approximation addieren", value=True, key="approx_add_one")

    with pymc_expander:
        with st.expander("PyMC", expanded=True):
            if PYMC_AVAILABLE:
                pymc_draws = st.slider("PyMC-Ziehungen pro Kette", min_value=200, max_value=3000, value=600, step=100)
                pymc_warmup = st.slider("PyMC-Warmup pro Kette", min_value=100, max_value=3000, value=300, step=100)
                pymc_chains = st.slider("PyMC-Ketten", min_value=1, max_value=4, value=2, step=1)
                add_one_pymc = st.toggle("1 in PyMC addieren", value=True, key="pymc_add_one")
            else:
                pymc_draws = 600
                pymc_warmup = 300
                pymc_chains = 2
                add_one_pymc = True
                st.warning("PyMC ist unter Python 3.14 in dieser Umgebung derzeit nicht verfuegbar. Fuer diese Methode bitte Python 3.12 oder 3.13 verwenden.")

    run_analysis = st.button("Validierung berechnen", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Zielwerte")
    edited_target_df = st.data_editor(
        st.session_state["target_table"],
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Parameter"],
        column_config={
            "Parameter": st.column_config.TextColumn("Parameter"),
            "Validierungszielwert": st.column_config.NumberColumn(
                "Validierungszielwert",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                required=True,
            ),
        },
        key="target_table_editor",
    )
    st.session_state["target_table"] = edited_target_df
    sync_target_state_from_table()
    st.caption("Die Zielwerte koennen direkt in der Tabelle angepasst werden. Es wird mit einer Nachkommastelle gearbeitet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Datensatz-Uebersicht")
    snapshot_rows: list[dict[str, object]] = []
    validation_errors: list[str] = []
    parameter_inputs: dict[str, dict[str, object]] = {}

    for parameter in PARAMETERS:
        try:
            vals_zu = clean_integer_series(input_df[parameter["zulauf_col"]], f"{parameter['name']} Zulauf")
            vals_ab = clean_integer_series(input_df[parameter["ablauf_col"]], f"{parameter['name']} Ablauf")
            parameter_inputs[parameter["id"]] = {
                "name": parameter["name"],
                "display_name": parameter["display_name"],
                "target": float(st.session_state["target_values"][parameter["id"]]),
                "vals_zu": vals_zu,
                "vals_ab": vals_ab,
            }
            snapshot_rows.append(
                {
                    "Parameter": parameter["name"],
                    "Zulauf": len(vals_zu),
                    "Ablauf": len(vals_ab),
                }
            )
        except ValueError as exc:
            validation_errors.append(str(exc))

    if snapshot_rows:
        st.dataframe(pd.DataFrame(snapshot_rows), hide_index=True, use_container_width=True)
    for error in validation_errors:
        st.error(error)
    st.markdown("</div>", unsafe_allow_html=True)


if run_analysis:
    selected_methods = [METHOD_OPTIONS[label] for label in selected_labels]

    if not selected_methods:
        st.error("Bitte waehlen Sie mindestens eine Auswertungsmethode aus.")
    elif validation_errors:
        st.error("Bitte korrigieren Sie zuerst die Eingabefehler.")
    else:
        available_parameters = [
            parameter["id"]
            for parameter in PARAMETERS
            if parameter_inputs[parameter["id"]]["vals_zu"] and parameter_inputs[parameter["id"]]["vals_ab"]
        ]
        skipped_parameters = [
            parameter_inputs[parameter["id"]]["name"]
            for parameter in PARAMETERS
            if parameter["id"] not in available_parameters
        ]
        nonpositive_zulauf = [
            details["name"]
            for details in parameter_inputs.values()
            if details["vals_zu"] and details["vals_ab"] and any(value <= 0 for value in details["vals_zu"])
        ]
        zero_ablauf_without_plus_one = [
            details["name"]
            for details in parameter_inputs.values()
            if (
                details["vals_zu"]
                and details["vals_ab"]
                and (
                    ("bootstrap" in selected_methods and not add_one and any(value == 0 for value in details["vals_ab"]))
                    or ("bayesian" in selected_methods and not add_one_bayes and any(value == 0 for value in details["vals_ab"]))
                    or ("pymc" in selected_methods and not add_one_pymc and any(value == 0 for value in details["vals_ab"]))
                )
            )
        ]

        if not available_parameters:
            st.error("Bitte geben Sie fuer mindestens einen Parameter sowohl Zulaufwerte als auch Ablaufwerte ein.")
        elif nonpositive_zulauf:
            st.error("Alle Zulaufwerte muessen groesser als 0 sein, da die Berechnung einen Logarithmus verwendet.")
        elif zero_ablauf_without_plus_one:
            st.error("Bei deaktivierter +1-Anpassung duerfen Ablaufwerte nicht 0 sein.")
        else:
            if skipped_parameters:
                st.info("Ohne vollstaendige Daten uebersprungen: " + ", ".join(skipped_parameters))
            with st.spinner("Validierungskennzahlen werden berechnet..."):
                summary_rows: list[dict[str, object]] = []
                distribution_rows: list[dict[str, object]] = []
                pymc_results: dict[str, dict[str, float | np.ndarray]] = {}
                pymc_errors: dict[str, str] = {}

                if "pymc" in selected_methods:
                    batch_payload = {
                        parameter_id: {
                            "vals_zu": parameter_inputs[parameter_id]["vals_zu"],
                            "vals_ab": parameter_inputs[parameter_id]["vals_ab"],
                            "draws": pymc_draws,
                            "warmup": pymc_warmup,
                            "chains": pymc_chains,
                            "n_sim": n_sim,
                            "q": q,
                            "alpha": alpha,
                            "add_one": add_one_pymc,
                            "seed": int(seed_value) + index,
                        }
                        for index, parameter_id in enumerate(available_parameters)
                    }
                    try:
                        batch_result = pymc_q10_lrv_batch(batch_payload)
                        pymc_results = batch_result["results"]
                        pymc_errors = batch_result["errors"]
                    except Exception as exc:
                        pymc_errors = {parameter_id: str(exc) for parameter_id in available_parameters}

                task_specs: list[dict[str, object]] = []
                task_counter = 0
                for parameter in PARAMETERS:
                    details = parameter_inputs[parameter["id"]]
                    if parameter["id"] not in available_parameters:
                        continue
                    for label, method_key in METHOD_OPTIONS.items():
                        if method_key not in selected_methods or method_key == "pymc":
                            continue
                        task_specs.append(
                            {
                                "parameter_id": parameter["id"],
                                "label": label,
                                "method_key": method_key,
                                "vals_zu": details["vals_zu"],
                                "vals_ab": details["vals_ab"],
                                "q": q,
                                "alpha": alpha,
                                "n_sim": n_sim,
                                "seed_value": int(seed_value) + task_counter,
                                "B": B,
                                "add_one": add_one,
                                "posterior_draws": posterior_draws,
                                "warmup": warmup,
                                "chains": chains,
                                "pymc_draws": pymc_draws,
                                "pymc_warmup": pymc_warmup,
                                "pymc_chains": pymc_chains,
                                "add_one_bayes": add_one_bayes,
                                "add_one_pymc": add_one_pymc,
                            }
                        )
                        task_counter += 1

                task_results: dict[tuple[str, str], dict[str, float | np.ndarray]] = {}
                task_errors: list[str] = []
                if task_specs:
                    with ThreadPoolExecutor(max_workers=min(4, len(task_specs))) as executor:
                        future_to_task = {executor.submit(execute_analysis_task, task): task for task in task_specs}
                        for future in as_completed(future_to_task):
                            task = future_to_task[future]
                            try:
                                parameter_id, label, result = future.result()
                                task_results[(parameter_id, label)] = result
                            except Exception as exc:
                                task_errors.append(f"{task['label']} fuer {parameter_inputs[task['parameter_id']]['name']} konnte nicht berechnet werden: {exc}")

                for error_message in task_errors:
                    st.error(error_message)

                for parameter in PARAMETERS:
                    details = parameter_inputs[parameter["id"]]
                    if parameter["id"] not in available_parameters:
                        continue
                    for label, method_key in METHOD_OPTIONS.items():
                        if method_key not in selected_methods:
                            continue

                        if method_key == "pymc":
                            if parameter["id"] in pymc_errors:
                                st.error(f"{label} fuer {details['name']} konnte nicht berechnet werden: {pymc_errors[parameter['id']]}")
                                continue
                            result = pymc_results.get(parameter["id"])
                            if result is None:
                                continue
                        else:
                            result = task_results.get((parameter["id"], label))
                            if result is None:
                                continue

                        summary_rows.append(
                            {
                                "Parameter": details["name"],
                                "Methode": label,
                                "Zielwert": details["target"],
                                "Lower Bound": float(result["L_alpha"]),
                                "Median": float(result["median"]),
                                "Obergrenze": float(result["upper_(1-alpha)"]),
                                "Mittelwert": float(result["mean"]),
                                "Standardabweichung": float(result["std_dev"]),
                                "Lower Bound >= Zielwert": "Ja" if float(result["L_alpha"]) >= details["target"] else "Nein",
                                "Median >= Zielwert": "Ja" if float(result["median"]) >= details["target"] else "Nein",
                                "Ziehungen": len(result["q10_samples"]),
                            }
                        )

                        distribution_rows.extend(
                            {
                                "Parameter": details["name"],
                                "Method": label,
                                "q10_sample": sample,
                            }
                            for sample in result["q10_samples"]
                        )

            st.session_state["analysis_results"] = {
                "summary_df": pd.DataFrame(summary_rows),
                "distribution_df": pd.DataFrame(distribution_rows),
                "available_parameters": available_parameters,
                "selected_methods": selected_methods,
                "selected_labels": selected_labels,
                "q": q,
                "skipped_parameters": skipped_parameters,
            }
            st.session_state.pop("report_pdf", None)

if "analysis_results" in st.session_state:
    summary_df = st.session_state["analysis_results"]["summary_df"]
    distribution_df = st.session_state["analysis_results"]["distribution_df"]
    available_parameters = st.session_state["analysis_results"]["available_parameters"]
    skipped_parameters = st.session_state["analysis_results"]["skipped_parameters"]
    q = st.session_state["analysis_results"]["q"]

    if skipped_parameters:
        st.info("Ohne vollstaendige Daten uebersprungen: " + ", ".join(skipped_parameters))

    if not summary_df.empty:
        pass_count = int(
            summary_df[
                (summary_df["Lower Bound >= Zielwert"] == "Ja") & (summary_df["Median >= Zielwert"] == "Ja")
            ].shape[0]
        )
        metric_cols = st.columns(3)
        metric_values = [
            ("Berechnete Parameter/Methoden", len(summary_df)),
            ("Beide Kriterien erfuellt", pass_count),
            ("Parameter mit Daten", len(available_parameters)),
        ]
        for column, (label, value) in zip(metric_cols, metric_values):
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        top_col, bottom_col = st.columns([1.05, 0.95], gap="large")

        with top_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Validierungstabelle")
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with bottom_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Histogramm")
            histogram_parameters = summary_df["Parameter"].drop_duplicates().tolist()
            default_histogram_parameter = st.session_state.get("histogram_parameter")
            if default_histogram_parameter not in histogram_parameters:
                st.session_state["histogram_parameter"] = histogram_parameters[0]
            selected_parameter = st.selectbox(
                "Parameter fuer Histogramm",
                options=histogram_parameters,
                key="histogram_parameter",
            )
            chart_df = distribution_df[distribution_df["Parameter"] == selected_parameter]
            chart_summary_df = summary_df[summary_df["Parameter"] == selected_parameter][
                ["Methode", "Lower Bound", "Zielwert"]
            ].rename(columns={"Methode": "Method"})
            st.pyplot(build_histogram_chart(chart_df, chart_summary_df, q), use_container_width=True)
            st.caption("Die violette Linie markiert den Validierungszielwert. Gestrichelte Linien markieren den Lower Bound der Methoden.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Fachliche Einordnung")
        st.write(
            "Die Validierung gilt fuer einen Parameter als erfuellt, wenn sowohl der Lower Bound als auch der Median "
            "den jeweiligen Zielwert erreichen oder ueberschreiten. In der Tabelle oben ist dies fuer jede Methode "
            "und jeden Parameter direkt sichtbar."
        )
        report_col1, report_col2 = st.columns([0.55, 0.45])
        with report_col1:
            if st.button("PDF-Report vorbereiten", use_container_width=True):
                with st.spinner("PDF-Report wird erstellt..."):
                    st.session_state["report_pdf"] = build_validation_report_pdf(
                        summary_df=summary_df,
                        distribution_df=distribution_df,
                        q=q,
                        selected_methods=st.session_state["analysis_results"]["selected_labels"],
                    )
        with report_col2:
            if "report_pdf" in st.session_state:
                st.download_button(
                    "Automatisiertes Reporting (PDF)",
                    data=st.session_state["report_pdf"],
                    file_name="validreuse_validierungsreport.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
elif not run_analysis:
    st.info("Geben Sie fuer einen oder mehrere Parameter Zu- und Ablaufwerte ein und starten Sie danach die Validierungsberechnung.")
