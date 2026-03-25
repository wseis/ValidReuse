from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from bayesian import bayesian_q10_lrv
from bootstrap import bootstrap_q10_lrv
from pymc_bayesian import pymc_q10_lrv


METHOD_OPTIONS = {
    "Empirical bootstrap": "bootstrap",
    "Bayesian approximation": "bayesian",
    "Bayesian (PyMC)": "pymc",
}


def clean_integer_series(series: pd.Series, column_name: str) -> list[int]:
    values: list[int] = []

    for index, raw_value in series.items():
        if pd.isna(raw_value) or raw_value == "":
            continue

        numeric_value = float(raw_value)
        if not numeric_value.is_integer():
            raise ValueError(f"{column_name} row {index + 1} must be an integer.")

        int_value = int(numeric_value)
        if int_value < 0:
            raise ValueError(f"{column_name} row {index + 1} must be 0 or greater.")

        values.append(int_value)

    return values


def build_default_table() -> pd.DataFrame:
    return pd.DataFrame({"vals_zu": [None] * 20, "vals_ab": [None] * 20})


def summarise_result(method_name: str, result: dict[str, float | list[float]]) -> dict[str, float | int | str]:
    q10_samples = result["q10_samples"]
    return {
        "Method": method_name,
        "Draws": len(q10_samples),
        "Lower bound": result["L_alpha"],
        "Median": result["median"],
        "Upper bound": result["upper_(1-alpha)"],
        "Mean": result["mean"],
        "Std. deviation": result["std_dev"],
    }


def build_histogram_chart(chart_df: pd.DataFrame, summary_df: pd.DataFrame, q: int) -> alt.Chart:
    color_domain = ["Empirical bootstrap", "Bayesian approximation", "Bayesian (PyMC)"]
    color_range = ["#2563eb", "#ea580c", "#059669"]
    histogram = (
        alt.Chart(chart_df)
        .mark_bar(opacity=0.45, binSpacing=0)
        .transform_bin(as_=["bin_start", "bin_end"], field="q10_sample", bin=alt.Bin(maxbins=36))
        .transform_aggregate(count="count()", groupby=["Method", "bin_start", "bin_end"])
        .transform_joinaggregate(total="sum(count)", groupby=["Method"])
        .transform_calculate(density="datum.count / datum.total")
        .encode(
            x=alt.X("bin_start:Q", bin="binned", title=f"q{q} bootstrap / posterior samples"),
            x2="bin_end:Q",
            y=alt.Y("density:Q", stack=None, title="Density"),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("Method:N"),
                alt.Tooltip("bin_start:Q", format=".4f", title="Bin start"),
                alt.Tooltip("density:Q", format=".6f", title="Density"),
            ],
        )
        .properties(height=400)
    )

    reference_lines = (
        alt.Chart(summary_df)
        .mark_rule(strokeDash=[6, 4], strokeWidth=2.5)
        .encode(
            x=alt.X("Lower bound:Q"),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(domain=color_domain, range=color_range),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Method:N"),
                alt.Tooltip("Lower bound:Q", format=".6f"),
            ],
        )
    )

    return histogram + reference_lines


st.set_page_config(page_title="ValidReuse Method Explorer", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 190, 92, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(67, 97, 238, 0.18), transparent 25%),
            linear-gradient(180deg, #f7f3eb 0%, #f3efe7 100%);
        color: #1f2937;
    }
    .block-container {
        padding-top: 2.25rem;
        padding-bottom: 2rem;
    }
    .hero {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 28px;
        padding: 1.75rem 1.75rem 1.25rem 1.75rem;
        box-shadow: 0 20px 50px rgba(148, 163, 184, 0.18);
        backdrop-filter: blur(12px);
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        letter-spacing: -0.03em;
        color: #132238;
    }
    .hero p {
        margin-top: 0.75rem;
        margin-bottom: 0;
        font-size: 1rem;
        color: #475569;
        max-width: 64rem;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>ValidReuse Method Explorer</h1>
        <p>
            Enter integer observations for <strong>vals_zu</strong> and <strong>vals_ab</strong>, then compare
            the current empirical bootstrap against two Bayesian alternatives built on the same input data.
            Results are shown side by side as a summary table and an overlaid histogram.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Input Table")
    st.caption("Use whole numbers only. Blank rows are ignored, and the table starts with 20 rows.")

    input_df = st.data_editor(
        build_default_table(),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "vals_zu": st.column_config.NumberColumn("vals_zu", step=1, min_value=0, required=False, format="%d"),
            "vals_ab": st.column_config.NumberColumn("vals_ab", step=1, min_value=0, required=False, format="%d"),
        },
        key="input_table",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Methods and Settings")
    selected_labels = st.multiselect(
        "Methods",
        options=list(METHOD_OPTIONS.keys()),
        default=list(METHOD_OPTIONS.keys()),
        help="Choose one or both methods to compare them on the same data.",
    )

    shared_col1, shared_col2 = st.columns(2)
    with shared_col1:
        q = st.slider("Percentile (q)", min_value=1, max_value=50, value=10, step=1)
        alpha = st.slider("Alpha", min_value=0.05, max_value=0.50, value=0.05, step=0.05)
    with shared_col2:
        n_sim = st.slider("Predictive draws", min_value=1000, max_value=20000, value=5000, step=1000)
        seed_value = st.number_input("Random seed", min_value=0, value=42, step=1)

    boot_expander, bayes_expander, pymc_expander = st.columns(3)
    with boot_expander:
        with st.expander("Empirical bootstrap settings", expanded=True):
            B = st.slider("Bootstrap samples (B)", min_value=250, max_value=5000, value=1000, step=250)
            add_one = st.toggle("Add 1 to vals_ab in bootstrap", value=True)

    with bayes_expander:
        with st.expander("Approximation settings", expanded=True):
            posterior_draws = st.slider("Posterior draws per chain", min_value=200, max_value=2000, value=600, step=100)
            warmup = st.slider("Warmup steps per chain", min_value=100, max_value=1500, value=300, step=100)
            chains = st.slider("Chains", min_value=1, max_value=4, value=2, step=1)
            add_one_bayes = st.toggle("Add 1 in approximation", value=True)
            st.caption("This is the lightweight Python Bayesian approximation currently used in the app.")

    with pymc_expander:
        with st.expander("PyMC settings", expanded=True):
            pymc_draws = st.slider("PyMC draws per chain", min_value=200, max_value=3000, value=600, step=100)
            pymc_warmup = st.slider("PyMC warmup per chain", min_value=100, max_value=3000, value=300, step=100)
            pymc_chains = st.slider("PyMC chains", min_value=1, max_value=4, value=2, step=1)
            add_one_pymc = st.toggle("Add 1 in PyMC simulation", value=True)
            st.caption("PyMC uses sampled posteriors for the same lognormal plus negative binomial model family.")

    run_analysis = st.button("Run Comparison", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset Snapshot")
    st.caption("Counts update live as you edit the table.")

    try:
        vals_zu = clean_integer_series(input_df["vals_zu"], "vals_zu")
        vals_ab = clean_integer_series(input_df["vals_ab"], "vals_ab")
        snapshot_df = pd.DataFrame(
            {
                "Metric": ["vals_zu count", "vals_ab count", "Selected methods"],
                "Value": [len(vals_zu), len(vals_ab), max(len(selected_labels), 0)],
            }
        )
        st.table(snapshot_df)

        if len(vals_zu) != len(vals_ab):
            st.warning("The two columns may have different counts. Both methods will use each column as entered.")
    except ValueError as exc:
        st.error(str(exc))
        vals_zu = []
        vals_ab = []

    st.markdown("</div>", unsafe_allow_html=True)


if run_analysis:
    selected_methods = [METHOD_OPTIONS[label] for label in selected_labels]

    if not selected_methods:
        st.error("Please select at least one analysis method.")
    elif not vals_zu or not vals_ab:
        st.error("Please enter at least one valid integer in both vals_zu and vals_ab.")
    elif any(value <= 0 for value in vals_zu):
        st.error("All selected methods require `vals_zu` to be greater than 0 because the analysis uses a logarithm.")
    elif "bootstrap" in selected_methods and not add_one and any(value == 0 for value in vals_ab):
        st.error("`vals_ab` cannot contain 0 for empirical bootstrap when `Add 1 to vals_ab in bootstrap` is turned off.")
    elif "bayesian" in selected_methods and not add_one_bayes and any(value == 0 for value in vals_ab):
        st.error("`vals_ab` cannot contain 0 for the Bayesian approximation when `Add 1 in approximation` is turned off.")
    elif "pymc" in selected_methods and not add_one_pymc and any(value == 0 for value in vals_ab):
        st.error("`vals_ab` cannot contain 0 for PyMC when `Add 1 in PyMC simulation` is turned off.")
    else:
        with st.spinner("Running selected analyses..."):
            method_results: list[tuple[str, dict[str, float | list[float]]]] = []

            if "bootstrap" in selected_methods:
                bootstrap_result = bootstrap_q10_lrv(
                    vals_zu=vals_zu,
                    vals_ab=vals_ab,
                    B=B,
                    n_sim=n_sim,
                    q=q,
                    alpha=alpha,
                    add_one=add_one,
                    seed=int(seed_value),
                )
                method_results.append(("Empirical bootstrap", bootstrap_result))

            if "bayesian" in selected_methods:
                bayesian_result = bayesian_q10_lrv(
                    vals_zu=vals_zu,
                    vals_ab=vals_ab,
                    draws=posterior_draws,
                    warmup=warmup,
                    chains=chains,
                    n_sim=n_sim,
                    q=q,
                    alpha=alpha,
                    add_one=add_one_bayes,
                    seed=int(seed_value),
                )
                method_results.append(("Bayesian approximation", bayesian_result))

            if "pymc" in selected_methods:
                pymc_result = pymc_q10_lrv(
                    vals_zu=vals_zu,
                    vals_ab=vals_ab,
                    draws=pymc_draws,
                    warmup=pymc_warmup,
                    chains=pymc_chains,
                    n_sim=n_sim,
                    q=q,
                    alpha=alpha,
                    add_one=add_one_pymc,
                    seed=int(seed_value),
                )
                method_results.append(("Bayesian (PyMC)", pymc_result))

        summary_df = pd.DataFrame([summarise_result(name, result) for name, result in method_results])
        chart_df = pd.concat(
            [
                pd.DataFrame({"Method": name, "q10_sample": result["q10_samples"]})
                for name, result in method_results
            ],
            ignore_index=True,
        )

        metric_cols = st.columns(max(1, len(method_results)))
        for column, row in zip(metric_cols, summary_df.to_dict(orient="records")):
            with column:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{row["Method"]}</div>
                        <div class="metric-value">{row["Lower bound"]:.4f}</div>
                        <div style="color:#64748b;font-size:0.92rem;">L_alpha lower bound</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        results_col, chart_col = st.columns([0.95, 1.05], gap="large")

        with results_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Method Comparison")
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Lower bound": st.column_config.NumberColumn(format="%.6f"),
                    "Median": st.column_config.NumberColumn(format="%.6f"),
                    "Upper bound": st.column_config.NumberColumn(format="%.6f"),
                    "Mean": st.column_config.NumberColumn(format="%.6f"),
                    "Std. deviation": st.column_config.NumberColumn(format="%.6f"),
                },
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with chart_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Distribution Comparison")
            st.altair_chart(build_histogram_chart(chart_df, summary_df, q), use_container_width=True)
            st.caption("Dashed reference lines show each method's L_alpha lower bound.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Interpretation")
        if len(method_results) == 1:
            only_row = summary_df.iloc[0]
            st.write(
                f"**{only_row['Method']}** centers around **{only_row['Median']:.4f}** with a lower bound of "
                f"**{only_row['Lower bound']:.4f}** at alpha = **{alpha:.2f}**."
            )
        else:
            best_row = summary_df.sort_values("Lower bound", ascending=False).iloc[0]
            st.write(
                f"On this dataset, **{best_row['Method']}** yields the higher lower bound "
                f"(**{best_row['Lower bound']:.4f}**) at alpha = **{alpha:.2f}**. Use the table and chart above "
                "to compare how concentrated or conservative each method is."
            )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Enter your data, choose one or both methods, and click `Run Comparison`.")
