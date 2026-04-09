"""
Streamlit dashboard — Active Learning Competitor Model Simulation
=================================================================
Visualises the AL simulation results from outputs/al_results/results.parquet.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT         = Path(__file__).parent
RESULTS_PATH = ROOT / "outputs" / "al_results" / "results.parquet"

# ── Constants ──────────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "random":          "Random",
    "uncertainty":     "Uncertainty",
    "error_based":     "Error-based",
    "shap_divergence": "SHAP divergence",
}

PALETTE = {
    "random":          "#888888",
    "uncertainty":     "#1f77b4",
    "error_based":     "#ff7f0e",
    "shap_divergence": "#2ca02c",
}

METRIC_OPTIONS = {
    "RMSE (€)":                    "rmse",
    "Relative RMSE":               "rel_rmse",
    "SHAP cosine similarity":      "shap_cosine_similarity",
}

# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_parquet(RESULTS_PATH)
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def convergence_figure(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    strategies: list[str],
    tariff_change_week: int | None = None,
) -> go.Figure:
    """Line chart: metric over weeks, one trace per strategy."""
    fig = go.Figure()

    for strat in strategies:
        grp = df[df["strategy"] == strat].sort_values("week")
        fig.add_trace(go.Scatter(
            x=grp["week"],
            y=grp[metric],
            mode="lines+markers",
            name=STRATEGY_LABELS.get(strat, strat),
            line=dict(color=PALETTE.get(strat, "#333"), width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{STRATEGY_LABELS.get(strat, strat)}</b><br>"
                "Week: %{x}<br>"
                f"{metric_label}: %{{y:.4f}}<br>"
                "Labeled: %{customdata:,}<extra></extra>"
            ),
            customdata=grp["n_labeled"].values,
        ))

    if tariff_change_week is not None:
        fig.add_vline(
            x=tariff_change_week,
            line_dash="dash",
            line_color="red",
            line_width=1.5,
            annotation_text="tariff change",
            annotation_position="top right",
            annotation_font_color="red",
        )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title=metric_label,
        legend_title="Strategy",
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def summary_table(df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Final-week metrics for each strategy, formatted for display."""
    max_week = df["week"].max()
    final = df[df["week"] == max_week].copy()
    final = final[final["strategy"].isin(strategies)]
    final = final.set_index("strategy_label")[
        ["n_labeled", "rmse", "rel_rmse", "shap_cosine_similarity"]
    ]
    final.index.name = "Strategy"
    final.columns = ["Labeled profiles", "RMSE (€)", "Relative RMSE", "SHAP similarity"]
    return final


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AL Competitor Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("AL simulation")
    st.caption("Competitor model active learning — strategy explorer")

    if not RESULTS_PATH.exists():
        st.error(
            f"Results file not found:\n`{RESULTS_PATH}`\n\n"
            "Run `notebooks/05_al_simulation.py` first."
        )
        st.stop()

    df_all = load_results()

    st.subheader("Strategies")
    all_strategies = list(STRATEGY_LABELS.keys())
    selected_strategies = [
        s for s in all_strategies
        if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}")
    ]

    st.subheader("Primary metric")
    metric_label = st.radio("", list(METRIC_OPTIONS.keys()), index=0)
    metric_col = METRIC_OPTIONS[metric_label]

    st.subheader("About")
    n_weeks = df_all["week"].max()
    n_rows  = df_all[df_all["scenario"] == "no_tariff_change"]["n_labeled"].max()
    st.markdown(
        f"**Simulation:** {n_weeks} weeks  \n"
        f"**Max labeled:** {n_rows:,} profiles  \n"
        f"**Scenarios:** {', '.join(df_all['scenario'].unique())}"
    )

# ── Guard: need at least one strategy ─────────────────────────────────────────

if not selected_strategies:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Strategy comparison", "Tariff change recovery"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Strategy comparison (no tariff change)
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    df_s1 = df_all[df_all["scenario"] == "no_tariff_change"]
    df_s1 = df_s1[df_s1["strategy"].isin(selected_strategies)]

    st.header("Strategy comparison — no tariff change")
    st.caption(
        "All strategies start from the same warm-start labeled set. "
        "Lower RMSE and higher SHAP similarity = faster convergence to the oracle tariff."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE on holdout (€)")
        fig_rmse = convergence_figure(df_s1, "rmse", "RMSE (€)", selected_strategies)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        st.subheader("SHAP cosine similarity")
        fig_shap = convergence_figure(
            df_s1, "shap_cosine_similarity", "Cosine similarity", selected_strategies
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    # Relative RMSE as an alternative metric (shown only if selected)
    if metric_col == "rel_rmse":
        st.subheader("Relative RMSE")
        fig_rel = convergence_figure(df_s1, "rel_rmse", "Relative RMSE", selected_strategies)
        st.plotly_chart(fig_rel, use_container_width=True)

    st.subheader(f"Final-week summary (week {n_weeks})")
    tbl = summary_table(df_s1, selected_strategies)
    st.dataframe(
        tbl.style.format({
            "Labeled profiles": "{:,.0f}",
            "RMSE (€)":         "{:.2f}",
            "Relative RMSE":    "{:.4f}",
            "SHAP similarity":  "{:.4f}",
        }),
        use_container_width=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Tariff change recovery
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    df_s2 = df_all[df_all["scenario"] == "tariff_change"]
    df_s2 = df_s2[df_s2["strategy"].isin(selected_strategies)]

    # Find the tariff change week from the data
    change_weeks = df_s2[df_s2["post_tariff_change"]]["week"]
    tariff_week  = int(change_weeks.min()) if len(change_weeks) else None

    st.header("Tariff change recovery")
    if tariff_week is not None:
        st.caption(
            f"Young-driver surcharge (+20%) injected at **week {tariff_week}**. "
            "RMSE is measured against the *new* oracle after that point. "
            "A good strategy recovers quickly without a full restart."
        )
    else:
        st.caption("No tariff change detected in the results.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RMSE recovery")
        fig_tc_rmse = convergence_figure(
            df_s2, "rmse", "RMSE (€)", selected_strategies,
            tariff_change_week=tariff_week,
        )
        st.plotly_chart(fig_tc_rmse, use_container_width=True)

    with col2:
        st.subheader("SHAP similarity recovery")
        fig_tc_shap = convergence_figure(
            df_s2, "shap_cosine_similarity", "Cosine similarity", selected_strategies,
            tariff_change_week=tariff_week,
        )
        st.plotly_chart(fig_tc_shap, use_container_width=True)

    st.subheader(f"Final-week summary (week {n_weeks})")
    tbl2 = summary_table(df_s2, selected_strategies)
    st.dataframe(
        tbl2.style.format({
            "Labeled profiles": "{:,.0f}",
            "RMSE (€)":         "{:.2f}",
            "Relative RMSE":    "{:.4f}",
            "SHAP similarity":  "{:.4f}",
        }),
        use_container_width=True,
    )

    # Side-by-side: pre vs post tariff change
    st.subheader("Pre- vs post-tariff change comparison")
    pre_week  = (tariff_week - 1) if tariff_week and tariff_week > 0 else 0
    post_week = n_weeks
    compare   = (
        df_s2[
            df_s2["week"].isin([pre_week, post_week])
            & df_s2["strategy"].isin(selected_strategies)
        ]
        .copy()
    )
    compare["period"] = compare["week"].apply(
        lambda w: f"Pre-change (wk {pre_week})" if w == pre_week else f"Post-change (wk {post_week})"
    )
    pivot = (
        compare
        .pivot_table(index="strategy_label", columns="period", values="rmse")
        .rename_axis("Strategy")
    )
    if not pivot.empty:
        st.dataframe(
            pivot.style.format("{:.2f}"),
            use_container_width=True,
        )
