"""
Streamlit dashboard — Active Learning Competitor Model Simulation
=================================================================
Visualises the AL simulation results from outputs/al_results/results.parquet.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT         = Path(__file__).parent
RESULTS_DIR  = ROOT / "outputs" / "al_results"

sys.path.insert(0, str(ROOT / "src"))

from market_model_al.segments import SEGMENTS  # noqa: E402 — needs sys.path first

# ── Constants ──────────────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    # CP strategies (one feature swept at a time)
    "random_cp":                    "Random (CP)",
    "random_market":                "Random market",
    "uncertainty_cp":               "Uncertainty (CP)",
    "error_based_cp":               "Error-based (CP)",
    "segment_adaptive_cp":          "Segment-adaptive (CP)",
    "disruption_cp":                "Disruption-adaptive (CP)",
    "random_cp_restart":                "Random (CP, restart)",
    "segment_adaptive_cp_restart":      "Segment-adaptive (CP, restart)",
    "random_gauss_restart":             "Random (Gaussian, restart)",
    "segment_adaptive_gauss_restart":   "Segment-adaptive (Gaussian, restart)",
    # Gaussian-perturbation variants
    "random_gauss":                 "Random (Gaussian)",
    "uncertainty_gauss":            "Uncertainty (Gaussian)",
    "error_based_gauss":            "Error-based (Gaussian)",
    "segment_adaptive_gauss":       "Segment-adaptive (Gaussian)",
    "disruption_gauss":             "Disruption-adaptive (Gaussian)",
    # Hybrid: error-based scoring on a representative market pool
    "informed_market":              "Informed market",
}

PALETTE = {
    # CP strategies
    "random_cp":                    "#888888",
    "random_market":                "#17becf",
    "uncertainty_cp":               "#1f77b4",
    "error_based_cp":               "#ff7f0e",
    "segment_adaptive_cp":          "#9467bd",
    "disruption_cp":                "#d62728",
    "random_cp_restart":                "#888888",
    "segment_adaptive_cp_restart":      "#9467bd",
    "random_gauss_restart":             "#cccccc",
    "segment_adaptive_gauss_restart":   "#c5b0d5",
    # Gaussian variants — lighter tints of their CP counterparts
    "random_gauss":                 "#cccccc",
    "uncertainty_gauss":            "#aec7e8",
    "error_based_gauss":            "#ffbb78",
    "segment_adaptive_gauss":       "#c5b0d5",
    "disruption_gauss":             "#f7b6b6",
    # Hybrid: bold teal-green, distinct from both random_market and error_based
    "informed_market":              "#2ca02c",
}

METRIC_OPTIONS = {
    "RMSE":                   "rmse",
    "Relative RMSE":          "rel_rmse",
    "SHAP cosine similarity": "shap_cosine_similarity",
}

# Y-axis labels for charts — may include units not shown in the radio buttons
METRIC_AXIS_LABELS = {
    "rmse":                   "RMSE (€)",
    "rel_rmse":               "Relative RMSE",
    "shap_cosine_similarity": "Cosine similarity",
}

METRIC_HELP = {
    "rmse": (
        "Root mean squared error on the fixed holdout set (2 000 policy rows, "
        "oracle-labeled once before any training).\n\n"
        "After a tariff change, holdout labels switch to the new oracle — so RMSE "
        "measures recovery of the *currently active* tariff."
    ),
    "rel_rmse": "RMSE divided by the mean holdout premium — normalises for scale.",
    "shap_cosine_similarity": (
        "Mean cosine similarity between oracle and competitor SHAP vectors on the holdout. "
        "Measures tariff *structure* recovery, not just accuracy.  "
        "A score of 1 means perfect structural alignment.\n\n"
        "**Simulation-only:** requires access to the oracle's internal model."
    ),
}

# ── Data loading ───────────────────────────────────────────────────────────────

_LEGACY_STRATEGY_NAMES = {
    # Maps pre-rename strategy names (without _cp suffix) to current names.
    # Allows the dashboard to display results from parquets generated before
    # the rename without requiring an immediate re-run of the simulation.
    "random":                    "random_cp",
    "uncertainty":               "uncertainty_cp",
    "error_based":               "error_based_cp",
    "segment_adaptive":          "segment_adaptive_cp",
    "disruption":                "disruption_cp",
    "random_restart":            "random_cp_restart",
    "segment_adaptive_restart":  "segment_adaptive_cp_restart",
}

@st.cache_data
def load_results() -> pd.DataFrame:
    # Load all per-strategy parquets (new format) plus the legacy monolithic
    # file if it still exists, then concatenate.
    parquets = [p for p in RESULTS_DIR.glob("*.parquet") if p.stem != "results_legacy"]
    if not parquets:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    df["strategy"] = df["strategy"].replace(_LEGACY_STRATEGY_NAMES)
    df["strategy_label"] = df["strategy"].map(STRATEGY_LABELS).fillna(df["strategy"])
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def has_shap(df: pd.DataFrame) -> bool:
    """True when the results contain non-NaN SHAP similarity values."""
    return "shap_cosine_similarity" in df.columns and df["shap_cosine_similarity"].notna().any()


def tariff_change_weeks(df: pd.DataFrame) -> list[int]:
    """Return sorted list of weeks where a tariff change was applied."""
    if "tariff_change_applied" not in df.columns:
        return []
    return sorted(df[df["tariff_change_applied"]]["week"].unique().tolist())


def convergence_figure(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    strategies: list[str],
    change_weeks: list[int] | None = None,
    plotly_theme: str = "plotly_dark",
) -> go.Figure:
    """Line chart: metric over weeks, one trace per strategy."""
    fig = go.Figure()

    for strat in strategies:
        grp = df[df["strategy"] == strat].sort_values("week")
        if grp.empty:
            continue
        is_cp      = "_cp" in strat   # random_cp, uncertainty_cp, …_cp_restart
        is_restart = strat.endswith("_restart")
        fig.add_trace(go.Scatter(
            x=grp["week"],
            y=grp[metric],
            mode="lines+markers",
            name=STRATEGY_LABELS.get(strat, strat),
            line=dict(
                color=PALETTE.get(strat, "#333"),
                width=2,
                dash="dash" if is_restart else "solid",
            ),
            marker=dict(
                size=8 if is_cp else 6,
                symbol="triangle-up" if is_cp else "circle",
            ),
            hovertemplate=(
                f"<b>{STRATEGY_LABELS.get(strat, strat)}</b><br>"
                "Week: %{x}<br>"
                f"{metric_label}: %{{y:.4f}}<br>"
                "Labeled: %{customdata:,}<extra></extra>"
            ),
            customdata=grp["n_labeled"].values,
        ))

    for i, w in enumerate(change_weeks or []):
        label = f"change {i + 1}" if len(change_weeks or []) > 1 else "tariff change"
        fig.add_vline(
            x=w,
            line_dash="dash",
            line_color="red",
            line_width=1.5,
            annotation_text=label,
            annotation_position="top right",
            annotation_font_color="red",
        )

    fig.update_layout(
        template=plotly_theme,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Week",
        yaxis_title=metric_label,
        legend_title="Strategy",
        hovermode="x unified",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def summary_table(df: pd.DataFrame, strategies: list[str], week: int | None = None) -> pd.DataFrame:
    """Metrics for each strategy at a given week (defaults to the last week)."""
    if week is None:
        week = df["week"].max()
    final = df[df["week"] == week].copy()
    final = final[final["strategy"].isin(strategies)]
    cols       = ["n_labeled", "rmse", "rel_rmse"]
    col_labels = ["Labeled profiles", "RMSE (€)", "Relative RMSE"]
    if has_shap(df):
        cols.append("shap_cosine_similarity")
        col_labels.append("SHAP similarity")
    final = final.set_index("strategy_label")[cols]
    final.index.name = "Strategy"
    final.columns = col_labels
    return final


def segment_heatmap(
    df: pd.DataFrame,
    week: int,
    strategies: list[str],
    metric: str = "rmse",
    plotly_theme: str = "plotly_dark",
) -> go.Figure:
    """Heatmap: strategies (rows) × segments (columns), colour = metric at *week*.

    Supports metric='rmse' (uses rmse_{key} columns) and metric='rel_rmse'
    (uses rel_rmse_{key} columns, stored at simulation time using per-segment
    mean premium as the denominator).

    Red = high error, green = low error.  Missing values (NaN) are shown in grey.
    """
    seg_keys   = [seg.key   for seg in SEGMENTS]
    seg_labels = [seg.label for seg in SEGMENTS]

    snap = df[df["week"] == week].copy()
    snap = snap[snap["strategy"].isin(strategies)]

    col_prefix   = "rel_rmse_" if metric == "rel_rmse" else "rmse_"
    val_fmt      = ".4f"        if metric == "rel_rmse" else ".2f"
    colorbar_ttl = "Rel. RMSE"  if metric == "rel_rmse" else "RMSE (€)"
    hover_label  = "Rel. RMSE"  if metric == "rel_rmse" else "RMSE"

    # Build matrix: rows = strategies (in sidebar order), cols = segments
    strat_order = [s for s in strategies if not snap[snap["strategy"] == s].empty]

    z, text = [], []
    for strat in strat_order:
        row_df = snap[snap["strategy"] == strat]
        row_z, row_t = [], []
        for key in seg_keys:
            col = f"{col_prefix}{key}"
            val = row_df[col].values[0] if col in row_df.columns and len(row_df) else float("nan")
            row_z.append(val if pd.notna(val) else None)
            row_t.append(format(val, val_fmt) if pd.notna(val) else "n/a")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=seg_labels,
        y=[STRATEGY_LABELS.get(s, s) for s in strat_order],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorscale="RdYlGn_r",
        colorbar=dict(title=colorbar_ttl),
        hoverongaps=False,
        hovertemplate=f"<b>%{{y}}</b><br>%{{x}}<br>{hover_label}: %{{text}}<extra></extra>",
    ))
    fig.update_layout(
        template=plotly_theme,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Segment",
        yaxis_title="Strategy",
        height=max(200, 60 + 50 * len(strat_order)),
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _strategy_card(info: dict) -> None:
    with st.expander(f"**{info['name']}**  —  {info['summary']}", expanded=False):
        if "detail" in info:
            st.markdown(info["detail"])
            st.divider()
        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("**Strengths**")
            for s in info["strengths"]:
                st.markdown(f"- {s}")
        with col_w:
            st.markdown("**Weaknesses**")
            for w in info["weaknesses"]:
                st.markdown(f"- {w}")
        st.markdown(f"**When to use:** {info['when']}")


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AL Competitor Model",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "An insurer scraping competitor quotes from an aggregator website can't afford to "
            "scrape everything — it has to choose what to request. This app explores that "
            "choice using active learning (AL). "
            "The oracle is a LightGBM model trained on a real Spanish motor vehicle portfolio "
            "([Mendeley Data, doi: 10.17632/5cxyb5fp4f.2](https://data.mendeley.com/datasets/5cxyb5fp4f/2)). "
            "Strategies are evaluated on holdout RMSE, per-segment RMSE, and SHAP cosine similarity.\n\n"
            "The central finding on a Spanish dataset: representativeness dominates informativeness. "
            "Random market sampling — no scoring, no model — outperforms every tested informativeness "
            "strategy globally and per segment. Even the principled best-of-both-worlds hybrid "
            "(a representative pool with an informativeness filter on top) cannot beat pure random "
            "market. Exploration wins.\n\n"
            "Built with Streamlit, LightGBM, SHAP, and Plotly.  \n"
            "David Fischer · April 2026"
        ),
    },
)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Active Learning Strategy Evaluator")
    st.markdown(
        "<p style='font-size:0.95rem; color: grey;'>"
        "An insurer scraping competitor quotes from an aggregator faces a classic active learning dilemma: "
        "query the profiles where the model is most wrong (informativeness), or query what the market "
        "actually looks like (representativeness)? "
        "This app runs the experiment on a Spanish dataset. The answer may surprise you."
        "</p>",
        unsafe_allow_html=True,
    )

    if not any(RESULTS_DIR.glob("*.parquet")):
        st.error(
            f"No results found in `{RESULTS_DIR}`\n\n"
            "Run `notebooks/05_al_simulation.py` first."
        )
        st.stop()

    df_all = load_results()
    n_weeks = df_all["week"].max()

    st.subheader("Strategies")

    available = df_all["strategy"].unique()

    cp_strategies          = [s for s in STRATEGY_LABELS
                               if s.endswith("_cp") and "disruption" not in s and s != "random_cp"]
    gauss_strategies       = [s for s in STRATEGY_LABELS
                               if s.endswith("_gauss") and "disruption" not in s and s != "random_gauss"]
    disruption_strategies  = [s for s in ["disruption_cp", "disruption_gauss"]
                               if s in available]
    restart_strategies     = [s for s in STRATEGY_LABELS
                               if s.endswith("_restart") and s in available]
    tariff_response        = disruption_strategies + restart_strategies

    selected_strategies = []

    with st.expander("Benchmark", expanded=True):
        for s in ["random_market", "random_cp", "random_gauss"]:
            if s in available:
                if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}"):
                    selected_strategies.append(s)

    if "informed_market" in available:
        with st.expander("Market-based AL", expanded=True):
            if st.checkbox(STRATEGY_LABELS["informed_market"], value=True, key="chk_informed_market"):
                selected_strategies.append("informed_market")

    with st.expander("Anchor-based AL — CP", expanded=False):
        for s in cp_strategies:
            if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}"):
                selected_strategies.append(s)

    with st.expander("Anchor-based AL — Gaussian", expanded=False):
        for s in gauss_strategies:
            if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}"):
                selected_strategies.append(s)

    if tariff_response:
        with st.expander("Tariff-change response", expanded=False):
            for s in tariff_response:
                if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}"):
                    selected_strategies.append(s)

    st.subheader("About")
    n_rows = df_all["n_labeled"].max()
    weekly_added = (
        df_all.sort_values(["simulation", "strategy", "week"])
        .groupby(["simulation", "strategy"])["n_labeled"]
        .diff()
        .dropna()
        .median()
    )
    all_sims = df_all["simulation"].unique().tolist()
    st.markdown(
        f"**Simulation:** {n_weeks} weeks  \n"
        f"**Max labeled:** {n_rows:,} profiles  \n"
        f"**~profiles/week:** {int(weekly_added):,}  \n"
        f"**Scenarios:** {len(all_sims)}"
    )

    st.divider()
    st.caption(
        "Based on a real Spanish motor portfolio "
        "([doi: 10.17632/5cxyb5fp4f.2](https://data.mendeley.com/datasets/5cxyb5fp4f/2)). "
        "Strategy rankings may differ with your own data, tariff structure, and budget."
    )
    st.caption("David Fischer · April 2026")

# ── Guard: need at least one strategy ─────────────────────────────────────────

if not selected_strategies:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

# ── Simulation name → label mapping (derived from parquet + config) ────────────
# We build a best-effort display name from the simulation column value.
# Config labels are the ground truth but the parquet only stores the name slug.
# Attempt to load them from config; fall back to the raw name.

@st.cache_data
def simulation_labels() -> dict[str, str]:
    try:
        from market_model_al.config import (
            load_simulation_cfg, load_tariff_changes_cfg, resolve_simulations
        )
        cfg     = load_simulation_cfg(ROOT / "config" / "simulation.yaml")
        library = load_tariff_changes_cfg(ROOT / "config" / "tariff_changes.yaml")
        sims    = resolve_simulations(cfg, library)
        return {s["name"]: s["label"] for s in sims}
    except Exception:
        return {}

sim_labels = simulation_labels()


def sim_display(name: str) -> str:
    return sim_labels.get(name, name)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "Strategy comparison", "Segment breakdown", "Strategy guide",
])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Strategy comparison — all simulations; metric picker drives the chart
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    all_sims_t1 = df_all["simulation"].unique().tolist()
    selected_sim_t1 = st.selectbox(
        "Scenario",
        options=all_sims_t1,
        format_func=sim_display,
        key="sim_select_t1",
    )
    df_t1 = df_all[
        (df_all["simulation"] == selected_sim_t1)
        & df_all["strategy"].isin(selected_strategies)
    ]
    tc_weeks_t1 = tariff_change_weeks(df_t1)

    st.header(sim_display(selected_sim_t1))

    has_cp_t1    = any("_cp"   in s or s in ("random_market", "informed_market") for s in selected_strategies)
    has_gauss_t1 = any("_gauss" in s for s in selected_strategies)
    if has_cp_t1 and has_gauss_t1:
        profile_desc = "Ceteris-paribus and Gaussian profiles are generated and oracle-labeled."
    elif has_gauss_t1:
        profile_desc = "Gaussian perturbation profiles are generated and oracle-labeled."
    else:
        profile_desc = "Ceteris-paribus profiles are generated and oracle-labeled."

    caption_parts = [
        f"Each week the strategy selects which anchor rows to scrape. "
        f"{profile_desc} "
        "Lower RMSE / higher SHAP similarity = faster recovery of the oracle tariff."
    ]
    if tc_weeks_t1:
        weeks_str = ", ".join(f"**week {w}**" for w in tc_weeks_t1)
        caption_parts.append(
            f"Tariff change(s) at {weeks_str} — dashed lines are restart variants."
        )
    st.caption("  ".join(caption_parts))

    # ── Metric picker ─────────────────────────────────────────────────────────
    available_metrics = {
        k: v for k, v in METRIC_OPTIONS.items()
        if v != "shap_cosine_similarity" or has_shap(df_t1)
    }
    metric_label_t1 = st.radio(
        "Metric",
        list(available_metrics.keys()),
        horizontal=True,
        key="metric_radio_t1",
    )
    metric_col_t1 = available_metrics[metric_label_t1]

    if metric_col_t1 == "shap_cosine_similarity":
        st.caption(
            "⚗️ **Simulation-only metric** — requires access to the oracle's internal model "
            "and cannot be observed in real-world deployment."
        )

    fig_t1 = convergence_figure(
        df_t1, metric_col_t1, METRIC_AXIS_LABELS[metric_col_t1],
        selected_strategies, change_weeks=tc_weeks_t1,
        plotly_theme="plotly_dark",
    )
    st.plotly_chart(fig_t1, width="stretch")

    # ── Segment composition heatmap ───────────────────────────────────────────
    seg_cols_t1 = [f"rmse_{seg.key}" for seg in SEGMENTS]
    has_segs_t1 = all(c in df_t1.columns for c in seg_cols_t1)
    st.subheader(
        "Segment breakdown",
        help=(
            "Metric broken down by actuarial segment for a chosen week. "
            "Red = high error, green = low error. "
            "Use the slider to scrub through time and see how each strategy's "
            "error shifts across segments — greedy strategies tend to fix one "
            "segment at the cost of starving others.\n\n"
            "Available for RMSE and Relative RMSE. "
            "SHAP cosine similarity has no per-segment equivalent."
        ),
    )
    if metric_col_t1 == "shap_cosine_similarity":
        st.info(
            "Segment breakdown is not available for SHAP cosine similarity — "
            "there is no per-segment equivalent of this metric.",
            icon="ℹ️",
        )
    elif not has_segs_t1:
        st.info(
            "Segment metrics not found — re-run `notebooks/05_al_simulation.py`.",
            icon="ℹ️",
        )
    else:
        all_weeks_heatmap = sorted(df_t1["week"].unique().tolist())
        heatmap_week = st.select_slider(
            "Week",
            options=all_weeks_heatmap,
            value=all_weeks_heatmap[-1],
            key="heatmap_week_t1",
        )
        fig_heatmap = segment_heatmap(
            df_t1, heatmap_week, selected_strategies,
            metric=metric_col_t1, plotly_theme="plotly_dark",
        )
        st.plotly_chart(fig_heatmap, width="stretch")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("Summary")
    all_weeks_t1 = sorted(df_t1["week"].unique().tolist())
    summary_week_t1 = st.selectbox(
        "Week",
        options=all_weeks_t1,
        index=len(all_weeks_t1) - 1,
        key="summary_week_t1",
    )
    tbl = summary_table(df_t1, selected_strategies, week=summary_week_t1)
    st.dataframe(
        tbl.style.format({
            "Labeled profiles": "{:,.0f}",
            "RMSE (€)":         "{:.2f}",
            "Relative RMSE":    "{:.4f}",
            "SHAP similarity":  "{:.4f}",
        }),
        width="stretch",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Segment breakdown
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    seg_cols   = [f"rmse_{seg.key}" for seg in SEGMENTS]
    has_segs   = all(c in df_all.columns for c in seg_cols)

    if not has_segs:
        st.info(
            "Segment metrics not found in the results file. "
            "Re-run `notebooks/05_al_simulation.py` to generate them."
        )
    else:
        sim_options_t2 = df_all["simulation"].unique().tolist()
        selected_sim_t2 = st.selectbox(
            "Scenario",
            options=sim_options_t2,
            format_func=sim_display,
            key="sim_select_t2",
        )
        df_t2 = df_all[
            (df_all["simulation"] == selected_sim_t2)
            & df_all["strategy"].isin(selected_strategies)
        ]
        tc_weeks_t2 = tariff_change_weeks(df_t2)

        st.header(f"Per-segment RMSE — {sim_display(selected_sim_t2)}")
        st.caption(
            "Each panel shows RMSE on the holdout subset for a specific driver/vehicle segment. "
            "A strategy that excels globally but fails in a segment — or vice versa — is visible here."
        )

        for seg in SEGMENTS:
            col = f"rmse_{seg.key}"
            st.subheader(f"{seg.label}  —  {seg.description}")
            fig = convergence_figure(
                df_t2, col, "RMSE (€)", selected_strategies, change_weeks=tc_weeks_t2,
                plotly_theme="plotly_dark",
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Segment RMSE summary")
        all_weeks_t2 = sorted(df_t2["week"].unique().tolist())
        summary_week_t2 = st.selectbox(
            "Week",
            options=all_weeks_t2,
            index=len(all_weeks_t2) - 1,
            key="summary_week_t2",
        )
        final_seg = (
            df_t2[df_t2["week"] == summary_week_t2]
            .set_index("strategy_label")[seg_cols]
        )
        final_seg.columns = [seg.label for seg in SEGMENTS]
        final_seg.index.name = "Strategy"
        st.dataframe(
            final_seg.style.format("{:.2f}").highlight_min(
                axis=0, props="background-color: #d4edda; color: black;"
            ),
            width="stretch",
        )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Strategy guide
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Strategy guide")
    st.caption(
        "How each strategy selects which profiles to scrape each week. "
        "All strategies share the same weekly profile budget — they differ in "
        "which anchors they pick and how the budget is split across the market space."
    )

    st.info(
        "**Simulation mechanics note:** within a single week, anchors are sampled without "
        "replacement from the candidate pool. However, there is no memory across weeks — "
        "the same anchor row can in principle be selected again in a later week, generating "
        "duplicate CP profiles in the labeled set. In practice this is rare given the large "
        "anchor pool (~100k rows) and small weekly selection (~19 anchors), but it is a known "
        "simplification shared by all strategies.",
        icon="ℹ️",
    )

    st.divider()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    st.subheader("Benchmark")
    st.caption("Random strategies — no model required, no informativeness scoring.")

    for info in [
        {
            "name": "Random market",
            "key": "random_market",
            "summary": "Samples each week from a combined pool of real portfolio rows and a synthetic market supplement.",
            "detail": (
                "The competitor's portfolio does not fully represent aggregator traffic — "
                "there are segments where the competitor quotes but rarely writes business. "
                "This strategy models that gap explicitly: each week it draws a fixed fraction "
                "of the weekly budget from a synthetic CP supplement (default 10 %) and the "
                "remainder from the real portfolio rows. The split is controlled by "
                "`market_supplement_ratio` in `simulation.yaml` and is applied consistently "
                "to the warm start as well, so both phases share the same market composition assumption."
            ),
            "strengths": [
                "Represents the true market space — not just the competitor's written portfolio.",
                "No informativeness scoring required — fast and simple.",
                "Consistent with the warm start: both use the same `market_supplement_ratio`.",
            ],
            "weaknesses": [
                "The `market_supplement_ratio` is a modelling assumption that must be set externally.",
                "Ignores where the competitor model is currently wrong.",
            ],
            "when": "Primary benchmark. Use whenever the competitor's portfolio is known to be selective.",
        },
        {
            "name": "Random (CP)",
            "key": "random_cp",
            "summary": "Selects anchors uniformly at random; generates ceteris-paribus profiles from them.",
            "strengths": [
                "Representative by construction within the anchor pool.",
                "No model required — fastest strategy.",
                "Robust: cannot be misled by a mis-specified scoring model.",
            ],
            "weaknesses": [
                "Anchored to the company portfolio — inherits its selection bias.",
                "Ignores where the competitor model is currently wrong.",
            ],
            "when": "Within-group baseline for CP strategies. Shows the floor that informativeness strategies must beat.",
        },
        {
            "name": "Random (Gaussian)",
            "key": "random_gauss",
            "summary": "Same as Random (CP) but generates joint Gaussian perturbations instead of ceteris-paribus sweeps.",
            "strengths": [
                "Exposes LightGBM to genuine joint-feature variation with no informativeness cost.",
                "No model required — as fast as Random (CP).",
            ],
            "weaknesses": [
                "Inherits the same portfolio selection bias as Random (CP).",
                "Joint variation does not compensate for the absence of natural feature correlations.",
            ],
            "when": "Within-group baseline for Gaussian strategies. Isolates the effect of the profile generator from anchor selection.",
        },
    ]:
        _strategy_card(info)

    st.divider()

    # ── Market-based AL ────────────────────────────────────────────────────────
    st.subheader("Market-based AL")
    st.caption("Draws from the market-representative pool and applies an informativeness filter.")

    _strategy_card({
        "name": "Informed market",
        "key": "informed_market",
        "summary": "Error-based informativeness scoring applied to a representative market pool — the best-of-both-worlds hybrid.",
        "detail": (
            "Each week it builds a large candidate pool using the same market composition as "
            "`random_market` (real portfolio rows + synthetic supplement), scores every candidate "
            "with a proxy model trained on relative residuals, and selects the top `weekly_budget` "
            "rows — those where the competitor model is currently most wrong. "
            "Because the pool is representative by construction, the scoring step cannot starve "
            "mainstream segments the way a globally greedy strategy would."
        ),
        "strengths": [
            "Representative pool ensures no segment is structurally excluded before scoring.",
            "Informativeness filter focuses the budget on where the model is currently wrong.",
            "Preserves natural feature correlations of real portfolio rows.",
        ],
        "weaknesses": [
            "Scoring the full candidate pool adds computational cost vs. pure random_market.",
            "Proxy model quality is limited by labeled set size — early weeks fall back toward random.",
        ],
        "when": (
            "The principled best-of-both-worlds challenger to random_market. "
            "If this hybrid still cannot beat random_market, the conclusion is definitive: "
            "representativeness dominates informativeness in competitor tariff recovery."
        ),
    })

    st.divider()

    # ── Anchor-based AL — CP ───────────────────────────────────────────────────
    st.subheader("Anchor-based AL — CP")
    st.caption(
        "Select anchors from the company portfolio using an informativeness criterion, "
        "then generate ceteris-paribus profiles (one feature swept at a time)."
    )

    for info in [
        {
            "name": "Uncertainty (CP)",
            "key": "uncertainty_cp",
            "summary": "Trains several bootstrap-resampled models and selects anchors where their predictions disagree most.",
            "detail": (
                "Bootstrap resampling fits the same model type multiple times, each time on a random sample "
                "drawn with replacement from the labeled set. Where the models disagree strongly, "
                "the labeled set provides weak or conflicting signal — a proxy for model uncertainty."
            ),
            "strengths": [
                "Targets regions where the model genuinely lacks confidence.",
                "Does not require oracle access during scoring.",
            ],
            "weaknesses": [
                "Bootstrap variance is a proxy for uncertainty, not for actual error.",
                "Concentrates on sparse or high-variance regions, starving well-populated segments.",
                "Slower than random due to fitting multiple models each week.",
            ],
            "when": "Useful when the feature space has large unexplored regions. Less effective once the warm start already covers the main effects.",
        },
        {
            "name": "Error-based (CP)",
            "key": "error_based_cp",
            "summary": "Trains a proxy model on labeled relative residuals and selects anchors predicted to have the highest relative error.",
            "strengths": [
                "Directly targets where the competitor model is currently most wrong.",
                "Uses relative residuals so high-premium policies are not systematically over-sampled.",
                "Recovers specific high-error segments faster than random (e.g. young drivers).",
            ],
            "weaknesses": [
                "Greedy: concentrates budget on the hardest segment, leaving others under-sampled.",
                "Global RMSE and SHAP similarity suffer from the distribution mismatch.",
            ],
            "when": "Best when you care primarily about one known high-error segment rather than global convergence.",
        },
        {
            "name": "Segment-adaptive (CP)",
            "key": "segment_adaptive_cp",
            "summary": "Scores each anchor by the global relative RMSE plus the relative RMSE of every named segment it belongs to.",
            "strengths": [
                "Dynamic: allocation shifts automatically as segment gaps open and close.",
                "No starvation: anchors outside named segments always receive the global baseline score.",
                "Converges toward random as segment gaps close.",
            ],
            "weaknesses": [
                "Reacts to persistent difficulty, not sudden disruption.",
                "Segment definitions are fixed; unlisted segments are invisible to the strategy.",
            ],
            "when": "Good all-round informativeness strategy when you expect persistent difficulty in specific named segments.",
        },
    ]:
        _strategy_card(info)

    st.divider()

    # ── Anchor-based AL — Gaussian ─────────────────────────────────────────────
    st.subheader("Anchor-based AL — Gaussian")
    st.caption(
        "Same anchor-selection logic as the CP variants above, "
        "but profiles are joint Gaussian perturbations instead of ceteris-paribus sweeps."
    )

    with st.expander("**How Gaussian profiles differ from ceteris-paribus profiles**", expanded=False):
        st.markdown(
            """
CP profiles vary **one feature at a time** across its full range while holding all others fixed.
This gives dense 1-D coverage of each marginal effect but provides no joint-feature variation
within a single anchor's batch.

Gaussian profiles vary **all continuous features simultaneously**. For each anchor, N profiles
are drawn by adding independent Gaussian noise to every continuous feature:

> σ_feature = sigma_frac × (feature_max − feature_min)

Values are clipped to the valid range and constraint-validated (e.g. `licence_age ≤ driver_age − 18`).
With the default `sigma_frac = 0.3`, profiles stay near the anchor's natural feature context while
exposing the model to genuine joint-feature variation.

**Budget parity:** profiles per anchor equals the CP constant (254), so both types consume
identical weekly budgets and results are directly comparable.
            """
        )

    for name, key in [
        ("Uncertainty (Gaussian)",         "uncertainty_gauss"),
        ("Error-based (Gaussian)",         "error_based_gauss"),
        ("Segment-adaptive (Gaussian)",    "segment_adaptive_gauss"),
    ]:
        cp_key = key.replace("_gauss", "_cp")
        with st.expander(f"**{name}**", expanded=False):
            st.markdown(
                f"Identical anchor-selection logic to **{STRATEGY_LABELS[cp_key]}** above. "
                f"The only difference is the profile generator: joint Gaussian perturbations "
                f"instead of ceteris-paribus sweeps."
            )

    st.divider()

    # ── Tariff-change response ─────────────────────────────────────────────────
    st.subheader("Tariff-change response")
    st.caption(
        "Reactive strategies designed for competitor repricing events. "
        "Not classic AL — they fire on a monitoring signal rather than an informativeness criterion."
    )

    for info in [
        {
            "name": "Disruption-adaptive (CP)",
            "key": "disruption_cp",
            "summary": "Monitors the week-on-week change in per-segment RMSE; concentrates budget on disrupted segments (≥15% spike).",
            "strengths": [
                "Uses the derivative of RMSE — fires on disruption, not on persistent difficulty.",
                "Does not discard labeled data from unchanged segments.",
                "Automatically resets after recovery.",
            ],
            "weaknesses": [
                "Blind to gradual drift — only reacts to sharp week-on-week spikes.",
                "The 15% threshold is a fixed hyperparameter.",
                "Falls back to random in the first week (no prior RMSE to compare against).",
            ],
            "when": "Best response to sudden, localised tariff changes. Outperforms restart strategies by retaining valid labels from unchanged segments.",
        },
        {
            "name": "Disruption-adaptive (Gaussian)",
            "key": "disruption_gauss",
            "summary": "Same disruption-detection logic as above; Gaussian profiles generated from the targeted anchors.",
            "strengths": [
                "Same monitoring benefits as Disruption-adaptive (CP).",
                "Joint profiles may help recover interaction effects disrupted by the tariff change.",
            ],
            "weaknesses": [
                "Same threshold sensitivity as Disruption-adaptive (CP).",
            ],
            "when": "Use in place of Disruption-adaptive (CP) when you want to test whether joint variation speeds interaction recovery after a tariff shock.",
        },
    ]:
        _strategy_card(info)

    if restart_strategies:
        st.markdown("**Restart variants** clear the labeled set after a tariff change and restart from the warm start. "
                    "They serve as a comparison point for the disruption-adaptive strategies — showing the cost of "
                    "discarding all accumulated labels.")
