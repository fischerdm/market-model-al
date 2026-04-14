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
RESULTS_PATH = ROOT / "outputs" / "al_results" / "results.parquet"

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
    "random_cp_restart":            "Random (CP, restart)",
    "segment_adaptive_cp_restart":  "Segment-adaptive (CP, restart)",
    # Gaussian-perturbation variants
    "random_gauss":                 "Random (Gaussian)",
    "uncertainty_gauss":            "Uncertainty (Gaussian)",
    "error_based_gauss":            "Error-based (Gaussian)",
    "segment_adaptive_gauss":       "Segment-adaptive (Gaussian)",
    "disruption_gauss":             "Disruption-adaptive (Gaussian)",
}

PALETTE = {
    # CP strategies
    "random_cp":                    "#888888",
    "random_market":                "#17becf",
    "uncertainty_cp":               "#1f77b4",
    "error_based_cp":               "#ff7f0e",
    "segment_adaptive_cp":          "#9467bd",
    "disruption_cp":                "#d62728",
    "random_cp_restart":            "#888888",
    "segment_adaptive_cp_restart":  "#9467bd",
    # Gaussian variants — lighter tints of their CP counterparts
    "random_gauss":                 "#cccccc",
    "uncertainty_gauss":            "#aec7e8",
    "error_based_gauss":            "#ffbb78",
    "segment_adaptive_gauss":       "#c5b0d5",
    "disruption_gauss":             "#f7b6b6",
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
    df = pd.read_parquet(RESULTS_PATH)
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


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AL Competitor Model",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "An insurer scraping competitor quotes from an aggregator website can't afford to "
            "scrape everything — it has to choose *what* to request. This app explores that "
            "choice using active learning (AL): a simulated competitor model is retrained each "
            "week on oracle-labeled ceteris-paribus profiles, and six query strategies compete "
            "to recover the competitor's tariff as fast as possible.\n\n"
            "The oracle is a LightGBM model trained on a real Spanish motor vehicle portfolio "
            "([Mendeley Data, doi: 10.17632/5cxyb5fp4f.2](https://data.mendeley.com/datasets/5cxyb5fp4f/2)). "
            "Strategies are evaluated on holdout RMSE, per-segment RMSE, and SHAP cosine similarity.\n\n"
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
        "Simulate how an insurer reverse-engineers a competitor's tariff by scraping quotes "
        "from an aggregator — and test which active learning strategy gets there fastest. "
        "A strategy where anchors for ceteris-paribus perturbations are randomly selected is harder to beat than you'd think. "
        "Explore why — and when it isn't — with this configurable app."
        "</p>",
        unsafe_allow_html=True,
    )

    if not RESULTS_PATH.exists():
        st.error(
            f"Results file not found:\n`{RESULTS_PATH}`\n\n"
            "Run `notebooks/05_al_simulation.py` first."
        )
        st.stop()

    df_all = load_results()
    n_weeks = df_all["week"].max()

    st.subheader("Strategies")
    base_strategies    = [s for s in STRATEGY_LABELS if not s.endswith("_restart")]
    restart_strategies = [s for s in STRATEGY_LABELS if s.endswith("_restart")]

    selected_base = [
        s for s in base_strategies
        if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}")
    ]

    st.caption("Restart variants")
    # Only show restart variants that exist in the results
    available_restarts = [s for s in restart_strategies if s in df_all["strategy"].unique()]
    selected_restarts = [
        s for s in available_restarts
        if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}")
    ]

    selected_strategies = selected_base + selected_restarts

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
        f"**Simulations:** {len(all_sims)}"
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
        "Simulation",
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

    caption_parts = [
        "Each week the strategy selects which anchor rows to scrape. "
        "All ceteris-paribus profiles are generated and oracle-labeled. "
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
            "Simulation",
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

    strategies_info = [
        {
            "name": "Random (CP)",
            "key": "random_cp",
            "summary": "Selects anchors uniformly at random from the candidate pool each week.",
            "strengths": [
                "Representative by construction — every region of feature space is sampled in proportion to its true frequency.",
                "No model required to score candidates, so it is the fastest strategy.",
                "Robust: cannot be fooled by a mis-specified scoring model.",
            ],
            "weaknesses": [
                "Ignores all information about where the competitor model is currently wrong.",
                "Budget is spread evenly even when some segments are well-understood and others are not.",
            ],
            "when": "Strong general-purpose baseline. Hard to beat on global RMSE and SHAP similarity because it never sacrifices representativeness.",
        },
        {
            "name": "Random market",
            "key": "random_market",
            "summary": "Samples each week from a combined pool of real portfolio rows and ceteris-paribus profiles, weighted by the assumed market coverage ratio.",
            "detail": (
                "The competitor's portfolio does not fully represent aggregator traffic — "
                "there are segments where the competitor quotes but rarely writes business. "
                "This strategy models that gap explicitly: each week it generates a CP pool "
                "from a small number of random anchors, then draws a fixed fraction of the "
                "weekly budget from that pool (default 10 %) and the remainder from the real "
                "portfolio rows. The split is controlled by `market_cp_ratio` in `simulation.yaml` "
                "and is applied consistently to the warm start as well, so the two phases share "
                "the same market composition assumption."
            ),
            "strengths": [
                "Represents the true market space — not just the competitor's written portfolio.",
                "Consistent with the warm start: both use the same `market_cp_ratio`.",
                "No informativeness scoring required — fast and simple like random.",
                "The CP ratio and anchor count are independently configurable.",
            ],
            "weaknesses": [
                "CP pool generation adds overhead compared to pure random sampling.",
                "The market_cp_ratio is a modelling assumption that must be set externally.",
                "Like random, it ignores where the competitor model is currently wrong.",
            ],
            "when": "Use as the primary benchmark when the competitor's portfolio is known to be selective — i.e. when pure random from the portfolio would under-represent segments the competitor prices but doesn't write.",
        },
        {
            "name": "Uncertainty (CP)",
            "key": "uncertainty_cp",
            "summary": "Trains several bootstrap-resampled models and selects anchors where their predictions disagree most (high variance).",
            "detail": (
                "Bootstrap resampling means fitting the same model type multiple times, each time on a random sample "
                "drawn with replacement from the labeled set. Because each bootstrap sample omits some rows and "
                "duplicates others, the resulting models differ slightly. Where they disagree strongly on a prediction, "
                "the labeled set provides weak or conflicting signal in that region — a proxy for model uncertainty."
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
                "Uses relative residuals (error / premium) so high-premium policies are not systematically over-sampled.",
                "Recovers specific high-error segments faster than random (e.g. young drivers).",
            ],
            "weaknesses": [
                "Greedy: concentrates budget on the hardest segment, leaving others under-sampled.",
                "Global RMSE and SHAP similarity suffer from the resulting distribution mismatch.",
            ],
            "when": "Best when you care primarily about one known high-error segment rather than global convergence.",
        },
        {
            "name": "Segment-adaptive (CP)",
            "key": "segment_adaptive_cp",
            "summary": "Scores each anchor by the global relative RMSE plus the relative RMSE of every named segment it belongs to.",
            "strengths": [
                "Dynamic: allocation shifts automatically as segment gaps open and close.",
                "Uses relative RMSE so high-premium segments are not permanently over-sampled.",
                "No starvation: anchors outside named segments always receive the global baseline score.",
                "Converges toward random as segment gaps close.",
            ],
            "weaknesses": [
                "Reacts to persistent difficulty, not sudden disruption.",
                "Segment definitions are fixed; segments not in the list are invisible to the strategy.",
            ],
            "when": "Good all-round strategy when you expect persistent difficulty in specific named segments.",
        },
        {
            "name": "Disruption-adaptive (CP)",
            "key": "disruption_cp",
            "summary": "Monitors the week-on-week *change* in per-segment RMSE. Concentrates budget on disrupted segments (≥15% relative RMSE increase); reverts to global random otherwise.",
            "strengths": [
                "Uses the derivative of RMSE, not its level — fires on disruption, not on permanent difficulty.",
                "Robust to segments that are always hard: ignores them unless they suddenly worsen.",
                "Does not discard any labeled data — old labels from unchanged segments remain valid.",
                "Automatically resets after recovery.",
            ],
            "weaknesses": [
                "Blind to gradual drift — only reacts to sharp week-on-week spikes.",
                "The 15% threshold is a fixed hyperparameter.",
                "Falls back to random on the first week (no prior RMSE to compare against).",
            ],
            "when": "Best response to sudden, localised tariff changes. Outperforms restart strategies because it does not discard valid labels from unchanged segments.",
        },
    ]

    for info in strategies_info:
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

    st.divider()
    st.subheader("Gaussian-perturbation variants  `(*_gauss)`")
    st.markdown(
        "Each CP strategy above has a Gaussian counterpart that uses the same anchor-selection "
        "logic but replaces the ceteris-paribus profile generator with **joint Gaussian perturbations**."
    )
    with st.expander("**How Gaussian profiles differ from ceteris-paribus profiles**", expanded=False):
        st.markdown(
            """
Ceteris-paribus (CP) profiles vary **one feature at a time** across its full range while
holding all other features fixed.  This gives dense 1-D coverage of each marginal effect but
provides no joint-feature variation within a single anchor's batch — LightGBM must infer
interactions from the way different anchors happen to combine.

Gaussian profiles vary **all continuous features simultaneously**.  For each anchor, N profiles
are drawn by adding independent Gaussian noise to every continuous feature:

> σ_feature = sigma_frac × (feature_max − feature_min)

Values are clipped to the valid feature range and then constraint-validated (e.g.
`licence_age ≤ driver_age − 18`).  With the default `sigma_frac = 0.3`, the profiles stay
within roughly one standard deviation of the anchor's own values, preserving the anchor's
natural context (e.g. a young-driver anchor generates young-driver profiles) while exposing
the model to real joint-feature variation.

**Budget parity:** the number of profiles per anchor is set equal to the CP constant (254),
so both profile types consume identical weekly budgets and results are directly comparable.

**Key hypothesis:** if joint profiles allow LightGBM to learn interaction effects more
efficiently than 1-D CP sweeps, Gaussian strategies should converge faster — especially
when the AL strategy is already targeting the right anchors.
            """
        )
    gauss_strategies_info = [
        ("Random (Gaussian)",             "random_gauss",           "Same as Random (CP) but profiles are joint Gaussian perturbations instead of 1-D CP sweeps. The primary baseline for testing whether the profile generator matters independently of anchor selection."),
        ("Uncertainty (Gaussian)",        "uncertainty_gauss",      "Bootstrap uncertainty scoring selects anchors; Gaussian profiles are generated from them. Tests whether joint variation amplifies the uncertainty strategy's coverage gains."),
        ("Error-based (Gaussian)",        "error_based_gauss",      "Proxy-model error scoring selects anchors in high-error regions; Gaussian profiles keep the budget focused there while varying features jointly."),
        ("Segment-adaptive (Gaussian)",   "segment_adaptive_gauss", "Segment RMSE allocation selects anchors; Gaussian profiles ensure the chosen segment receives varied joint coverage rather than axis-aligned sweeps."),
        ("Disruption-adaptive (Gaussian)","disruption_gauss",       "Disruption detection selects anchors in newly-shocked segments; Gaussian profiles help the model recover interactions disrupted by the tariff change."),
    ]
    for name, key, desc in gauss_strategies_info:
        cp_key = key.replace("_gauss", "_cp")   # e.g. random_gauss → random_cp
        with st.expander(f"**{name}**  —  {desc}", expanded=False):
            st.markdown(
                f"Identical anchor-selection logic to **{STRATEGY_LABELS[cp_key]}** "
                f"above. The only difference is the profile generator: joint Gaussian perturbations "
                f"instead of ceteris-paribus sweeps. See the explainer above for details."
            )
