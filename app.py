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
    "random":                     "Random",
    "uncertainty":                "Uncertainty",
    "error_based":                "Error-based",
    "segment_adaptive":           "Segment-adaptive",
    "disruption":                 "Disruption-adaptive",
    "random_restart":             "Random (restart)",
    "segment_adaptive_restart":   "Segment-adaptive (restart)",
}

PALETTE = {
    "random":                     "#888888",
    "uncertainty":                "#1f77b4",
    "error_based":                "#ff7f0e",
    "segment_adaptive":           "#9467bd",
    "disruption":                 "#d62728",
    "random_restart":             "#888888",
    "segment_adaptive_restart":   "#9467bd",
}

METRIC_OPTIONS = {
    "RMSE (€)":               "rmse",
    "Relative RMSE":          "rel_rmse",
    "SHAP cosine similarity": "shap_cosine_similarity",
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

@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_parquet(RESULTS_PATH)
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
            marker=dict(size=6),
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


def summary_table(df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Final-week metrics for each strategy, formatted for display."""
    max_week = df["week"].max()
    final = df[df["week"] == max_week].copy()
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
    plotly_theme: str = "plotly_dark",
) -> go.Figure:
    """Heatmap: strategies (rows) × segments (columns), colour = RMSE at *week*.

    Red = high error, green = low error.  Missing values (NaN) are shown in grey.
    """
    seg_keys   = [seg.key   for seg in SEGMENTS]
    seg_labels = [seg.label for seg in SEGMENTS]

    snap = df[df["week"] == week].copy()
    snap = snap[snap["strategy"].isin(strategies)]

    # Build matrix: rows = strategies (in sidebar order), cols = segments
    strat_labels = [STRATEGY_LABELS.get(s, s) for s in strategies if not snap[snap["strategy"] == s].empty]
    strat_order  = [s for s in strategies if not snap[snap["strategy"] == s].empty]

    z, text = [], []
    for strat in strat_order:
        row_df = snap[snap["strategy"] == strat]
        row_z, row_t = [], []
        for key in seg_keys:
            col = f"rmse_{key}"
            val = row_df[col].values[0] if col in row_df.columns and len(row_df) else float("nan")
            row_z.append(val if pd.notna(val) else None)
            row_t.append(f"{val:.2f}" if pd.notna(val) else "n/a")
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
        colorbar=dict(title="RMSE (€)"),
        hoverongaps=False,
        hovertemplate="<b>%{y}</b><br>%{x}<br>RMSE: %{text}<extra></extra>",
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
    n_weeks = df_all["week"].max()

    st.subheader("Strategies")
    base_strategies = [s for s in STRATEGY_LABELS if not s.endswith("_restart")]
    selected_base = [
        s for s in base_strategies
        if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}")
    ]
    selected_strategies = selected_base + [
        f"{s}_restart" for s in ["random", "segment_adaptive"] if s in selected_base
    ]

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
        df_t1, metric_col_t1, metric_label_t1,
        selected_strategies, change_weeks=tc_weeks_t1,
        plotly_theme="plotly_dark",
    )
    st.plotly_chart(fig_t1, width="stretch")

    # ── Segment composition heatmap ───────────────────────────────────────────
    seg_cols_t1 = [f"rmse_{seg.key}" for seg in SEGMENTS]
    if all(c in df_t1.columns for c in seg_cols_t1):
        st.subheader(
            "Segment RMSE heatmap",
            help=(
                "RMSE broken down by actuarial segment for a chosen week. "
                "Red = high error, green = low error. "
                "Use the slider to scrub through time and see how each strategy's "
                "error shifts across segments — greedy strategies tend to fix one "
                "segment at the cost of starving others."
            ),
        )
        all_weeks   = sorted(df_t1["week"].unique().tolist())
        heatmap_week = st.select_slider(
            "Week",
            options=all_weeks,
            value=all_weeks[-1],
            key="heatmap_week_t1",
        )
        fig_heatmap = segment_heatmap(df_t1, heatmap_week, selected_strategies, plotly_theme="plotly_dark")
        st.plotly_chart(fig_heatmap, width="stretch")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader(f"Final-week summary (week {n_weeks})")
    tbl = summary_table(df_t1, selected_strategies)
    st.dataframe(
        tbl.style.format({
            "Labeled profiles": "{:,.0f}",
            "RMSE (€)":         "{:.2f}",
            "Relative RMSE":    "{:.4f}",
            "SHAP similarity":  "{:.4f}",
        }),
        width="stretch",
    )

    # ── Pre vs post tariff change (only when relevant) ────────────────────────
    if tc_weeks_t1:
        first_tc  = tc_weeks_t1[0]
        pre_week  = max(0, first_tc - 1)
        post_week = n_weeks
        st.subheader(f"Pre- vs post-change RMSE (week {pre_week} → week {post_week})")
        compare = (
            df_t1[
                df_t1["week"].isin([pre_week, post_week])
                & df_t1["strategy"].isin(selected_strategies)
            ].copy()
        )
        compare["period"] = compare["week"].apply(
            lambda w: f"Pre-change (wk {pre_week})"
            if w == pre_week else f"Post-change (wk {post_week})"
        )
        pivot = (
            compare
            .pivot_table(index="strategy_label", columns="period", values="rmse")
            .rename_axis("Strategy")
        )
        if not pivot.empty:
            st.dataframe(pivot.style.format("{:.2f}"), width="stretch")

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

        st.subheader(f"Final-week segment RMSE (week {n_weeks})")
        max_week  = df_t2["week"].max()
        final_seg = (
            df_t2[df_t2["week"] == max_week]
            .set_index("strategy_label")[seg_cols]
        )
        final_seg.columns = [seg.label for seg in SEGMENTS]
        final_seg.index.name = "Strategy"
        st.dataframe(
            final_seg.style.format("{:.2f}").highlight_min(axis=0, color="#d4edda"),
            width="stretch",
        )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Strategy guide
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    st.header("Strategy guide")
    st.caption(
        "How each strategy selects which anchor rows to scrape each week. "
        "All strategies share the same weekly profile budget and the same "
        "ceteris-paribus profile generation step — they differ only in *which anchors* they pick."
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
            "name": "Random",
            "key": "random",
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
            "name": "Uncertainty",
            "key": "uncertainty",
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
            "name": "Error-based",
            "key": "error_based",
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
            "name": "Segment-adaptive",
            "key": "segment_adaptive",
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
            "name": "Disruption-adaptive",
            "key": "disruption",
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
