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


def has_shap(df: pd.DataFrame) -> bool:
    """True when the results contain non-NaN SHAP similarity values."""
    return "shap_cosine_similarity" in df.columns and df["shap_cosine_similarity"].notna().any()


def summary_table(df: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    """Final-week metrics for each strategy, formatted for display."""
    max_week = df["week"].max()
    final = df[df["week"] == max_week].copy()
    final = final[final["strategy"].isin(strategies)]
    cols = ["n_labeled", "rmse", "rel_rmse"]
    col_labels = ["Labeled profiles", "RMSE (€)", "Relative RMSE"]
    if has_shap(df):
        cols.append("shap_cosine_similarity")
        col_labels.append("SHAP similarity")
    final = final.set_index("strategy_label")[cols]
    final.index.name = "Strategy"
    final.columns = col_labels
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
    base_strategies = [s for s in STRATEGY_LABELS if not s.endswith("_restart")]
    selected_base = [
        s for s in base_strategies
        if st.checkbox(STRATEGY_LABELS[s], value=True, key=f"chk_{s}")
    ]
    # Restart variants follow their parent automatically (shown in tab 2 only)
    selected_strategies = selected_base + [
        f"{s}_restart" for s in ["random", "segment_adaptive"] if s in selected_base
    ]

    st.subheader("Primary metric")
    metric_label = st.radio("Metric", list(METRIC_OPTIONS.keys()), index=0, label_visibility="collapsed")
    metric_col = METRIC_OPTIONS[metric_label]

    st.subheader("About")
    n_weeks = df_all["week"].max()
    n_rows  = df_all[df_all["scenario"] == "no_tariff_change"]["n_labeled"].max()
    weekly_added = (
        df_all[df_all["scenario"] == "no_tariff_change"]
        .sort_values(["strategy", "week"])
        .groupby("strategy")["n_labeled"]
        .diff()
        .dropna()
        .median()
    )
    st.markdown(
        f"**Simulation:** {n_weeks} weeks  \n"
        f"**Max labeled:** {n_rows:,} profiles  \n"
        f"**~profiles/week:** {int(weekly_added):,}  \n"
        f"**Scenarios:** {', '.join(df_all['scenario'].unique())}"
    )

# ── Guard: need at least one strategy ─────────────────────────────────────────

if not selected_strategies:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Strategy comparison", "Tariff change recovery", "Segment breakdown", "Strategy guide"])

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Strategy comparison (no tariff change)
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    df_s1 = df_all[df_all["scenario"] == "no_tariff_change"]
    df_s1 = df_s1[df_s1["strategy"].isin(selected_strategies)]

    st.header("Strategy comparison — no tariff change")
    st.caption(
        "Each week the strategy selects which anchor rows to scrape. "
        "All ceteris-paribus profiles from the selected anchors are generated and labeled. "
        "Lower RMSE and higher SHAP similarity = the competitor model recovers the oracle tariff faster."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(
            "RMSE on holdout (€)",
            help=(
                "The holdout is a fixed set of 2,000 real policy rows drawn from the dataset "
                "before any training begins. These rows are oracle-labeled once and never used "
                "as anchors or training data during the AL loop. RMSE is computed on this set "
                "each week, so it is comparable across strategies and weeks.\n\n"
                "Because the holdout is drawn from the real data distribution, it is "
                "population-representative — a strategy that performs well here has learned "
                "the tariff broadly, not just in the regions it chose to scrape."
            ),
        )
        fig_rmse = convergence_figure(df_s1, "rmse", "RMSE (€)", selected_strategies)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        if has_shap(df_s1):
            st.subheader(
                "SHAP cosine similarity  ⚗️ simulation only",
                help=(
                    "Measures how well the competitor model has recovered the global tariff structure "
                    "of the oracle — not just accuracy in specific regions, but whether each feature "
                    "pushes prices in the right direction and with the right relative magnitude. "
                    "A score of 1 means perfect structural alignment.\n\n"
                    "Simulation-only metric: computed by comparing the competitor model's SHAP values "
                    "against the oracle's SHAP values on the holdout set. In a real-world deployment "
                    "you do not have access to the competitor's internal model, so this metric cannot "
                    "be observed in practice. It is included here as a diagnostic to reveal how well "
                    "each strategy recovers the underlying tariff structure, not just prediction accuracy."
                ),
            )
            fig_shap = convergence_figure(
                df_s1, "shap_cosine_similarity", "Cosine similarity", selected_strategies
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP similarity was disabled in `config/simulation.yaml` for this run.", icon="ℹ️")

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
    tc_scenario_names = [s for s in df_all["scenario"].unique() if s != "no_tariff_change"]

    if not tc_scenario_names:
        st.info("No tariff-change scenarios found in the results. "
                "Add entries to `perturbation_schedule` in `config/simulation.yaml` "
                "and re-run the simulation.")
        st.stop()

    selected_scenario = st.selectbox(
        "Tariff-change scenario",
        options=tc_scenario_names,
        key="tc_scenario_select",
    )

    df_s2 = df_all[df_all["scenario"] == selected_scenario]
    df_s2 = df_s2[df_s2["strategy"].isin(selected_strategies)]

    # Find the tariff change week from the data
    change_weeks = df_s2[df_s2["post_tariff_change"]]["week"]
    tariff_week  = int(change_weeks.min()) if len(change_weeks) else None

    st.header("Tariff change recovery")
    if tariff_week is not None:
        st.caption(
            f"Tariff change injected at **week {tariff_week}**. "
            "RMSE is measured against the *new* oracle after that point. "
            "A good strategy recovers quickly without a full restart."
        )
    else:
        st.caption("No tariff change detected in the results.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(
            "RMSE recovery",
            help=(
                "The holdout is a fixed set of 2,000 real policy rows drawn from the dataset "
                "before any training begins. These rows are oracle-labeled once and never used "
                "as anchors or training data during the AL loop. After the tariff change, "
                "holdout labels are recomputed with the new oracle — so RMSE measures recovery "
                "of the new tariff, not the old one.\n\n"
                "Because the holdout is drawn from the real data distribution, it is "
                "population-representative — a strategy that performs well here has learned "
                "the new tariff broadly, not just in the repriced segment."
            ),
        )
        fig_tc_rmse = convergence_figure(
            df_s2, "rmse", "RMSE (€)", selected_strategies,
            tariff_change_week=tariff_week,
        )
        st.plotly_chart(fig_tc_rmse, use_container_width=True)

    with col2:
        if has_shap(df_s2):
            st.subheader(
                "SHAP similarity recovery  ⚗️ simulation only",
                help=(
                    "Measures how well the competitor model has recovered the global tariff structure "
                    "of the oracle — not just accuracy in specific regions, but whether each feature "
                    "pushes prices in the right direction and with the right relative magnitude. "
                    "A score of 1 means perfect structural alignment.\n\n"
                    "Simulation-only metric: computed by comparing the competitor model's SHAP values "
                    "against the oracle's SHAP values on the holdout set. In a real-world deployment "
                    "you do not have access to the competitor's internal model, so this metric cannot "
                    "be observed in practice. It is included here as a diagnostic to reveal how well "
                    "each strategy recovers the underlying tariff structure, not just prediction accuracy."
                ),
            )
            fig_tc_shap = convergence_figure(
                df_s2, "shap_cosine_similarity", "Cosine similarity", selected_strategies,
                tariff_change_week=tariff_week,
            )
            st.plotly_chart(fig_tc_shap, use_container_width=True)
        else:
            st.info("SHAP similarity was disabled in `config/simulation.yaml` for this run.", icon="ℹ️")

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

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Segment breakdown
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    # Import segment metadata for labels/descriptions
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).parent / "src"))
    from market_model_al.segments import SEGMENTS

    seg_cols = [f"rmse_{seg.key}" for seg in SEGMENTS]
    has_segments = all(c in df_all.columns for c in seg_cols)

    if not has_segments:
        st.info(
            "Segment metrics not found in the results file. "
            "Re-run `notebooks/05_al_simulation.py` to generate them."
        )
    else:
        df_s1_seg = df_all[
            (df_all["scenario"] == "no_tariff_change")
            & df_all["strategy"].isin(selected_strategies)
        ]

        st.header("Per-segment RMSE — no tariff change")
        st.caption(
            "Each panel shows RMSE on the holdout subset for a specific driver/vehicle segment. "
            "A strategy that excels globally but fails in a segment — or vice versa — is visible here."
        )

        # One row of charts per segment
        for seg in SEGMENTS:
            col = f"rmse_{seg.key}"
            st.subheader(f"{seg.label}  —  {seg.description}")
            fig = convergence_figure(
                df_s1_seg, col, "RMSE (€)", selected_strategies
            )
            st.plotly_chart(fig, use_container_width=True)

        # Final-week segment summary table
        st.subheader(f"Final-week segment RMSE (week {n_weeks})")
        max_week = df_s1_seg["week"].max()
        final_seg = (
            df_s1_seg[df_s1_seg["week"] == max_week]
            .set_index("strategy_label")[seg_cols]
        )
        final_seg.columns = [seg.label for seg in SEGMENTS]
        final_seg.index.name = "Strategy"
        st.dataframe(
            final_seg.style.format("{:.2f}").highlight_min(axis=0, color="#d4edda"),
            use_container_width=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: Strategy guide
# ──────────────────────────────────────────────────────────────────────────────

with tab4:
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
            "tag": "Baseline",
            "tag_color": "#888888",
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
            "tag": "Informativeness",
            "tag_color": "#1f77b4",
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
            "tag": "Informativeness",
            "tag_color": "#ff7f0e",
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
            "tag": "Adaptive",
            "tag_color": "#9467bd",
            "summary": "Scores each anchor by the global relative RMSE plus the relative RMSE of every named segment it belongs to. Anchors in high-error segments are prioritised; anchors outside all segments compete on the global baseline.",
            "strengths": [
                "Dynamic: allocation shifts automatically as segment gaps open and close.",
                "Uses relative RMSE (RMSE / mean premium) so high-premium segments are not permanently over-sampled.",
                "No starvation: anchors outside named segments always receive the global baseline score.",
                "Converges toward random as segment gaps close.",
            ],
            "weaknesses": [
                "Reacts to persistent difficulty, not sudden disruption — segments that are always hard receive extra budget every week, even when nothing has changed.",
                "Segment definitions are fixed; segments not in the list are invisible to the strategy.",
            ],
            "when": "Good all-round strategy when you expect persistent difficulty in specific named segments.",
        },
        {
            "name": "Disruption-adaptive",
            "key": "disruption",
            "tag": "Adaptive",
            "tag_color": "#d62728",
            "summary": "Monitors the week-on-week *change* in per-segment RMSE. When a segment's RMSE increases by more than 15% relative to the previous week, the full budget is concentrated on random sampling within the disrupted segment(s). Reverts to global random when no disruption is detected.",
            "strengths": [
                "Uses the derivative of RMSE, not its level — fires on disruption, not on permanent difficulty.",
                "Robust to segments that are always hard (e.g. young drivers): ignores them unless they suddenly worsen.",
                "Does not discard any labeled data — old labels from unchanged segments remain valid and in the training set.",
                "Automatically resets after recovery: no manual intervention needed.",
            ],
            "weaknesses": [
                "Blind to gradual drift — only reacts to sharp week-on-week spikes.",
                "The 15% threshold is a fixed hyperparameter; too low triggers false positives, too high misses soft changes.",
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
