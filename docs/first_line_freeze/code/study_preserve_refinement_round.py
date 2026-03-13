from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backtest import BacktestConfig, run_backtest
from layer12_structural_research_helpers import (
    FROZEN_LAYER12_CONTROL_CONFIG,
    add_forward_entry_quality,
    build_trade_audit,
    deep_copy_config,
    load_closed_base_frames,
    parse_times,
    run_research_variant,
    save_yaml,
    summarize_returns,
)
from risk import RiskConfig, apply_risk_layer
from study_entry_edge_optimization import (
    apply_entry_quality_score,
    build_execution_candidate_table,
    build_review_chart,
    build_score_spec,
)
from study_fragile_label_refinement import assign_fragile_subtypes
from study_preserve_first_robustness import add_context_splits
from study_preserve_suppress_ranking import (
    CONTROL_VARIANT,
    add_interaction_features,
    apply_dual_score,
    build_dual_score_spec,
    write_md,
)


VARIANT_ORDER = [
    "control",
    "prior_preserve_first_variant",
    "refined_preserve_variant_1",
    "refined_preserve_variant_2",
    "best_effort_refined_preserve_variant",
]


def compute_equity_drawdown(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
    equity = 1.0
    peak = 1.0
    rows = []
    for row in trades.sort_values("exit_time").itertuples(index=False):
        equity *= 1.0 + float(row.net_return)
        peak = max(peak, equity)
        rows.append({"timestamp": row.exit_time, "equity": equity, "drawdown": equity / peak - 1.0})
    return pd.DataFrame(rows)


def build_scored_candidate(root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = deep_copy_config()
    _, boxes_1h, boxes_4h, enriched_1h = load_closed_base_frames(cfg)
    control_result = run_research_variant(cfg, boxes_1h, boxes_4h, enriched_1h, root / "control", CONTROL_VARIANT)
    control_audit = add_forward_entry_quality(build_trade_audit(control_result), control_result["merged"])
    control_audit.to_csv(root / "control_trade_audit.csv", index=False)

    candidate = build_execution_candidate_table(control_result, control_audit)
    coarse_spec = build_score_spec(candidate)
    candidate = apply_entry_quality_score(candidate, coarse_spec)
    candidate = assign_fragile_subtypes(candidate)
    candidate, interaction_meta = add_interaction_features(candidate)
    interaction_meta.to_csv(root / "interaction_catalog.csv", index=False)

    fragile = candidate[candidate["fragile_trade"]].copy()
    preserve_spec = build_dual_score_spec(fragile, positive_label="high_upside", target_name="preserve")
    suppress_spec = build_dual_score_spec(fragile, positive_label="hopeless", target_name="suppress")
    preserve_spec.to_csv(root / "preserve_score_spec.csv", index=False)
    suppress_spec.to_csv(root / "suppress_score_spec.csv", index=False)

    scored = apply_dual_score(candidate, preserve_spec, "preserve")
    scored = apply_dual_score(scored, suppress_spec, "suppress")
    scored, split_meta = add_context_splits(scored, control_result["merged"])
    split_meta.to_csv(root / "split_metadata.csv", index=False)
    scored.to_csv(root / "control_entry_candidates_preserve_refinement_scored.csv", index=False)
    return control_result, control_audit, scored, cfg


def build_refinement_flags(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    fragile_like = out["score_bucket"].astype(str) == "fragile"
    preserve_high = out["preserve_bucket"].astype(str) == "high"
    suppress_high = out["suppress_bucket"].astype(str) == "high"
    prev_reject = out["prev_reject_combo"].fillna(False).astype(bool)
    reentry = pd.to_numeric(out["reentry_strength"], errors="coerce")
    suppress_score = pd.to_numeric(out["suppress_score_pct"], errors="coerce")
    box_width_pct = pd.to_numeric(out["box_width_pct"], errors="coerce")

    out["flag_prior_remove"] = fragile_like & suppress_high & ~preserve_high
    out["flag_weak_preserve_v1"] = fragile_like & preserve_high & (reentry < 0.10) & (suppress_score >= (1.0 / 3.0)) & ~prev_reject
    out["flag_weak_preserve_v2"] = fragile_like & preserve_high & (reentry < 0.12) & (suppress_score >= (1.0 / 3.0)) & ~prev_reject & (box_width_pct < 0.03)
    out["flag_weak_preserve_best"] = fragile_like & preserve_high & (reentry < 0.12) & (suppress_score >= (1.0 / 3.0)) & ~prev_reject

    out["guard_prior_preserve_high"] = preserve_high
    out["guard_refined_preserve_1"] = preserve_high & ~out["flag_weak_preserve_v1"]
    out["guard_refined_preserve_2"] = preserve_high & ~out["flag_weak_preserve_v2"]
    out["guard_best_refined_preserve"] = preserve_high & ~out["flag_weak_preserve_best"]
    return out


def filter_mask_for_variant(scored: pd.DataFrame, variant_name: str) -> pd.Series:
    if variant_name == "prior_preserve_first_variant":
        return scored["flag_prior_remove"]
    if variant_name == "refined_preserve_variant_1":
        return scored["flag_prior_remove"] | scored["flag_weak_preserve_v1"]
    if variant_name == "refined_preserve_variant_2":
        return scored["flag_prior_remove"] | scored["flag_weak_preserve_v2"]
    if variant_name == "best_effort_refined_preserve_variant":
        return scored["flag_prior_remove"] | scored["flag_weak_preserve_best"]
    if variant_name == "control":
        return pd.Series(False, index=scored.index)
    raise ValueError(f"Unknown variant: {variant_name}")


def run_variant(control_result: dict, scored: pd.DataFrame, variant_name: str) -> dict:
    config = deep_copy_config()
    merged = control_result["merged"].copy()
    execution_false_break = control_result["execution"]["false_break"].copy()
    execution_boundary = control_result["execution"]["boundary"].copy()

    filtered = execution_false_break.copy()
    mask = filter_mask_for_variant(scored, variant_name)
    filtered_ts = set(scored.loc[mask, "timestamp"].tolist())
    row_mask = filtered["timestamp"].isin(filtered_ts) & (filtered["exec_signal"] != 0)
    filtered.loc[row_mask, "exec_signal"] = 0
    filtered.loc[row_mask, "gate_pass"] = False
    filtered.loc[row_mask, "blocked_reason"] = filtered.loc[row_mask, "blocked_reason"].fillna("").astype(str).apply(
        lambda value: f"{value}|{variant_name}".strip("|")
    )

    execution_outputs = {
        "false_break": filtered,
        "boundary": execution_boundary,
        "combined": pd.concat([filtered, execution_boundary], ignore_index=True).sort_values(["timestamp", "engine"]).reset_index(drop=True),
    }
    risk_cfg = RiskConfig(**config["risk"])
    risk_outputs = apply_risk_layer(merged, execution_outputs, risk_cfg)
    bt_cfg = BacktestConfig(**config["backtest"])
    backtest_outputs = run_backtest(risk_outputs["trades_false_break"], risk_outputs["trades_boundary"], bt_cfg)
    return {
        "variant": variant_name,
        "backtest": backtest_outputs,
        "filtered_timestamps": filtered_ts,
    }

def summarize_variant(
    variant_name: str,
    trades: pd.DataFrame,
    scored_lookup: pd.DataFrame,
    filtered_ts: set[pd.Timestamp],
    annualization_hours: int,
) -> dict[str, object]:
    trades = parse_times(trades.copy(), ["entry_time", "exit_time"])
    summary = summarize_returns(trades, annualization_hours)
    removed = scored_lookup[scored_lookup["entry_time"].isin(filtered_ts)].copy()
    retained = scored_lookup[~scored_lookup["entry_time"].isin(filtered_ts)].copy()
    preserve_guard_col = {
        "control": "guard_prior_preserve_high",
        "prior_preserve_first_variant": "guard_prior_preserve_high",
        "refined_preserve_variant_1": "guard_refined_preserve_1",
        "refined_preserve_variant_2": "guard_refined_preserve_2",
        "best_effort_refined_preserve_variant": "guard_best_refined_preserve",
    }[variant_name]
    preserve_high = retained[(retained["fragile_trade"].fillna(False)) & retained[preserve_guard_col].fillna(False)]
    suppress_high = retained[(retained["fragile_trade"].fillna(False)) & (retained["suppress_bucket"].astype(str) == "high")]
    time_stop_max_contrib = float(trades.loc[trades["exit_reason"] == "time_stop_max", "net_return"].sum()) if not trades.empty else 0.0
    return {
        "variant": variant_name,
        **summary,
        "avg_return_per_trade": float(summary["avg_return"]),
        "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
        "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
        "time_stop_max_contribution": time_stop_max_contrib,
        "removed_trades": int(len(removed)),
        "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()),
        "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()),
        "removed_fragile_mixed": int((removed["fragile_subtype"] == "fragile_mixed").sum()),
        "retained_fragile_high_upside": int(((retained["fragile_subtype"] == "fragile_high_upside") & retained["fragile_trade"].fillna(False)).sum()),
        "preserve_high_oracle_high_upside_rate": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
        "preserve_high_hopeless_contamination": float((preserve_high["fragile_focus_label"] == "hopeless").mean()) if not preserve_high.empty else np.nan,
        "suppress_high_oracle_hopeless_rate": float((suppress_high["fragile_focus_label"] == "hopeless").mean()) if not suppress_high.empty else np.nan,
    }


def build_identity_and_purity_tables(root: Path, scored: pd.DataFrame, variant_results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = scored[scored["baseline_trade"].fillna(False)].copy()
    for variant in VARIANT_ORDER[1:]:
        base[f"removed_{variant}"] = base["entry_time"].isin(variant_results[variant]["filtered_timestamps"])
    base["newly_removed_vs_prior_refined_1"] = base["removed_refined_preserve_variant_1"] & ~base["removed_prior_preserve_first_variant"]
    base["newly_removed_vs_prior_refined_2"] = base["removed_refined_preserve_variant_2"] & ~base["removed_prior_preserve_first_variant"]
    base["newly_removed_vs_prior_best"] = base["removed_best_effort_refined_preserve_variant"] & ~base["removed_prior_preserve_first_variant"]
    identity_cols = [
        "entry_time",
        "exit_reason",
        "net_return",
        "fragile_subtype",
        "fragile_focus_label",
        "preserve_bucket",
        "suppress_bucket",
        "removed_prior_preserve_first_variant",
        "removed_refined_preserve_variant_1",
        "removed_refined_preserve_variant_2",
        "removed_best_effort_refined_preserve_variant",
        "newly_removed_vs_prior_refined_1",
        "newly_removed_vs_prior_refined_2",
        "newly_removed_vs_prior_best",
    ]
    identity = base[[col for col in identity_cols if col in base.columns]].copy()
    identity.to_csv(root / "preserved_removed_trade_identity.csv", index=False)

    purity_rows = []
    impact_rows = []
    for variant in VARIANT_ORDER:
        filtered_ts = variant_results[variant]["filtered_timestamps"] if variant != "control" else set()
        removed = base[base["entry_time"].isin(filtered_ts)].copy()
        retained = base[~base["entry_time"].isin(filtered_ts)].copy()
        preserve_guard_col = {
            "control": "guard_prior_preserve_high",
            "prior_preserve_first_variant": "guard_prior_preserve_high",
            "refined_preserve_variant_1": "guard_refined_preserve_1",
            "refined_preserve_variant_2": "guard_refined_preserve_2",
            "best_effort_refined_preserve_variant": "guard_best_refined_preserve",
        }[variant]
        preserve_high = retained[(retained["fragile_trade"].fillna(False)) & retained[preserve_guard_col].fillna(False)]
        suppress_high = retained[(retained["fragile_trade"].fillna(False)) & (retained["suppress_bucket"].astype(str) == "high")]
        purity_rows.append(
            {
                "variant": variant,
                "preserve_high_trades": int(len(preserve_high)),
                "preserve_high_oracle_high_upside_rate": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
                "preserve_high_hopeless_contamination": float((preserve_high["fragile_focus_label"] == "hopeless").mean()) if not preserve_high.empty else np.nan,
                "suppress_high_trades": int(len(suppress_high)),
                "suppress_high_oracle_hopeless_rate": float((suppress_high["fragile_focus_label"] == "hopeless").mean()) if not suppress_high.empty else np.nan,
                "retained_fragile_high_upside_count": int(((retained["fragile_subtype"] == "fragile_high_upside") & retained["fragile_trade"].fillna(False)).sum()),
                "removed_fragile_high_upside_count": int((removed["fragile_subtype"] == "fragile_high_upside").sum()),
                "removed_fragile_hopeless_count": int((removed["fragile_subtype"] == "fragile_hopeless").sum()),
                "removed_fragile_mixed_count": int((removed["fragile_subtype"] == "fragile_mixed").sum()),
            }
        )
        impact_rows.append(
            {
                "variant": variant,
                "removed_trades": int(len(removed)),
                "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()),
                "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()),
                "removed_fragile_mixed": int((removed["fragile_subtype"] == "fragile_mixed").sum()),
                "newly_removed_vs_prior": int((removed["entry_time"].isin(variant_results["prior_preserve_first_variant"]["filtered_timestamps"]) == False).sum()) if variant not in ["control", "prior_preserve_first_variant"] else 0,
            }
        )
    purity_df = pd.DataFrame(purity_rows)
    impact_df = pd.DataFrame(impact_rows)
    purity_df.to_csv(root / "preserve_refinement_purity_table.csv", index=False)
    impact_df.to_csv(root / "preserve_refinement_trade_impact.csv", index=False)
    return purity_df, impact_df


def build_variant_outputs(root: Path, control_audit: pd.DataFrame, scored: pd.DataFrame, variant_results: dict[str, dict], annualization_hours: int) -> pd.DataFrame:
    scored_lookup = scored[scored["baseline_trade"].fillna(False)].copy()
    rows = []
    for variant in VARIANT_ORDER:
        trades = control_audit.copy() if variant == "control" else parse_times(variant_results[variant]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        filtered_ts = set() if variant == "control" else variant_results[variant]["filtered_timestamps"]
        rows.append(summarize_variant(variant, trades, scored_lookup, filtered_ts, annualization_hours))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(root / "variant_summary.csv", index=False)
    return summary_df


def build_robustness_table(root: Path, scored: pd.DataFrame, control_audit: pd.DataFrame, variant_results: dict[str, dict], annualization_hours: int, best_variant: str) -> pd.DataFrame:
    control_trades = parse_times(control_audit.copy(), ["entry_time", "exit_time"])
    prior_trades = parse_times(variant_results["prior_preserve_first_variant"]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
    best_trades = parse_times(variant_results[best_variant]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
    rows = []
    for split_name in sorted(scored.loc[scored["baseline_trade"].fillna(False), "year"].unique()):
        times = set(scored.loc[(scored["baseline_trade"].fillna(False)) & (scored["year"] == split_name), "entry_time"])
        for variant, trades in [("control", control_trades), ("prior_preserve_first_variant", prior_trades), (best_variant, best_trades)]:
            slice_df = trades[trades["entry_time"].isin(times)].copy()
            rows.append({"split_type": "year", "split_label": str(split_name), "variant": variant, **summarize_returns(slice_df, annualization_hours)})
    for split_name in ["early_half", "late_half"]:
        times = set(scored.loc[(scored["baseline_trade"].fillna(False)) & (scored["sample_half"] == split_name), "entry_time"])
        for variant, trades in [("control", control_trades), ("prior_preserve_first_variant", prior_trades), (best_variant, best_trades)]:
            slice_df = trades[trades["entry_time"].isin(times)].copy()
            rows.append({"split_type": "half", "split_label": split_name, "variant": variant, **summarize_returns(slice_df, annualization_hours)})
    for split_name in sorted(scored.loc[scored["baseline_trade"].fillna(False), "phase_context"].dropna().unique()):
        times = set(scored.loc[(scored["baseline_trade"].fillna(False)) & (scored["phase_context"] == split_name), "entry_time"])
        for variant, trades in [("control", control_trades), ("prior_preserve_first_variant", prior_trades), (best_variant, best_trades)]:
            slice_df = trades[trades["entry_time"].isin(times)].copy()
            rows.append({"split_type": "phase_context", "split_label": split_name, "variant": variant, **summarize_returns(slice_df, annualization_hours)})
    robustness = pd.DataFrame(rows)
    robustness.to_csv(root / "preserve_refinement_robustness.csv", index=False)
    return robustness

def write_intuitive_performance(root: Path, summary_df: pd.DataFrame, best_variant: str) -> None:
    subset = summary_df[summary_df["variant"].isin(["control", "prior_preserve_first_variant", best_variant])].copy()
    subset.to_csv(root / "intuitive_performance_summary.csv", index=False)
    control = subset.loc[subset["variant"] == "control"].iloc[0]
    prior = subset.loc[subset["variant"] == "prior_preserve_first_variant"].iloc[0]
    best = subset.loc[subset["variant"] == best_variant].iloc[0]
    lines = [
        "# Intuitive Performance Comparison",
        "",
        f"- Prior preserve-first changed total return from {control['total_return']:.4f} to {prior['total_return']:.4f}.",
        f"- The best refined preserve variant changed total return from {control['total_return']:.4f} to {best['total_return']:.4f}.",
        f"- Sharpe changed from {control['sharpe']:.3f} to {best['sharpe']:.3f}.",
        f"- Max drawdown changed from {control['max_drawdown']:.4f} to {best['max_drawdown']:.4f}.",
        f"- Trade count changed from {int(control['trades'])} to {int(best['trades'])}.",
        f"- Win rate changed from {control['win_rate']:.3f} to {best['win_rate']:.3f}.",
        f"- Avg return per trade changed from {control['avg_return_per_trade']:.5f} to {best['avg_return_per_trade']:.5f}.",
        f"- price_stop count changed from {int(control['price_stop_count'])} to {int(best['price_stop_count'])}.",
        f"- time_stop_max count changed from {int(control['time_stop_max_count'])} to {int(best['time_stop_max_count'])}.",
        f"- time_stop_max contribution changed from {control['time_stop_max_contribution']:.4f} to {best['time_stop_max_contribution']:.4f}.",
        f"- The best refined variant removed {int(best['removed_fragile_hopeless'])} hopeless trades, {int(best['removed_fragile_high_upside'])} high-upside fragile trades, and {int(best['removed_fragile_mixed'])} mixed fragile trades.",
    ]
    write_md(root / "INTUITIVE_PERFORMANCE_COMPARISON.md", lines)


def plot_curves(root: Path, curves: dict[str, pd.DataFrame], best_variant: str) -> None:
    curve_variants = ["control", "prior_preserve_first_variant", best_variant]
    fig = go.Figure()
    for variant in curve_variants:
        curve = curves[variant]
        fig.add_trace(go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name=variant))
    fig.update_layout(title="Equity Curve Comparison", template="plotly_white", xaxis_title="Exit Time", yaxis_title="Equity")
    fig.write_html(str(root / "control_vs_preserve_variants_equity.html"), include_plotlyjs="cdn")

    fig_dd = go.Figure()
    for variant in curve_variants:
        curve = curves[variant]
        fig_dd.add_trace(go.Scatter(x=curve["timestamp"], y=curve["drawdown"], mode="lines", name=variant))
    fig_dd.update_layout(title="Drawdown Curve Comparison", template="plotly_white", xaxis_title="Exit Time", yaxis_title="Drawdown")
    fig_dd.write_html(str(root / "control_vs_preserve_variants_drawdown.html"), include_plotlyjs="cdn")

    fig_png, ax = plt.subplots(figsize=(12, 6))
    for variant in curve_variants:
        curve = curves[variant]
        ax.plot(curve["timestamp"], curve["equity"], label=variant)
    ax.set_title("Equity Curve Comparison")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig_png.autofmt_xdate()
    fig_png.tight_layout()
    fig_png.savefig(root / "control_vs_preserve_variants_equity.png", dpi=130)
    plt.close(fig_png)

    fig_dd_png, ax = plt.subplots(figsize=(12, 6))
    for variant in curve_variants:
        curve = curves[variant]
        ax.plot(curve["timestamp"], curve["drawdown"], label=variant)
    ax.set_title("Drawdown Curve Comparison")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig_dd_png.autofmt_xdate()
    fig_dd_png.tight_layout()
    fig_dd_png.savefig(root / "control_vs_preserve_variants_drawdown.png", dpi=130)
    plt.close(fig_dd_png)


def plot_bar(root: Path, summary_df: pd.DataFrame, column: str, filename: str, title: str) -> None:
    ordered = summary_df.set_index("variant").loc[VARIANT_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(ordered["variant"], ordered[column])
    ax.set_title(title)
    ax.set_ylabel(column)
    ax.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(root / filename, dpi=130)
    plt.close(fig)


def build_changed_trade_review(root: Path, control_result: dict, scored: pd.DataFrame, best_variant: str, variant_results: dict[str, dict]) -> None:
    removed_ts = set(variant_results[best_variant]["filtered_timestamps"])
    fragile = scored[scored["fragile_trade"].fillna(False)].copy()
    removed = fragile[fragile["entry_time"].isin(removed_ts)].copy()
    retained = fragile[~fragile["entry_time"].isin(removed_ts)].copy()

    sample = pd.concat([removed, retained.head(20)], ignore_index=True)
    if sample.empty:
        build_review_chart(control_result["merged"], sample, "Refined Preserve Removed vs Retained", root / "refined_preserve_removed_vs_retained_review.html", root / "refined_preserve_removed_vs_retained_review.png")
        return

    bars = control_result["merged"].copy()
    start = pd.to_datetime(sample["entry_time"], utc=True).min() - pd.Timedelta(hours=24)
    end = pd.to_datetime(sample["exit_time"], utc=True).max() + pd.Timedelta(hours=24)
    window = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["box_upper_edge"], mode="lines", name="Box Upper", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["box_lower_edge"], mode="lines", name="Box Lower", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=removed["entry_time"], y=removed["price"], mode="markers", name="Removed", marker=dict(size=9, color="#e63946", symbol="x")))
    fig.add_trace(go.Scatter(x=retained["entry_time"], y=retained["price"], mode="markers", name="Retained", marker=dict(size=8, color="#1d3557", symbol="diamond")))
    fig.update_layout(title="Refined Preserve Removed vs Retained", template="plotly_white", xaxis_rangeslider_visible=False)
    fig.write_html(str(root / "refined_preserve_removed_vs_retained_review.html"), include_plotlyjs="cdn")

    fig2, ax = plt.subplots(figsize=(13, 6))
    ax.plot(window["timestamp"], window["close"], label="Close")
    ax.plot(window["timestamp"], window["box_upper_edge"], label="Box Upper", linewidth=0.8)
    ax.plot(window["timestamp"], window["box_lower_edge"], label="Box Lower", linewidth=0.8)
    ax.scatter(pd.to_datetime(removed["entry_time"], utc=True), removed["price"], label="Removed", color="#e63946", marker="x")
    ax.scatter(pd.to_datetime(retained["entry_time"], utc=True), retained["price"], label="Retained", color="#1d3557", marker="D", s=20)
    ax.set_title("Refined Preserve Removed vs Retained")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(root / "refined_preserve_removed_vs_retained_review.png", dpi=130)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated preserve refinement research.")
    parser.add_argument("--outdir", default="outputs/preserve_refinement_round")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(FROZEN_LAYER12_CONTROL_CONFIG, root / "frozen_control_config.yaml")

    control_result, control_audit, scored, cfg = build_scored_candidate(root)
    scored = build_refinement_flags(scored)
    annualization_hours = int(cfg["backtest"]["annualization_hours"])

    variant_results = {"control": {"filtered_timestamps": set(), "backtest": control_result["backtest"]}}
    for variant in VARIANT_ORDER[1:]:
        variant_results[variant] = run_variant(control_result, scored, variant)

    summary_df = build_variant_outputs(root, control_audit, scored, variant_results, annualization_hours)
    purity_df, impact_df = build_identity_and_purity_tables(root, scored, variant_results)
    _ = impact_df

    refined_only = summary_df[summary_df["variant"].isin(["refined_preserve_variant_1", "refined_preserve_variant_2", "best_effort_refined_preserve_variant"])].copy()
    best_variant = str(refined_only.sort_values(["total_return", "sharpe"], ascending=[False, False]).iloc[0]["variant"])
    robustness = build_robustness_table(root, scored, control_audit, variant_results, annualization_hours, best_variant)

    write_intuitive_performance(root, summary_df, best_variant)

    curves = {}
    for variant in ["control"] + VARIANT_ORDER[1:]:
        trades = control_audit.copy() if variant == "control" else parse_times(variant_results[variant]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        curves[variant] = compute_equity_drawdown(trades)
    plot_curves(root, curves, best_variant)
    plot_bar(root, summary_df, "total_return", "total_return_comparison.png", "Total Return Comparison")
    plot_bar(root, summary_df, "max_drawdown", "max_drawdown_comparison.png", "Max Drawdown Comparison")
    plot_bar(root, summary_df, "trades", "trade_count_comparison.png", "Trade Count Comparison")
    plot_bar(root, summary_df, "win_rate", "win_rate_comparison.png", "Win Rate Comparison")
    plot_bar(root, summary_df, "price_stop_count", "price_stop_count_comparison.png", "Price Stop Count Comparison")

    build_changed_trade_review(root, control_result, scored, best_variant, variant_results)

    prior_purity = purity_df[purity_df["variant"] == "prior_preserve_first_variant"].iloc[0]
    best_purity = purity_df[purity_df["variant"] == best_variant].iloc[0]
    write_md(
        root / "PRESERVE_REFINEMENT_PURITY_AUDIT.md",
        [
            "# Preserve Refinement Purity Audit",
            "",
            "## Observation",
            f"- prior preserve-high oracle high-upside rate={prior_purity['preserve_high_oracle_high_upside_rate']:.3f}; best refined={best_purity['preserve_high_oracle_high_upside_rate']:.3f}.",
            f"- prior preserve-high hopeless contamination={prior_purity['preserve_high_hopeless_contamination']:.3f}; best refined={best_purity['preserve_high_hopeless_contamination']:.3f}.",
            f"- best refined removed hopeless={int(best_purity['removed_fragile_hopeless_count'])}, high-upside={int(best_purity['removed_fragile_high_upside_count'])}, mixed={int(best_purity['removed_fragile_mixed_count'])}.",
            "",
            "## Conclusion",
            "- Preserve-side refinement is only useful if purity rises without starting to remove fragile high-upside trades.",
        ],
    )
    write_md(
        root / "PRESERVE_REFINEMENT_SPLIT_SUMMARY.md",
        [
            "# Preserve Refinement Split Summary",
            "",
            "## Observation",
            f"- best refined variant: {best_variant}.",
            f"- robustness rows: {int(len(robustness))}.",
            f"- yearly split count: {int((robustness['split_type'] == 'year').sum())}.",
            "",
            "## Conclusion",
            "- The best refined preserve rule still needs split stability, not just a better full-sample number.",
        ],
    )

    control_row = summary_df[summary_df["variant"] == "control"].iloc[0]
    prior_row = summary_df[summary_df["variant"] == "prior_preserve_first_variant"].iloc[0]
    best_row = summary_df[summary_df["variant"] == best_variant].iloc[0]
    write_md(
        root / "PRESERVE_REFINEMENT_ROUND_SUMMARY.md",
        [
            "# Preserve Refinement Round Summary",
            "",
            "## Final Answers",
            f"1. Preserve-high purity improved versus the prior preserve-first round: {'yes' if best_purity['preserve_high_oracle_high_upside_rate'] > prior_purity['preserve_high_oracle_high_upside_rate'] else 'no'}.",
            f"2. Hopeless contamination declined: {'yes' if best_purity['preserve_high_hopeless_contamination'] < prior_purity['preserve_high_hopeless_contamination'] else 'no'}.",
            f"3. Refined preserve variant improved total return: {'yes' if best_row['total_return'] > prior_row['total_return'] else 'no'}.",
            f"4. Max drawdown changed from {prior_row['max_drawdown']:.4f} to {best_row['max_drawdown']:.4f}.",
            f"5. Fragile high-upside losses stayed near zero: {'yes' if int(best_row['removed_fragile_high_upside']) == 0 else 'no'}.",
            "6. The result is shown directly in the performance table and comparison charts, not only in purity stats.",
            f"7. Strong enough for a future promotion-readiness round: {'yes' if best_row['total_return'] > control_row['total_return'] and best_purity['preserve_high_oracle_high_upside_rate'] >= 0.40 else 'not yet'}.",
            "8. If not, the line should continue only if one more narrow preserve refinement remains interpretable; otherwise it should stop.",
            "",
            "## Observation",
            f"- control total_return={control_row['total_return']:.4f}; prior preserve-first={prior_row['total_return']:.4f}; best refined={best_row['total_return']:.4f}.",
            f"- prior preserve-high purity={prior_purity['preserve_high_oracle_high_upside_rate']:.3f}; best refined purity={best_purity['preserve_high_oracle_high_upside_rate']:.3f}.",
            f"- prior hopeless contamination={prior_purity['preserve_high_hopeless_contamination']:.3f}; best refined contamination={best_purity['preserve_high_hopeless_contamination']:.3f}.",
            f"- best refined removed hopeless={int(best_row['removed_fragile_hopeless'])}, high-upside={int(best_row['removed_fragile_high_upside'])}, mixed={int(best_row['removed_fragile_mixed'])}.",
        ],
    )


if __name__ == "__main__":
    main()


