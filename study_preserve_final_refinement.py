from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from layer12_structural_research_helpers import (
    FROZEN_LAYER12_CONTROL_CONFIG,
    parse_times,
    save_yaml,
    summarize_returns,
    winner_sensitivity,
)
from study_preserve_refinement_round import (
    build_refinement_flags,
    build_scored_candidate,
    compute_equity_drawdown,
    run_variant,
)
from study_preserve_suppress_ranking import write_md


ALIAS_TO_INTERNAL = {
    "control": "control",
    "prior_preserve_first": "prior_preserve_first_variant",
    "refined_preserve_variant": "refined_preserve_variant_1",
    "experimental_narrow_candidate": "refined_preserve_variant_2",
}

VARIANT_ORDER = list(ALIAS_TO_INTERNAL.keys())


def guard_col_for_alias(alias: str) -> str:
    return {
        "control": "guard_prior_preserve_high",
        "prior_preserve_first": "guard_prior_preserve_high",
        "refined_preserve_variant": "guard_refined_preserve_1",
        "experimental_narrow_candidate": "guard_refined_preserve_2",
    }[alias]


def mask_for_alias(scored: pd.DataFrame, alias: str) -> pd.Series:
    internal = ALIAS_TO_INTERNAL[alias]
    if internal == "control":
        return pd.Series(False, index=scored.index)
    if internal == "prior_preserve_first_variant":
        return scored["flag_prior_remove"].fillna(False)
    if internal == "refined_preserve_variant_1":
        return (scored["flag_prior_remove"] | scored["flag_weak_preserve_v1"]).fillna(False)
    if internal == "refined_preserve_variant_2":
        return (scored["flag_prior_remove"] | scored["flag_weak_preserve_v2"]).fillna(False)
    raise ValueError(f"Unknown alias: {alias}")


def build_variant_results(control_result: dict, scored: pd.DataFrame) -> dict[str, dict]:
    results = {"control": {"filtered_timestamps": set(), "backtest": control_result["backtest"]}}
    for alias, internal in ALIAS_TO_INTERNAL.items():
        if alias == "control":
            continue
        run = run_variant(control_result, scored, internal)
        results[alias] = {
            "variant": alias,
            "internal_variant": internal,
            "filtered_timestamps": run["filtered_timestamps"],
            "backtest": run["backtest"],
        }
    return results


def summarize_variant(alias: str, trades: pd.DataFrame, scored_lookup: pd.DataFrame, filtered_ts: set[pd.Timestamp], annualization_hours: int) -> dict[str, object]:
    trades = parse_times(trades.copy(), ["entry_time", "exit_time"])
    summary = summarize_returns(trades, annualization_hours)
    removed = scored_lookup[scored_lookup["entry_time"].isin(filtered_ts)].copy()
    retained = scored_lookup[~scored_lookup["entry_time"].isin(filtered_ts)].copy()
    preserve_high = retained[(retained["fragile_trade"].fillna(False)) & retained[guard_col_for_alias(alias)].fillna(False)]
    time_stop_max_mask = trades["exit_reason"] == "time_stop_max" if not trades.empty else pd.Series(dtype=bool)
    return {
        "variant": alias,
        **summary,
        "avg_return_per_trade": float(summary["avg_return"]),
        "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
        "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
        "time_stop_max_contribution": float(trades.loc[time_stop_max_mask, "net_return"].sum()) if not trades.empty else 0.0,
        "removed_trades": int(len(removed)),
        "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()),
        "removed_fragile_mixed": int((removed["fragile_subtype"] == "fragile_mixed").sum()),
        "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()),
        "preserve_high_trades": int(len(preserve_high)),
        "preserve_high_oracle_high_upside_rate": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
        "preserve_high_hopeless_contamination": float((preserve_high["fragile_focus_label"] == "hopeless").mean()) if not preserve_high.empty else np.nan,
        "preserve_high_mixed_rate": float((preserve_high["fragile_subtype"] == "fragile_mixed").mean()) if not preserve_high.empty else np.nan,
    }


def build_variant_summary(root: Path, control_audit: pd.DataFrame, scored: pd.DataFrame, results: dict[str, dict], annualization_hours: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    scored_lookup = scored[scored["baseline_trade"].fillna(False)].copy()
    curves: dict[str, pd.DataFrame] = {}
    rows = []
    for alias in VARIANT_ORDER:
        trades = control_audit.copy() if alias == "control" else parse_times(results[alias]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        filtered_ts = set() if alias == "control" else results[alias]["filtered_timestamps"]
        rows.append(summarize_variant(alias, trades, scored_lookup, filtered_ts, annualization_hours))
        curves[alias] = compute_equity_drawdown(trades)
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "variant_summary.csv", index=False)
    return summary, curves


def build_purity_table(root: Path, scored: pd.DataFrame, results: dict[str, dict]) -> pd.DataFrame:
    base = scored[scored["baseline_trade"].fillna(False)].copy()
    rows = []
    for alias in VARIANT_ORDER:
        filtered_ts = set() if alias == "control" else results[alias]["filtered_timestamps"]
        removed = base[base["entry_time"].isin(filtered_ts)].copy()
        retained = base[~base["entry_time"].isin(filtered_ts)].copy()
        preserve_high = retained[(retained["fragile_trade"].fillna(False)) & retained[guard_col_for_alias(alias)].fillna(False)]
        rows.append(
            {
                "variant": alias,
                "preserve_high_trades": int(len(preserve_high)),
                "oracle_high_upside_rate": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
                "hopeless_contamination_rate": float((preserve_high["fragile_focus_label"] == "hopeless").mean()) if not preserve_high.empty else np.nan,
                "mixed_rate": float((preserve_high["fragile_subtype"] == "fragile_mixed").mean()) if not preserve_high.empty else np.nan,
                "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()),
                "removed_fragile_mixed": int((removed["fragile_subtype"] == "fragile_mixed").sum()),
                "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()),
            }
        )
    purity = pd.DataFrame(rows)
    purity.to_csv(root / "preserve_purity_table.csv", index=False)
    return purity


def build_removed_trade_identity(root: Path, scored: pd.DataFrame, results: dict[str, dict]) -> pd.DataFrame:
    base = scored[scored["baseline_trade"].fillna(False)].copy()
    for alias in VARIANT_ORDER[1:]:
        base[f"removed_{alias}"] = base["entry_time"].isin(results[alias]["filtered_timestamps"])
    base["newly_removed_vs_prior_preserve_first"] = base["removed_refined_preserve_variant"] & ~base["removed_prior_preserve_first"] if False else False
    base["newly_removed_by_refined_vs_prior"] = base["removed_refined_preserve_variant"] & ~base["removed_prior_preserve_first"]
    base["newly_removed_by_experimental_vs_prior"] = base["removed_experimental_narrow_candidate"] & ~base["removed_prior_preserve_first"]
    cols = [
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_return",
        "fragile_subtype",
        "fragile_focus_label",
        "preserve_bucket",
        "suppress_bucket",
        "removed_prior_preserve_first",
        "removed_refined_preserve_variant",
        "removed_experimental_narrow_candidate",
        "newly_removed_by_refined_vs_prior",
        "newly_removed_by_experimental_vs_prior",
    ]
    identity = base[[c for c in cols if c in base.columns]].copy()
    identity.to_csv(root / "removed_trade_identity.csv", index=False)
    return identity


def summarize_split(trades: pd.DataFrame, annualization_hours: int) -> dict[str, float]:
    summary = summarize_returns(trades, annualization_hours)
    return {
        **summary,
        "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
        "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
        "avg_return_per_trade": float(summary["avg_return"]),
    }


def build_robustness_check(root: Path, scored: pd.DataFrame, control_audit: pd.DataFrame, results: dict[str, dict], annualization_hours: int) -> pd.DataFrame:
    base = scored[scored["baseline_trade"].fillna(False)].copy()
    trade_frames = {
        "control": parse_times(control_audit.copy(), ["entry_time", "exit_time"]),
        "prior_preserve_first": parse_times(results["prior_preserve_first"]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"]),
        "refined_preserve_variant": parse_times(results["refined_preserve_variant"]["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"]),
    }
    rows = []
    for label in ["early_half", "late_half"]:
        times = set(base.loc[base["sample_half"] == label, "entry_time"])
        for alias, trades in trade_frames.items():
            slice_df = trades[trades["entry_time"].isin(times)].copy()
            rows.append({"split_type": "half", "split_label": label, "variant": alias, **summarize_split(slice_df, annualization_hours)})
    for year in sorted(base["year"].dropna().unique()):
        times = set(base.loc[base["year"] == year, "entry_time"])
        for alias, trades in trade_frames.items():
            slice_df = trades[trades["entry_time"].isin(times)].copy()
            rows.append({"split_type": "year", "split_label": str(int(year)), "variant": alias, **summarize_split(slice_df, annualization_hours)})
    for alias, trades in trade_frames.items():
        ws = winner_sensitivity(trades, annualization_hours)
        ws = ws[ws["remove_top_winners"].isin([5, 10])].copy()
        for row in ws.itertuples(index=False):
            rows.append(
                {
                    "split_type": "winner_removal",
                    "split_label": f"remove_top_{int(row.remove_top_winners)}",
                    "variant": alias,
                    "trades": int(row.trades),
                    "win_rate": float(row.win_rate),
                    "avg_return": float(row.avg_return),
                    "total_return": float(row.total_return),
                    "sharpe": float(row.sharpe),
                    "max_drawdown": float(row.max_drawdown),
                    "avg_return_per_trade": float(row.avg_return),
                    "price_stop_count": np.nan,
                    "time_stop_max_count": np.nan,
                }
            )
    robustness = pd.DataFrame(rows)
    robustness.to_csv(root / "robustness_check.csv", index=False)
    return robustness


def plot_curves(root: Path, curves: dict[str, pd.DataFrame]) -> None:
    fig = go.Figure()
    for alias in ["control", "prior_preserve_first", "refined_preserve_variant"]:
        curve = curves[alias]
        fig.add_trace(go.Scatter(x=curve["timestamp"], y=curve["equity"], mode="lines", name=alias))
    fig.update_layout(title="Equity Curve Comparison", template="plotly_white", xaxis_title="Exit Time", yaxis_title="Equity")
    fig.write_html(str(root / "equity_comparison.html"), include_plotlyjs="cdn")

    fig_dd = go.Figure()
    for alias in ["control", "prior_preserve_first", "refined_preserve_variant"]:
        curve = curves[alias]
        fig_dd.add_trace(go.Scatter(x=curve["timestamp"], y=curve["drawdown"], mode="lines", name=alias))
    fig_dd.update_layout(title="Drawdown Curve Comparison", template="plotly_white", xaxis_title="Exit Time", yaxis_title="Drawdown")
    fig_dd.write_html(str(root / "drawdown_comparison.html"), include_plotlyjs="cdn")

    fig_png, ax = plt.subplots(figsize=(12, 6))
    for alias in ["control", "prior_preserve_first", "refined_preserve_variant"]:
        curve = curves[alias]
        ax.plot(curve["timestamp"], curve["equity"], label=alias)
    ax.set_title("Equity Curve Comparison")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig_png.autofmt_xdate()
    fig_png.tight_layout()
    fig_png.savefig(root / "equity_comparison.png", dpi=130)
    plt.close(fig_png)

    fig_dd_png, ax = plt.subplots(figsize=(12, 6))
    for alias in ["control", "prior_preserve_first", "refined_preserve_variant"]:
        curve = curves[alias]
        ax.plot(curve["timestamp"], curve["drawdown"], label=alias)
    ax.set_title("Drawdown Curve Comparison")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig_dd_png.autofmt_xdate()
    fig_dd_png.tight_layout()
    fig_dd_png.savefig(root / "drawdown_comparison.png", dpi=130)
    plt.close(fig_dd_png)


def plot_bar(root: Path, summary: pd.DataFrame, column: str, filename: str, title: str) -> None:
    ordered = summary.set_index("variant").loc[VARIANT_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ordered["variant"], ordered[column])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(root / filename, dpi=130)
    plt.close(fig)


def write_summary(root: Path, summary: pd.DataFrame, purity: pd.DataFrame, robustness: pd.DataFrame) -> None:
    control = summary[summary["variant"] == "control"].iloc[0]
    prior = summary[summary["variant"] == "prior_preserve_first"].iloc[0]
    refined = summary[summary["variant"] == "refined_preserve_variant"].iloc[0]
    prior_purity = purity[purity["variant"] == "prior_preserve_first"].iloc[0]
    refined_purity = purity[purity["variant"] == "refined_preserve_variant"].iloc[0]

    late = robustness[(robustness["split_type"] == "half") & (robustness["split_label"] == "late_half")]
    late_control = float(late[late["variant"] == "control"]["total_return"].iloc[0])
    late_refined = float(late[late["variant"] == "refined_preserve_variant"]["total_return"].iloc[0])
    years = robustness[robustness["split_type"] == "year"].copy()
    year_wins = int((years.pivot(index="split_label", columns="variant", values="total_return")["refined_preserve_variant"] > years.pivot(index="split_label", columns="variant", values="total_return")["control"]).sum())

    lines = [
        "# Preserve Final Refinement Summary",
        "",
        "## Final Answers",
        f"1. Did preserve-high purity improve? {'Yes' if refined_purity['oracle_high_upside_rate'] > prior_purity['oracle_high_upside_rate'] else 'No'}. Prior={prior_purity['oracle_high_upside_rate']:.3f}, refined={refined_purity['oracle_high_upside_rate']:.3f}.",
        f"2. Did hopeless contamination decline? {'Yes' if refined_purity['hopeless_contamination_rate'] < prior_purity['hopeless_contamination_rate'] else 'No'}. Prior={prior_purity['hopeless_contamination_rate']:.3f}, refined={refined_purity['hopeless_contamination_rate']:.3f}.",
        f"3. Did fragile_high_upside remain preserved? {'Yes' if int(refined_purity['removed_fragile_high_upside']) == 0 else 'No'}. Removed fragile_high_upside={int(refined_purity['removed_fragile_high_upside'])}.",
        f"4. Is improvement stable across splits? {'Yes, modestly' if late_refined >= late_control and year_wins >= 4 else 'No, it still looks split-local'}. Late-half total return moved from {late_control:.4f} to {late_refined:.4f}; refined beat control in {year_wins} yearly slices.",
        f"5. Is result strong enough for promotion-readiness audit? {'No' if refined_purity['oracle_high_upside_rate'] < 0.40 else 'Possibly'}. The full-sample result improved from {control['total_return']:.4f} to {refined['total_return']:.4f}, but preserve-high purity only reached {refined_purity['oracle_high_upside_rate']:.3f}.",
        "",
        "## Plain-Language Comparison",
        f"- Control total return was {control['total_return']:.4f}. Prior preserve-first was {prior['total_return']:.4f}. Refined preserve variant was {refined['total_return']:.4f}.",
        f"- Sharpe moved from {control['sharpe']:.3f} to {refined['sharpe']:.3f}.",
        f"- Max drawdown moved from {control['max_drawdown']:.4f} to {refined['max_drawdown']:.4f}.",
        f"- Trade count moved from {int(control['trades'])} to {int(refined['trades'])}.",
        f"- price_stop count moved from {int(control['price_stop_count'])} to {int(refined['price_stop_count'])}.",
        f"- time_stop_max count moved from {int(control['time_stop_max_count'])} to {int(refined['time_stop_max_count'])}.",
        f"- The refined preserve variant removed {int(refined['removed_fragile_hopeless'])} hopeless trades, {int(refined['removed_fragile_mixed'])} mixed trades, and {int(refined['removed_fragile_high_upside'])} fragile high-upside trades.",
        "",
        "## Recommendation",
        "- Do not move to promotion-readiness yet.",
        "- Stop this optimization line here unless there is a strong reason to spend one more cycle on preserve-side purity. The result improved, but the preserve side is still not clean enough to justify a promotion audit.",
    ]
    write_md(root / "PRESERVE_FINAL_REFINEMENT_SUMMARY.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final isolated preserve refinement round.")
    parser.add_argument("--outdir", default="outputs/preserve_final_refinement_round")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(FROZEN_LAYER12_CONTROL_CONFIG, root / "frozen_control_config.yaml")

    control_result, control_audit, scored, cfg = build_scored_candidate(root)
    scored = build_refinement_flags(scored)
    annualization_hours = int(cfg["backtest"]["annualization_hours"])

    results = build_variant_results(control_result, scored)
    summary, curves = build_variant_summary(root, control_audit, scored, results, annualization_hours)
    purity = build_purity_table(root, scored, results)
    _ = build_removed_trade_identity(root, scored, results)
    robustness = build_robustness_check(root, scored, control_audit, results, annualization_hours)

    plot_curves(root, curves)
    plot_bar(root, summary, "total_return", "total_return_comparison.png", "Total Return Comparison")
    plot_bar(root, summary, "max_drawdown", "max_drawdown_comparison.png", "Max Drawdown Comparison")
    plot_bar(root, summary, "trades", "trade_count_comparison.png", "Trade Count Comparison")
    plot_bar(root, summary, "price_stop_count", "price_stop_comparison.png", "Price Stop Comparison")

    write_summary(root, summary, purity, robustness)


if __name__ == "__main__":
    main()
