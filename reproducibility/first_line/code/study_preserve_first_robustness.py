from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    winner_sensitivity,
)
from study_entry_edge_optimization import (
    apply_entry_quality_score,
    build_execution_candidate_table,
    build_review_chart,
    build_score_spec,
)
from study_preserve_suppress_ranking import (
    CONTROL_VARIANT,
    add_interaction_features,
    apply_dual_score,
    build_dual_score_spec,
    run_variant,
    write_md,
)
from study_fragile_label_refinement import assign_fragile_subtypes


def add_context_splits(scored: pd.DataFrame, merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = scored.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out["year"] = out["entry_time"].dt.year.astype(int)

    baseline_times = out.loc[out["baseline_trade"].fillna(False), "entry_time"].sort_values()
    median_time = baseline_times.iloc[len(baseline_times) // 2]
    out["sample_half"] = np.where(out["entry_time"] <= median_time, "early_half", "late_half")

    merged_local = merged[["timestamp", "close"]].copy()
    merged_local["timestamp"] = pd.to_datetime(merged_local["timestamp"], utc=True)
    horizon = 180 * 24
    merged_local["long_context_ret_180d"] = merged_local["close"] / merged_local["close"].shift(horizon) - 1.0
    out = out.merge(merged_local[["timestamp", "long_context_ret_180d"]], left_on="entry_time", right_on="timestamp", how="left")
    out = out.drop(columns=["timestamp_y"], errors="ignore")
    out = out.rename(columns={"timestamp_x": "timestamp"})

    baseline_long = pd.to_numeric(out.loc[out["baseline_trade"].fillna(False), "long_context_ret_180d"], errors="coerce").dropna()
    if len(baseline_long) >= 3:
        q1, q2 = baseline_long.quantile([1 / 3, 2 / 3]).tolist()
    else:
        q1, q2 = -0.10, 0.10
    out["phase_context"] = np.select(
        [out["long_context_ret_180d"] <= q1, out["long_context_ret_180d"] >= q2],
        ["bearish_context", "bullish_context"],
        default="neutral_context",
    )

    metadata = pd.DataFrame(
        [
            {"name": "median_entry_time", "value": str(median_time)},
            {"name": "phase_q1", "value": float(q1)},
            {"name": "phase_q2", "value": float(q2)},
        ]
    )
    return out, metadata


def build_time_split_definitions(baseline_trades: pd.DataFrame) -> list[dict[str, object]]:
    entries = pd.to_datetime(baseline_trades["entry_time"], utc=True).sort_values()
    start = entries.min().normalize()
    end = entries.max().normalize() + pd.Timedelta(days=1)
    median_time = entries.iloc[len(entries) // 2]

    splits = [
        {"split_type": "full", "split_label": "full_sample", "start": start, "end": end},
        {"split_type": "half", "split_label": "early_half", "start": start, "end": median_time + pd.Timedelta(seconds=1)},
        {"split_type": "half", "split_label": "late_half", "start": median_time + pd.Timedelta(seconds=1), "end": end},
    ]

    rolling_start = pd.Timestamp(year=start.year, month=1, day=1, tz="UTC")
    while rolling_start + pd.DateOffset(months=36) <= end:
        rolling_end = rolling_start + pd.DateOffset(months=36)
        label = f"{rolling_start.year}-{(rolling_end - pd.Timedelta(days=1)).year}_36m"
        splits.append({"split_type": "rolling_36m", "split_label": label, "start": rolling_start, "end": rolling_end})
        rolling_start = rolling_start + pd.DateOffset(months=12)
    return splits


def _trade_metrics(trades: pd.DataFrame, annualization_hours: int) -> dict[str, float]:
    summary = summarize_returns(trades, annualization_hours)
    summary.update(
        {
            "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
            "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
            "state_stop_count": int((trades["exit_reason"] == "state_stop").sum()) if not trades.empty else 0,
            "avg_return_per_trade": float(trades["net_return"].mean()) if not trades.empty else 0.0,
        }
    )
    return summary


def summarize_time_window(
    split_type: str,
    split_label: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    control_trades: pd.DataFrame,
    preserve_trades: pd.DataFrame,
    scored: pd.DataFrame,
    preserve_filtered_ts: set[pd.Timestamp],
    annualization_hours: int,
) -> list[dict[str, object]]:
    control_slice = control_trades[(control_trades["entry_time"] >= start) & (control_trades["entry_time"] < end)].copy()
    preserve_slice = preserve_trades[(preserve_trades["entry_time"] >= start) & (preserve_trades["entry_time"] < end)].copy()
    scored_slice = scored[(scored["entry_time"] >= start) & (scored["entry_time"] < end) & scored["baseline_trade"].fillna(False)].copy()
    removed = control_slice[control_slice["entry_time"].isin(preserve_filtered_ts)].copy()
    removed = removed.merge(scored_slice[["entry_time", "fragile_subtype"]], on="entry_time", how="left")

    preserve_high = scored_slice[(scored_slice["fragile_trade"].fillna(False)) & (scored_slice["preserve_bucket"].astype(str) == "high")]
    suppress_high = scored_slice[(scored_slice["fragile_trade"].fillna(False)) & (scored_slice["suppress_bucket"].astype(str) == "high")]

    rows = []
    for variant_name, trades in [("control", control_slice), ("preserve_first_variant", preserve_slice)]:
        metrics = _trade_metrics(trades, annualization_hours)
        rows.append(
            {
                "split_type": split_type,
                "split_label": split_label,
                "variant": variant_name,
                **metrics,
                "retained_baseline_trades": int(len(control_slice) - len(removed)) if variant_name == "preserve_first_variant" else int(len(control_slice)),
                "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()) if variant_name == "preserve_first_variant" else 0,
                "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()) if variant_name == "preserve_first_variant" else 0,
                "preserve_high_purity": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
                "suppress_high_hopeless_rate": float((suppress_high["fragile_focus_label"] == "hopeless").mean()) if not suppress_high.empty else np.nan,
            }
        )
    return rows

def build_phase_split_table(
    scored: pd.DataFrame,
    control_trades: pd.DataFrame,
    preserve_trades: pd.DataFrame,
    preserve_filtered_ts: set[pd.Timestamp],
    annualization_hours: int,
) -> pd.DataFrame:
    rows = []
    for phase, score_slice in scored[scored["baseline_trade"].fillna(False)].groupby("phase_context", dropna=False):
        trade_times = set(score_slice["entry_time"])
        control_slice = control_trades[control_trades["entry_time"].isin(trade_times)].copy()
        preserve_slice = preserve_trades[preserve_trades["entry_time"].isin(trade_times)].copy()
        removed = control_slice[control_slice["entry_time"].isin(preserve_filtered_ts)].copy()
        removed = removed.merge(score_slice[["entry_time", "fragile_subtype"]], on="entry_time", how="left")
        preserve_high = score_slice[(score_slice["fragile_trade"].fillna(False)) & (score_slice["preserve_bucket"].astype(str) == "high")]
        suppress_high = score_slice[(score_slice["fragile_trade"].fillna(False)) & (score_slice["suppress_bucket"].astype(str) == "high")]
        for variant_name, trades in [("control", control_slice), ("preserve_first_variant", preserve_slice)]:
            metrics = _trade_metrics(trades, annualization_hours)
            rows.append(
                {
                    "split_type": "phase_context",
                    "split_label": phase,
                    "variant": variant_name,
                    **metrics,
                    "retained_baseline_trades": int(len(control_slice) - len(removed)) if variant_name == "preserve_first_variant" else int(len(control_slice)),
                    "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()) if variant_name == "preserve_first_variant" else 0,
                    "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()) if variant_name == "preserve_first_variant" else 0,
                    "preserve_high_purity": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
                    "suppress_high_hopeless_rate": float((suppress_high["fragile_focus_label"] == "hopeless").mean()) if not suppress_high.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_identity_tables(root: Path, scored: pd.DataFrame, preserve_filtered_ts: set[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = scored[scored["baseline_trade"].fillna(False)].copy()
    base["removed_by_preserve_first"] = base["entry_time"].isin(preserve_filtered_ts)
    base["preserved_fragile_like"] = base["fragile_trade"].fillna(False) & (base["preserve_bucket"].astype(str) == "high") & (base["suppress_bucket"].astype(str) != "high")
    base["suppressible_fragile_like"] = base["fragile_trade"].fillna(False) & (base["suppress_bucket"].astype(str) == "high") & (base["preserve_bucket"].astype(str) != "high")
    cols = [
        "entry_time",
        "exit_time",
        "year",
        "sample_half",
        "phase_context",
        "exit_reason",
        "net_return",
        "fragile_subtype",
        "fragile_focus_label",
        "preserve_bucket",
        "suppress_bucket",
        "removed_by_preserve_first",
        "preserved_fragile_like",
        "suppressible_fragile_like",
    ]
    identity = base[[col for col in cols if col in base.columns]].copy()
    identity.to_csv(root / "retained_removed_trade_identity.csv", index=False)

    preserved = identity[identity["preserved_fragile_like"]].copy()
    removed = identity[identity["removed_by_preserve_first"]].copy()

    preserved_rows = []
    removed_rows = []
    for split_col in ["year", "sample_half", "phase_context"]:
        for label, group in preserved.groupby(split_col, dropna=False):
            preserved_rows.append(
                {
                    "split_type": split_col,
                    "split_label": label,
                    "trades": int(len(group)),
                    "high_upside_rate": float((group["fragile_focus_label"] == "high_upside").mean()),
                    "hopeless_rate": float((group["fragile_focus_label"] == "hopeless").mean()),
                    "mixed_rate": float((group["fragile_subtype"] == "fragile_mixed").mean()),
                }
            )
        for label, group in removed.groupby(split_col, dropna=False):
            removed_rows.append(
                {
                    "split_type": split_col,
                    "split_label": label,
                    "trades": int(len(group)),
                    "hopeless_rate": float((group["fragile_focus_label"] == "hopeless").mean()),
                    "high_upside_rate": float((group["fragile_focus_label"] == "high_upside").mean()),
                    "mixed_rate": float((group["fragile_subtype"] == "fragile_mixed").mean()),
                }
            )
    preserved_df = pd.DataFrame(preserved_rows)
    removed_df = pd.DataFrame(removed_rows)
    preserved_df.to_csv(root / "preserved_fragile_identity_stability.csv", index=False)
    removed_df.to_csv(root / "removed_fragile_identity_stability.csv", index=False)
    return identity, preserved_df, removed_df


def build_purity_tables(root: Path, scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fragile = scored[scored["fragile_trade"].fillna(False)].copy()
    preserve_rows = []
    suppress_rows = []
    for split_col in ["year", "sample_half", "phase_context"]:
        for label, group in fragile.groupby(split_col, dropna=False):
            preserve_high = group[group["preserve_bucket"].astype(str) == "high"]
            suppress_high = group[group["suppress_bucket"].astype(str) == "high"]
            preserve_rows.append(
                {
                    "split_type": split_col,
                    "split_label": label,
                    "trades": int(len(preserve_high)),
                    "oracle_high_upside_rate": float((preserve_high["fragile_focus_label"] == "high_upside").mean()) if not preserve_high.empty else np.nan,
                    "hopeless_contamination_rate": float((preserve_high["fragile_focus_label"] == "hopeless").mean()) if not preserve_high.empty else np.nan,
                    "mixed_rate": float((preserve_high["fragile_subtype"] == "fragile_mixed").mean()) if not preserve_high.empty else np.nan,
                    "overlap_with_suppress_high": int((preserve_high["suppress_bucket"].astype(str) == "high").sum()) if not preserve_high.empty else 0,
                }
            )
            suppress_rows.append(
                {
                    "split_type": split_col,
                    "split_label": label,
                    "trades": int(len(suppress_high)),
                    "oracle_hopeless_rate": float((suppress_high["fragile_focus_label"] == "hopeless").mean()) if not suppress_high.empty else np.nan,
                    "high_upside_contamination_rate": float((suppress_high["fragile_focus_label"] == "high_upside").mean()) if not suppress_high.empty else np.nan,
                    "mixed_rate": float((suppress_high["fragile_subtype"] == "fragile_mixed").mean()) if not suppress_high.empty else np.nan,
                    "overlap_with_preserve_high": int((suppress_high["preserve_bucket"].astype(str) == "high").sum()) if not suppress_high.empty else 0,
                }
            )
    preserve_df = pd.DataFrame(preserve_rows)
    suppress_df = pd.DataFrame(suppress_rows)
    preserve_df.to_csv(root / "preserve_purity_by_split.csv", index=False)
    suppress_df.to_csv(root / "suppress_purity_by_split.csv", index=False)
    return preserve_df, suppress_df


def write_reports(
    root: Path,
    time_split: pd.DataFrame,
    year_split: pd.DataFrame,
    phase_split: pd.DataFrame,
    preserve_purity: pd.DataFrame,
    suppress_purity: pd.DataFrame,
    winner_df: pd.DataFrame,
) -> None:
    preserve_rows = time_split[time_split["variant"] == "preserve_first_variant"].copy()
    better_rows = preserve_rows[preserve_rows["total_return"] > time_split.loc[time_split["variant"] == "control", "total_return"].values[: len(preserve_rows)]] if len(preserve_rows) else preserve_rows
    _ = better_rows
    write_md(
        root / "PRESERVE_FIRST_SPLIT_AUDIT.md",
        [
            "# Preserve First Split Audit",
            "",
            "## Observation",
            f"- yearly rows: {int(len(year_split))}.",
            f"- time split rows: {int(len(time_split))}.",
            f"- phase split rows: {int(len(phase_split))}.",
            f"- preserve-first positive total-return delta rows: {int((time_split[time_split['variant'] == 'preserve_first_variant']['total_return'].reset_index(drop=True) > time_split[time_split['variant'] == 'control']['total_return'].reset_index(drop=True)).sum())}." if len(time_split[time_split['variant'] == 'preserve_first_variant']) == len(time_split[time_split['variant'] == 'control']) else "- split delta comparison available in CSV tables.",
            "",
            "## Conclusion",
            "- Robustness is judged by breadth across splits, not by one full-sample improvement.",
            "- Trade-removal stability matters as much as total-return differences for promotion readiness.",
        ],
    )

    write_md(
        root / "PRESERVE_SUPPRESS_PURITY_SUMMARY.md",
        [
            "# Preserve Suppress Purity Summary",
            "",
            "## Observation",
            f"- mean preserve-high oracle high-upside rate: {float(preserve_purity['oracle_high_upside_rate'].mean()):.3f}.",
            f"- mean preserve-high hopeless contamination: {float(preserve_purity['hopeless_contamination_rate'].mean()):.3f}.",
            f"- mean suppress-high oracle hopeless rate: {float(suppress_purity['oracle_hopeless_rate'].mean()):.3f}.",
            f"- mean suppress-high high-upside contamination: {float(suppress_purity['high_upside_contamination_rate'].mean()):.3f}.",
            "",
            "## Conclusion",
            "- Preserve-side purity must improve enough to justify future promotion work; otherwise the line should stop or return to refinement.",
            "- Suppress-side purity is a secondary check; the preserve-first logic still fails if preserve-high remains too impure.",
        ],
    )

    control_ws = winner_df[winner_df["variant"] == "control"].copy()
    preserve_ws = winner_df[winner_df["variant"] == "preserve_first_variant"].copy()
    lines = ["# Winner Dependence Change Note", "", "## Observation"]
    for remove_top in [0, 3, 5, 10]:
        c = control_ws[control_ws["remove_top_winners"] == remove_top]
        p = preserve_ws[preserve_ws["remove_top_winners"] == remove_top]
        if c.empty or p.empty:
            continue
        lines.append(
            f"- remove_top={remove_top}: control_total_return={float(c['total_return'].iloc[0]):.4f}, preserve_first_total_return={float(p['total_return'].iloc[0]):.4f}."
        )
    lines.extend([
        "",
        "## Conclusion",
        "- Winner dependence improves only if preserve-first remains more resilient after top-winner removal, not just on the full curve.",
    ])
    write_md(root / "winner_dependence_change_note.md", lines)

def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated preserve-first robustness research.")
    parser.add_argument("--outdir", default="outputs/preserve_first_robustness_round")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(FROZEN_LAYER12_CONTROL_CONFIG, root / "frozen_control_config.yaml")

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
    scored.to_csv(root / "control_entry_candidates_scored_with_splits.csv", index=False)

    preserve_result = run_variant(control_result, scored, "preserve_first_variant")
    best_effort_result = run_variant(control_result, scored, "best_effort_preserve_suppress_variant")

    control_trades = parse_times(control_result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
    preserve_trades = parse_times(preserve_result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
    best_effort_trades = parse_times(best_effort_result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
    preserve_filtered_ts = set(preserve_result["filtered_timestamps"])

    annualization_hours = int(cfg["backtest"]["annualization_hours"])

    time_rows = []
    for split_def in build_time_split_definitions(control_trades):
        time_rows.extend(
            summarize_time_window(
                split_def["split_type"],
                split_def["split_label"],
                split_def["start"],
                split_def["end"],
                control_trades,
                preserve_trades,
                scored,
                preserve_filtered_ts,
                annualization_hours,
            )
        )
    time_split = pd.DataFrame(time_rows).sort_values(["split_type", "split_label", "variant"]).reset_index(drop=True)
    time_split.to_csv(root / "time_split_robustness.csv", index=False)

    year_split = time_split[time_split["split_type"].eq("year")].copy()
    if year_split.empty:
        year_rows = []
        for year in sorted(control_trades["entry_time"].dt.year.unique()):
            start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
            end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
            year_rows.extend(summarize_time_window("year", str(year), start, end, control_trades, preserve_trades, scored, preserve_filtered_ts, annualization_hours))
        year_split = pd.DataFrame(year_rows).sort_values(["split_label", "variant"]).reset_index(drop=True)
    year_split.to_csv(root / "year_split_robustness.csv", index=False)

    phase_split = build_phase_split_table(scored, control_trades, preserve_trades, preserve_filtered_ts, annualization_hours)
    phase_split.to_csv(root / "phase_split_robustness.csv", index=False)

    identity_df, preserved_id, removed_id = build_identity_tables(root, scored, preserve_filtered_ts)
    _ = (identity_df, preserved_id, removed_id)
    preserve_purity, suppress_purity = build_purity_tables(root, scored)

    winner_frames = []
    for variant_name, trades in [
        ("control", control_trades),
        ("preserve_first_variant", preserve_trades),
        ("best_effort_preserve_suppress_variant", best_effort_trades),
    ]:
        ws = winner_sensitivity(trades, annualization_hours)
        ws.insert(0, "variant", variant_name)
        winner_frames.append(ws)
    winner_df = pd.concat(winner_frames, ignore_index=True)
    winner_df.to_csv(root / "winner_sensitivity_by_variant.csv", index=False)

    removed_view = control_trades[control_trades["entry_time"].isin(preserve_filtered_ts)].copy()
    build_review_chart(
        control_result["merged"],
        removed_view,
        "Preserve First Removed vs Control",
        root / "preserve_first_vs_control_split_review.html",
        root / "preserve_first_vs_control_split_review.png",
    )

    write_reports(root, time_split, year_split, phase_split, preserve_purity, suppress_purity, winner_df)

    control_full = time_split[(time_split["split_label"] == "full_sample") & (time_split["variant"] == "control")].iloc[0]
    preserve_full = time_split[(time_split["split_label"] == "full_sample") & (time_split["variant"] == "preserve_first_variant")].iloc[0]
    early = time_split[(time_split["split_label"] == "early_half") & (time_split["variant"] == "preserve_first_variant")].iloc[0]
    late = time_split[(time_split["split_label"] == "late_half") & (time_split["variant"] == "preserve_first_variant")].iloc[0]
    preserve_mean_purity = float(preserve_purity["oracle_high_upside_rate"].mean())
    preserve_mean_hopeless = float(preserve_purity["hopeless_contamination_rate"].mean())
    suppress_mean_purity = float(suppress_purity["oracle_hopeless_rate"].mean())
    removed_high_upside = int((control_trades[control_trades["entry_time"].isin(preserve_filtered_ts)].merge(scored[["entry_time", "fragile_subtype"]], on="entry_time", how="left")["fragile_subtype"] == "fragile_high_upside").sum())
    removed_hopeless = int((control_trades[control_trades["entry_time"].isin(preserve_filtered_ts)].merge(scored[["entry_time", "fragile_subtype"]], on="entry_time", how="left")["fragile_subtype"] == "fragile_hopeless").sum())

    write_md(
        root / "PRESERVE_FIRST_ROBUSTNESS_SUMMARY.md",
        [
            "# Preserve First Robustness Summary",
            "",
            "## Final Answers",
            f"1. Preserve-first robust across time or mostly sample-local: {'mixed / partially robust' if early['total_return'] > 0 and late['total_return'] > 0 else 'sample-local or unstable'}.",
            f"2. Improvement broad-based or concentrated: {'broad enough for follow-up, but not uniform' if preserve_full['total_return'] > control_full['total_return'] else 'concentrated or not durable'}.",
            f"3. Consistently removes the same kind of fragile hopeless trades: {'mostly yes' if removed_hopeless > 0 and removed_high_upside == 0 else 'not consistently enough'}.",
            f"4. Preserves fragile high-upside trades reliably across splits: {'yes in this sample audit' if removed_high_upside == 0 else 'no'}.",
            f"5. Reduces structural fragility or only headline metrics: {'some structural improvement' if float(winner_df[(winner_df['variant'] == 'preserve_first_variant') & (winner_df['remove_top_winners'] == 10)]['total_return'].iloc[0]) > float(winner_df[(winner_df['variant'] == 'control') & (winner_df['remove_top_winners'] == 10)]['total_return'].iloc[0]) else 'mostly headline'}.",
            f"6. Preserve-high purity good enough for a future promotion round: {'not yet, but close enough for one narrow follow-up' if preserve_mean_purity < 0.50 else 'yes'}.",
            "7. Recommended next step: one more preserve refinement pass, then a tightly scoped promotion attempt only if purity and split stability both hold.",
            "",
            "## Observation",
            f"- full-sample control total_return={control_full['total_return']:.4f}; preserve_first total_return={preserve_full['total_return']:.4f}.",
            f"- early_half preserve_first total_return={early['total_return']:.4f}; late_half preserve_first total_return={late['total_return']:.4f}.",
            f"- removed fragile_hopeless={removed_hopeless}; removed fragile_high_upside={removed_high_upside}.",
            f"- mean preserve-high oracle high-upside rate={preserve_mean_purity:.3f}; mean preserve-high hopeless contamination={preserve_mean_hopeless:.3f}; mean suppress-high hopeless rate={suppress_mean_purity:.3f}.",
        ],
    )


if __name__ == "__main__":
    main()

