from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from layer12_structural_research_helpers import (
    FROZEN_LAYER12_CONTROL_CONFIG,
    ResearchVariant,
    add_common_bar_features,
    add_forward_entry_quality,
    binary_feature_diagnostics,
    build_trade_audit,
    candidate_quality_label,
    deep_copy_config,
    effect_size,
    extract_overshoot_episodes,
    load_closed_base_frames,
    parse_times,
    quantile_hit_rate_spread,
    run_research_variant,
    save_yaml,
    summarize_returns,
    winner_sensitivity,
)
from risk import RiskConfig


VARIANTS = [
    ResearchVariant(name="control", anchor_policy="latest", reversal_policy="baseline"),
    ResearchVariant(name="locked_anchor", anchor_policy="first", reversal_policy="baseline"),
    ResearchVariant(name="reversal_composite", anchor_policy="latest", reversal_policy="composite_narrow"),
]


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_variant_summary(results: dict[str, dict], annualization_hours: int) -> pd.DataFrame:
    rows = []
    for name, result in results.items():
        trades = parse_times(result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        summary = summarize_returns(trades, annualization_hours)
        rows.append(
            {
                "variant": name,
                "anchor_policy": result["variant"].anchor_policy,
                "reversal_policy": result["variant"].reversal_policy,
                **summary,
                "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
                "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
                "time_stop_early_count": int((trades["exit_reason"] == "time_stop_early").sum()) if not trades.empty else 0,
                "state_stop_count": int((trades["exit_reason"] == "state_stop").sum()) if not trades.empty else 0,
                "winner_share_top3": float(trades["net_return"].nlargest(min(3, len(trades))).sum() / trades["net_return"].sum())
                if not trades.empty and trades["net_return"].sum() != 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def build_candidate_table(results: dict[str, dict]) -> pd.DataFrame:
    control = results["control"]
    locked = results["locked_anchor"]
    reversal = results["reversal_composite"]
    control_bars = add_common_bar_features(control["merged"], control["config"]).reset_index(drop=True)
    episodes = extract_overshoot_episodes(control["merged"], control["config"])
    if episodes.empty:
        return episodes

    risk_cfg = RiskConfig(**control["config"]["risk"])
    control_exec = control["execution"]["false_break"].set_index("timestamp")
    locked_exec = locked["execution"]["false_break"].set_index("timestamp")
    reversal_exec = reversal["execution"]["false_break"].set_index("timestamp")
    control_entries = set(parse_times(control["backtest"]["trades_false_break"], ["entry_time"])["entry_time"].tolist())
    locked_entries = set(parse_times(locked["backtest"]["trades_false_break"], ["entry_time"])["entry_time"].tolist())
    reversal_entries = set(parse_times(reversal["backtest"]["trades_false_break"], ["entry_time"])["entry_time"].tolist())

    rows = []
    for episode in episodes.itertuples(index=False):
        if episode.status != "reentry" or int(episode.reentry_idx) < 0:
            continue
        bar = control_bars.iloc[int(episode.reentry_idx)]
        side = -1 if episode.side == "short" else 1
        stop_distance = max(
            risk_cfg.min_stop_pct * float(bar["close"]),
            risk_cfg.atr_mult * float(bar["atr"]) if pd.notna(bar["atr"]) else 0.0,
            risk_cfg.box_stop_mult * float(bar["box_width"]) if pd.notna(bar["box_width"]) else 0.0,
        )
        quality = candidate_quality_label(
            control_bars,
            int(episode.reentry_idx),
            side,
            float(bar["close"]),
            float(bar["box_midline"]),
            stop_distance,
            risk_cfg.max_hold_bars,
        )
        ts = pd.Timestamp(episode.reentry_time)
        rows.append(
            {
                "episode_id": int(episode.episode_id),
                "timestamp": ts,
                "side": episode.side,
                "anchor_shift_bars": int(getattr(episode, "anchor_shift_bars", 0)),
                "overshoot_bars": int(episode.overshoot_bars),
                "first_within_window": bool(getattr(episode, "first_within_window", False)),
                "last_within_window": bool(getattr(episode, "last_within_window", False)),
                "first_anchor_age_at_reentry": getattr(episode, "first_anchor_age_at_reentry", np.nan),
                "last_anchor_age_at_reentry": getattr(episode, "last_anchor_age_at_reentry", np.nan),
                "first_breakout_volume_ratio": getattr(episode, "first_breakout_volume_ratio", np.nan),
                "last_breakout_volume_ratio": getattr(episode, "last_breakout_volume_ratio", np.nan),
                "reentry_volume_ratio": getattr(episode, "reentry_volume_ratio", np.nan),
                "first_overshoot_distance": getattr(episode, "first_overshoot_distance", np.nan),
                "last_overshoot_distance": getattr(episode, "last_overshoot_distance", np.nan),
                "reentry_box_width_pct": getattr(episode, "reentry_box_width_pct", np.nan),
                "reentry_box_width_atr_mult": getattr(episode, "reentry_box_width_atr_mult", np.nan),
                "reentry_market_phase": getattr(episode, "reentry_market_phase", ""),
                "reentry_session": getattr(episode, "reentry_session", ""),
                "quality_label": quality,
                "good_candidate": quality in {"target_first", "positive_hold"},
                "junk_candidate": quality in {"stop_first", "stop_first_same_bar", "negative_hold"},
                "baseline_exec_signal": int(control_exec.at[ts, "exec_signal"]) if ts in control_exec.index else 0,
                "locked_exec_signal": int(locked_exec.at[ts, "exec_signal"]) if ts in locked_exec.index else 0,
                "reversal_exec_signal": int(reversal_exec.at[ts, "exec_signal"]) if ts in reversal_exec.index else 0,
                "executed_baseline": ts in control_entries,
                "executed_locked_anchor": ts in locked_entries,
                "executed_reversal_variant": ts in reversal_entries,
                "rev_wick": bool(bar["rev_wick_long"] if side > 0 else bar["rev_wick_short"]),
                "rev_engulf": bool(bar["rev_engulf_long"] if side > 0 else bar["rev_engulf_short"]),
                "rev_body_reclaim": bool(bar["rev_body_reclaim_long"] if side > 0 else bar["rev_body_reclaim_short"]),
                "rev_prev_reject": bool(bar["rev_prev_reject_long"] if side > 0 else bar["rev_prev_reject_short"]),
                "rev_outside_inside": bool(bar["rev_outside_inside_long"] if side > 0 else bar["rev_outside_inside_short"]),
                "rev_composite_score": int(bar["rev_composite_score_long"] if side > 0 else bar["rev_composite_score_short"]),
                "volume_rank_24": float(bar["volume_rank_24"]),
                "volume_hour_norm": float(bar["volume_hour_norm"]),
            }
        )
    return pd.DataFrame(rows)

def make_geometry_outputs(root: Path, control_audit: pd.DataFrame, control_result: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = control_audit.copy()
    audit["winner"] = audit["net_return"] > 0
    audit["high_entry_precision"] = audit["entry_precision_12h"] <= audit["entry_precision_12h"].median()
    features = [
        "box_width_pct",
        "box_width_atr_mult",
        "edge_distance_pct",
        "edge_distance_atr",
        "overshoot_distance_pct",
        "overshoot_distance_atr",
        "reentry_distance_pct",
        "reentry_distance_atr",
    ]
    winner_diag = binary_feature_diagnostics(audit, features, "winner", True)
    winner_diag["comparison"] = "winner_vs_loser"
    stop_subset = audit[audit["exit_reason"].isin(["price_stop", "time_stop_max"])].copy()
    stop_subset["time_stop_max_flag"] = stop_subset["exit_reason"] == "time_stop_max"
    stop_diag = binary_feature_diagnostics(stop_subset, features, "time_stop_max_flag", True)
    stop_diag["comparison"] = "time_stop_max_vs_price_stop"
    precision_diag = binary_feature_diagnostics(audit, features, "high_entry_precision", True)
    precision_diag["comparison"] = "high_vs_low_entry_precision"
    geometry = pd.concat([winner_diag, stop_diag, precision_diag], ignore_index=True)
    geometry.to_csv(root / "box_geometry_comparison.csv", index=False)

    split_rows = []
    for feature_name, bucket_col in [("box_width_pct", "box_width_bucket"), ("box_width_atr_mult", "atr_width_bucket")]:
        for bucket, group in audit.groupby(bucket_col, dropna=False):
            split_rows.append(
                {
                    "feature": feature_name,
                    "bucket": str(bucket),
                    "trades": int(len(group)),
                    "win_rate": float((group["net_return"] > 0).mean()),
                    "avg_return": float(group["net_return"].mean()),
                    "price_stop_rate": float((group["exit_reason"] == "price_stop").mean()),
                    "time_stop_max_rate": float((group["exit_reason"] == "time_stop_max").mean()),
                }
            )
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(root / "pct_width_vs_atr_width_feature_split.csv", index=False)

    bars = add_common_bar_features(control_result["merged"], control_result["config"])
    bars["year"] = bars["timestamp"].dt.year
    trade_phase = audit[["entry_time", "net_return", "exit_reason"]].copy()
    trade_phase["year"] = trade_phase["entry_time"].dt.year
    trade_phase = trade_phase.merge(
        bars[["timestamp", "market_phase"]].rename(columns={"timestamp": "entry_time"}),
        on="entry_time",
        how="left",
    )
    bar_group = bars.groupby(["year", "market_phase"], dropna=False).agg(
        valid_box_bars=("box_valid", "sum"),
        median_box_width_pct=("box_width_pct", "median"),
        median_box_width_atr_mult=("box_width_atr_mult", "median"),
        raw_false_break_signals=("event_false_break_signal", lambda x: int((x != 0).sum())),
    )
    trade_group = trade_phase.groupby(["year", "market_phase"], dropna=False).agg(
        trade_count=("net_return", "size"),
        win_rate=("net_return", lambda x: float((x > 0).mean())),
        avg_return=("net_return", "mean"),
        price_stop_rate=("exit_reason", lambda x: float((x == "price_stop").mean())),
    )
    year_phase = bar_group.join(trade_group, how="left").reset_index()
    year_phase.to_csv(root / "year_phase_box_quality.csv", index=False)
    return geometry, split_df, year_phase


def make_anchor_outputs(root: Path, candidate_table: pd.DataFrame, variant_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor_compare = candidate_table.copy()
    anchor_compare.to_csv(root / "first_anchor_vs_last_anchor_comparison.csv", index=False)

    stats_rows = []
    for side_name, group in [("all", candidate_table), *list(candidate_table.groupby("side", dropna=False))]:
        stats_rows.append(
            {
                "side": side_name,
                "episodes": int(len(group)),
                "multi_bar_share": float((group["overshoot_bars"] > 1).mean()) if not group.empty else np.nan,
                "anchor_shift_share": float((group["anchor_shift_bars"] > 0).mean()) if not group.empty else np.nan,
                "mean_anchor_shift_bars": float(group["anchor_shift_bars"].mean()) if not group.empty else np.nan,
                "first_window_only_drop_share": float((group["last_within_window"] & ~group["first_within_window"]).mean()) if not group.empty else np.nan,
                "baseline_exec_rate": float(group["executed_baseline"].mean()) if not group.empty else np.nan,
                "locked_anchor_exec_rate": float(group["executed_locked_anchor"].mean()) if not group.empty else np.nan,
            }
        )
    anchor_stats = pd.DataFrame(stats_rows)
    anchor_stats.to_csv(root / "overshoot_anchor_drift_stats.csv", index=False)

    locked_summary = variant_summary[variant_summary["variant"].isin(["control", "locked_anchor"])].copy()
    locked_summary["delta_vs_control_total_return"] = locked_summary["total_return"] - float(
        locked_summary.loc[locked_summary["variant"] == "control", "total_return"].iloc[0]
    )
    locked_summary.to_csv(root / "locked_anchor_research_variant_summary.csv", index=False)
    return anchor_stats, anchor_compare


def make_reversal_outputs(root: Path, control_audit: pd.DataFrame, candidate_table: pd.DataFrame, variant_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = control_audit.copy()
    audit["wick_required_to_pass"] = audit["wick_reversal_used"] & ((audit["base_confirms"] - 1) < 2)
    pattern_rows = []
    for label, frame in [("executed_baseline", audit), ("candidate_reentries", candidate_table)]:
        for pattern in ["wick_reversal_used", "engulfing_reversal", "body_reclaim_reversal", "prev_reject_combo"] if label == "executed_baseline" else ["rev_wick", "rev_engulf", "rev_body_reclaim", "rev_prev_reject", "rev_outside_inside"]:
            pattern_rows.append(
                {
                    "sample": label,
                    "pattern": pattern,
                    "count": int(frame[pattern].sum()) if pattern in frame.columns else 0,
                    "rate": float(frame[pattern].mean()) if pattern in frame.columns and len(frame) else np.nan,
                }
            )
    pattern_df = pd.DataFrame(pattern_rows)
    pattern_df.to_csv(root / "reversal_pattern_feature_table.csv", index=False)

    missed_good = candidate_table[(~candidate_table["executed_baseline"]) & (candidate_table["good_candidate"])].copy()
    executed = candidate_table[candidate_table["executed_baseline"]].copy()
    compare_rows = []
    for pattern in ["rev_wick", "rev_engulf", "rev_body_reclaim", "rev_prev_reject", "rev_outside_inside"]:
        compare_rows.append(
            {
                "pattern": pattern,
                "executed_rate": float(executed[pattern].mean()) if not executed.empty else np.nan,
                "missed_good_rate": float(missed_good[pattern].mean()) if not missed_good.empty else np.nan,
                "missed_good_minus_executed": float(missed_good[pattern].mean() - executed[pattern].mean()) if not executed.empty and not missed_good.empty else np.nan,
            }
        )
    missed_vs_exec = pd.DataFrame(compare_rows)
    missed_vs_exec.to_csv(root / "missed_good_vs_executed_reversal_patterns.csv", index=False)

    reversal_summary = variant_summary[variant_summary["variant"].isin(["control", "reversal_composite"])].copy()
    reversal_summary["delta_vs_control_total_return"] = reversal_summary["total_return"] - float(
        reversal_summary.loc[reversal_summary["variant"] == "control", "total_return"].iloc[0]
    )
    reversal_summary.to_csv(root / "reversal_variant_summary.csv", index=False)
    return pattern_df, missed_vs_exec, reversal_summary


def make_volume_outputs(root: Path, control_audit: pd.DataFrame, candidate_table: pd.DataFrame, control_result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = control_audit.copy()
    audit["winner"] = audit["net_return"] > 0
    audit["time_stop_max_flag"] = audit["exit_reason"] == "time_stop_max"
    audit["price_stop_flag"] = audit["exit_reason"] == "price_stop"
    features = [
        "fb_breakout_volume_ratio",
        "fb_reentry_volume_ratio",
        "fb_reentry_vs_breakout_ratio",
        "volume_rel_20",
        "volume_rank_24",
        "volume_hour_norm",
    ]
    vol_rows = []
    for comp_name, frame, label in [
        ("winner_vs_loser", audit, "winner"),
        ("time_stop_max_vs_price_stop", audit[audit["exit_reason"].isin(["price_stop", "time_stop_max"])], "time_stop_max_flag"),
        ("missed_good_vs_junk", candidate_table[candidate_table["good_candidate"] | candidate_table["junk_candidate"]], "good_candidate"),
    ]:
        if frame.empty:
            continue
        diag = binary_feature_diagnostics(frame, features, label, True)
        diag["comparison"] = comp_name
        vol_rows.append(diag)
    volume_effect = pd.concat(vol_rows, ignore_index=True) if vol_rows else pd.DataFrame()
    volume_effect.to_csv(root / "volume_feature_effectiveness.csv", index=False)

    bars = add_common_bar_features(control_result["merged"], control_result["config"])
    noise = bars.groupby(["session", "hour_utc"], dropna=False).agg(
        bars=("volume", "size"),
        mean_volume=("volume", "mean"),
        std_volume=("volume", "std"),
        median_rel_volume=("volume_rel_20", "median"),
        median_hour_norm=("volume_hour_norm", "median"),
    ).reset_index()
    noise["volume_cv"] = noise["std_volume"] / noise["mean_volume"].replace(0.0, np.nan)
    noise.to_csv(root / "volume_noise_by_session.csv", index=False)
    return volume_effect, noise


def make_entry_outputs(root: Path, control_audit: pd.DataFrame, results: dict[str, dict], annualization_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit = control_audit.copy()
    entry_precision = audit.groupby("exit_reason", dropna=False).agg(
        trades=("net_return", "size"),
        avg_return=("net_return", "mean"),
        median_entry_precision_6h=("entry_precision_6h", "median"),
        median_entry_precision_12h=("entry_precision_12h", "median"),
        median_entry_precision_24h=("entry_precision_24h", "median"),
        mean_edge_distance_atr=("edge_distance_atr", "mean"),
        mean_overshoot_distance_atr=("overshoot_distance_atr", "mean"),
        mean_confirm_richness=("confirm_richness", "mean"),
        mean_forward_mfe_6h=("forward_mfe_6h", "mean"),
        mean_forward_mae_6h=("forward_mae_6h", "mean"),
    ).reset_index()
    entry_precision.to_csv(root / "entry_precision_by_exit_reason.csv", index=False)

    compare_rows = []
    price_stop = audit[audit["exit_reason"] == "price_stop"]
    time_stop_max = audit[audit["exit_reason"] == "time_stop_max"]
    for feature in ["entry_precision_6h", "entry_precision_12h", "entry_precision_24h", "edge_distance_atr", "overshoot_distance_atr", "confirm_richness", "forward_mfe_6h", "forward_mae_6h"]:
        compare_rows.append(
            {
                "feature": feature,
                "price_stop_mean": float(price_stop[feature].mean()) if not price_stop.empty else np.nan,
                "time_stop_max_mean": float(time_stop_max[feature].mean()) if not time_stop_max.empty else np.nan,
                "effect_size_time_stop_max_minus_price_stop": effect_size(time_stop_max[feature], price_stop[feature]),
                "quantile_spread": quantile_hit_rate_spread(audit[feature], (audit["exit_reason"] == "time_stop_max").astype(float)),
            }
        )
    price_vs_time = pd.DataFrame(compare_rows)
    price_vs_time.to_csv(root / "price_stop_vs_time_stop_max_entry_quality.csv", index=False)

    winner_rows = []
    for name, result in results.items():
        trades = parse_times(result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        ws = winner_sensitivity(trades, annualization_hours)
        ws.insert(0, "variant", name)
        winner_rows.append(ws)
    winner_df = pd.concat(winner_rows, ignore_index=True)
    winner_df.to_csv(root / "winner_dependence_under_research_variants.csv", index=False)
    return entry_precision, price_vs_time, winner_df

def write_reports(
    root: Path,
    variant_summary: pd.DataFrame,
    geometry: pd.DataFrame,
    anchor_stats: pd.DataFrame,
    pattern_df: pd.DataFrame,
    volume_effect: pd.DataFrame,
    entry_compare: pd.DataFrame,
) -> None:
    control = variant_summary.loc[variant_summary["variant"] == "control"].iloc[0]
    locked = variant_summary.loc[variant_summary["variant"] == "locked_anchor"].iloc[0]
    reversal = variant_summary.loc[variant_summary["variant"] == "reversal_composite"].iloc[0]
    geom_winner = geometry[geometry["comparison"] == "winner_vs_loser"].set_index("feature")
    atr_strength = abs(float(geom_winner.loc["box_width_atr_mult", "effect_size"])) if "box_width_atr_mult" in geom_winner.index else np.nan
    pct_strength = abs(float(geom_winner.loc["box_width_pct", "effect_size"])) if "box_width_pct" in geom_winner.index else np.nan
    anchor_all = anchor_stats.loc[anchor_stats["side"] == "all"].iloc[0]
    reversal_rows = pattern_df[pd.notna(pattern_df["rate"])].copy()
    top_patterns = reversal_rows.sort_values("rate", ascending=False).head(3)
    volume_best = volume_effect.reindex(volume_effect["effect_size"].abs().sort_values(ascending=False).index).head(5)
    entry_top = entry_compare.reindex(entry_compare["effect_size_time_stop_max_minus_price_stop"].abs().sort_values(ascending=False).index).head(5)

    write_md(
        root / "LAYER1_BOX_AUDIT.md",
        [
            "# Layer 1 Box Audit",
            "",
            "## Observation",
            f"- Control trades: {int(control['trades'])}, win rate: {control['win_rate']:.2%}, total return: {control['total_return']:.2%}.",
            f"- Width pct winner/loser effect size: {pct_strength:.3f}.",
            f"- Width ATR-multiple winner/loser effect size: {atr_strength:.3f}.",
            "",
            "## Conclusion",
            "- ATR-normalized geometry is stronger only if its effect size materially exceeds raw pct width.",
            f"- In this run, ATR strength {'exceeds' if np.isfinite(atr_strength) and np.isfinite(pct_strength) and atr_strength > pct_strength else 'does not clearly exceed'} pct width strength.",
        ],
    )

    write_md(
        root / "LAYER2_ANCHOR_AUDIT.md",
        [
            "# Layer 2 Anchor Audit",
            "",
            "## Observation",
            f"- Multi-bar overshoot share: {anchor_all['multi_bar_share']:.2%}.",
            f"- Anchor shift share: {anchor_all['anchor_shift_share']:.2%}.",
            f"- First-window-only drop share: {anchor_all['first_window_only_drop_share']:.2%}.",
            f"- Locked-anchor total return delta vs control: {locked['total_return'] - control['total_return']:.2%}.",
            "",
            "## Conclusion",
            "- Anchor drift is real when anchor_shift_share is non-zero; materiality depends on whether the locked-anchor variant changes realized trade outcomes.",
        ],
    )

    write_md(
        root / "LAYER2_REVERSAL_AUDIT.md",
        [
            "# Layer 2 Reversal Audit",
            "",
            "## Observation",
            *[f"- {row.sample} / {row.pattern}: rate={row.rate:.2%}." for row in top_patterns.itertuples(index=False)],
            f"- Reversal-composite total return delta vs control: {reversal['total_return'] - control['total_return']:.2%}.",
            "",
            "## Conclusion",
            "- Alternative two-bar/composite reversal patterns matter only if they appear disproportionately in good missed candidates and/or improve the narrow research variant without increasing tail dependence.",
        ],
    )

    write_md(
        root / "VOLUME_AUDIT.md",
        [
            "# Volume Audit",
            "",
            "## Observation",
            *[f"- {row.comparison} / {row.feature}: effect={row.effect_size:.3f}, spread={row.quantile_hit_rate_spread:.3f}." for row in volume_best.itertuples(index=False)],
            "",
            "## Conclusion",
            "- Volume should remain off unless it shows stable separation beyond weak audit-level signal strength.",
        ],
    )

    write_md(
        root / "ENTRY_EDGE_AUDIT.md",
        [
            "# Entry Edge Audit",
            "",
            "## Observation",
            *[f"- {row.feature}: effect(time_stop_max - price_stop)={row.effect_size_time_stop_max_minus_price_stop:.3f}." for row in entry_top.itertuples(index=False)],
            "",
            "## Conclusion",
            "- If time_stop_max winners consistently show better entry-precision and excursion metrics than price_stop losers, weak entry precision is a plausible driver of long-tail dependence.",
        ],
    )

    promising = variant_summary.sort_values(["total_return", "max_drawdown"], ascending=[False, False]).iloc[0]
    falsified = []
    if not (np.isfinite(atr_strength) and np.isfinite(pct_strength) and atr_strength > pct_strength):
        falsified.append("ATR geometry clearly outperforming pct width")
    if volume_best.empty or volume_best["effect_size"].abs().max() < 0.2:
        falsified.append("volume being a strong hard gate")
    if locked["total_return"] <= control["total_return"] and locked["trades"] <= control["trades"]:
        falsified.append("anchor locking obviously improving the strategy")

    write_md(
        root / "NEXT_STEP_LAYER12_RESEARCH_SUMMARY.md",
        [
            "# Next Step Layer 1/2 Research Summary",
            "",
            "## Final Answers",
            f"1. Box-width pct rigidity bottleneck: {'not clearly falsified' if np.isfinite(atr_strength) and np.isfinite(pct_strength) and atr_strength > pct_strength else 'critique looks overstated in this run'}.",
            f"2. ATR-normalized geometry explanatory value: {'higher than pct width' if np.isfinite(atr_strength) and np.isfinite(pct_strength) and atr_strength > pct_strength else 'not materially better than pct width'}.",
            f"3. Breakout-anchor drift real: {'yes' if anchor_all['anchor_shift_share'] > 0 else 'no'}, materiality delta: {locked['total_return'] - control['total_return']:.2%} total return.",
            f"4. Reversal confirmation too narrow: {'possibly' if reversal['total_return'] > control['total_return'] else 'not clearly proven'}; reversal variant delta: {reversal['total_return'] - control['total_return']:.2%}.",
            f"5. Volume help: {'weak audit-only value at best' if not volume_best.empty else 'no reliable value observed'}.",
            f"6. Weak entry precision main driver: {'supported' if entry_top['effect_size_time_stop_max_minus_price_stop'].abs().max() > 0.3 else 'not strongly proven'} by entry-quality splits.",
            f"7. Most promising Layer 1/2 improvement: {promising['variant']}.",
            f"8. Critiques falsified: {', '.join(falsified) if falsified else 'none clearly falsified'}.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Layer 1/Layer 2 structural research.")
    parser.add_argument("--outdir", default="outputs/layer12_structural_research")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(FROZEN_LAYER12_CONTROL_CONFIG, root / "frozen_control_config.yaml")

    cfg = deep_copy_config()
    _, boxes_1h, boxes_4h, enriched_1h = load_closed_base_frames(cfg)

    results = {}
    for variant in VARIANTS:
        variant_dir = root / variant.name
        results[variant.name] = run_research_variant(cfg, boxes_1h, boxes_4h, enriched_1h, variant_dir, variant)

    annualization_hours = int(cfg["backtest"]["annualization_hours"])
    variant_summary = build_variant_summary(results, annualization_hours)
    variant_summary.to_csv(root / "variant_summary.csv", index=False)

    control_audit = add_forward_entry_quality(build_trade_audit(results["control"]), results["control"]["merged"])
    control_audit.to_csv(root / "control_trade_audit.csv", index=False)

    candidate_table = build_candidate_table(results)
    candidate_table.to_csv(root / "first_anchor_vs_last_anchor_comparison.csv", index=False)

    geometry, _, _ = make_geometry_outputs(root, control_audit, results["control"])
    anchor_stats, _ = make_anchor_outputs(root, candidate_table, variant_summary)
    pattern_df, _, _ = make_reversal_outputs(root, control_audit, candidate_table, variant_summary)
    volume_effect, _ = make_volume_outputs(root, control_audit, candidate_table, results["control"])
    _, entry_compare, _ = make_entry_outputs(root, control_audit, results, annualization_hours)

    write_reports(root, variant_summary, geometry, anchor_stats, pattern_df, volume_effect, entry_compare)
    print(f"Layer 1/2 structural research complete. Outputs saved to: {root}")


if __name__ == "__main__":
    main()
