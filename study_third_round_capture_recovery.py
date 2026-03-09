"""Third-round capture audit and narrow recovery experiments for false-break trades."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd

from make_tradeable_bs_equity_chart import build_html, build_png
from research_context import ATTRITION_OVERLAP_NOTE, compute_tradable_context_mask
from risk import RiskConfig, compute_atr
from study_event_first import run_study
from study_first_round_improvements import (
    WIDTH_BINS,
    WIDTH_LABELS,
    _deep_merge,
    _exit_reason_ratio,
    _force_closed_inputs,
    _load_yaml,
    _save_yaml,
    _summary,
)

H4_WIDTH_BINS = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, np.inf]
H4_WIDTH_LABELS = ["<=2.0%", "2.0-3.0%", "3.0-4.0%", "4.0-5.0%", "5.0-6.0%", ">6.0%"]
CONFIRM_COUNT_BINS = [-0.5, 0.5, 1.5, 2.5, 10.0]
CONFIRM_COUNT_LABELS = ["0", "1", "2", "3+"]
POSTHOC_HORIZONS = [2, 4, 6]

FROZEN_MASTER_BASELINE_OVERRIDES = {
    "event_false_break": {
        "overshoot_width_mult": 0.08,
        "reentry_window": 6,
        "min_confirmations": 2,
        "recapture_to_mid_frac": 0.40,
        "momentum_lookback": 3,
        "reversal_wick_ratio": 0.45,
        "volume_confirmation_enabled": False,
        "volume_confirmation_mode": "require",
        "volume_lookback": 20,
        "volume_min_periods": 10,
        "breakout_volume_max_ratio": 1.15,
        "reentry_volume_min_ratio": 0.95,
        "reentry_vs_breakout_min_ratio": 1.05,
    },
    "execution": {
        "width_pct_min": 0.001,
        "width_pct_max": 0.06,
        "method_agreement_min": 0.34,
        "transition_risk_max": 0.65,
        "confidence_min_false_break": 0.45,
        "confidence_min_boundary": 0.50,
        "side_cooldown_bars_false_break": 4,
        "side_cooldown_bars_boundary": 6,
        "allow_states": ["STABLE", "SQUEEZE"],
        "false_break_regime_mode": "down_only",
        "regime_lookback_hours": 720,
        "regime_down_low": -0.20,
        "regime_down_high": -0.05,
        "regime_up_low": 0.05,
        "regime_up_high": 0.20,
        "false_break_up_long_only": False,
        "allow_warmup_trades": False,
        "bad_width_bucket_filter_enabled": True,
        "bad_width_bucket_min_pct": 0.010,
        "bad_width_bucket_max_pct": 0.015,
        "expected_value_filter_enabled": False,
        "expected_value_mode": "to_midline",
        "min_reward_to_cost_ratio": 1.5,
        "expected_value_round_trip_cost_rate": None,
        "capture_recovery_exec_conf_low_enabled": False,
        "capture_recovery_exec_conf_low_min_reward_to_cost": 3.0,
        "capture_recovery_exec_conf_low_min_reentry_strength": 0.25,
        "capture_recovery_confirm_near_miss_enabled": False,
        "capture_recovery_confirm_near_miss_exact_base_confirms": 1,
        "capture_recovery_confirm_near_miss_min_box_confidence": 0.98,
        "capture_recovery_confirm_near_miss_min_h4_method": 0.75,
        "capture_recovery_confirm_near_miss_max_h4_transition": 0.08,
        "capture_recovery_confirm_near_miss_min_reward_to_cost": 2.5,
        "capture_recovery_confirm_near_miss_min_reentry_strength": 0.15,
    },
    "risk": {
        "atr_window": 14,
        "atr_mult": 1.10,
        "box_stop_mult": 0.60,
        "min_stop_pct": 0.002,
        "max_hold_bars": 30,
        "early_progress_bars": 16,
        "early_progress_min_return": 0.0003,
        "state_stop_grace_bars": 2,
        "side_cooldown_bars": 4,
        "circuit_breaker_loss_streak": 3,
        "circuit_breaker_pause_bars": 16,
        "take_profit_enabled": False,
        "take_profit_mode": "midline",
        "early_failure_filter_enabled": False,
        "early_failure_bars": 6,
        "early_failure_min_progress": 0.003,
        "early_failure_max_adverse": 0.008,
        "early_failure_scope": "all",
        "early_failure_confirm_threshold": 2,
        "early_failure_width_min_pct": 0.010,
        "early_failure_width_max_pct": 0.015,
    },
    "backtest": {
        "fee_bps_per_leg": 5,
        "slippage_bps_per_leg": 2,
        "half_spread_bps_per_leg": 1,
        "combine_mode": "equal_weight",
        "engine_weights": {"false_break": 1.00, "boundary": 0.00},
        "annualization_hours": 8760,
    },
}

VARIANTS = [
    ("baseline_master_current", copy.deepcopy(FROZEN_MASTER_BASELINE_OVERRIDES)),
    ("variant_capture_audit_only", copy.deepcopy(FROZEN_MASTER_BASELINE_OVERRIDES)),
    (
        "variant_near_miss_recovery_1",
        {"execution": {"capture_recovery_exec_conf_low_enabled": True, "capture_recovery_exec_conf_low_min_reward_to_cost": 3.0, "capture_recovery_exec_conf_low_min_reentry_strength": 0.25}},
    ),
    (
        "variant_near_miss_recovery_2",
        {"execution": {"capture_recovery_confirm_near_miss_enabled": True, "capture_recovery_confirm_near_miss_exact_base_confirms": 1, "capture_recovery_confirm_near_miss_min_box_confidence": 0.98, "capture_recovery_confirm_near_miss_min_h4_method": 0.75, "capture_recovery_confirm_near_miss_max_h4_transition": 0.08, "capture_recovery_confirm_near_miss_min_reward_to_cost": 2.5, "capture_recovery_confirm_near_miss_min_reentry_strength": 0.15}},
    ),
    (
        "variant_soft_confirm_strong_context",
        {"execution": {"capture_recovery_confirm_near_miss_enabled": True, "capture_recovery_confirm_near_miss_exact_base_confirms": 1, "capture_recovery_confirm_near_miss_min_box_confidence": 0.99, "capture_recovery_confirm_near_miss_min_h4_method": 0.80, "capture_recovery_confirm_near_miss_max_h4_transition": 0.06, "capture_recovery_confirm_near_miss_min_reward_to_cost": 3.0, "capture_recovery_confirm_near_miss_min_reentry_strength": 0.10}},
    ),
    (
        "variant_best_effort_capture_recovery",
        {"execution": {"capture_recovery_exec_conf_low_enabled": True, "capture_recovery_exec_conf_low_min_reward_to_cost": 3.0, "capture_recovery_exec_conf_low_min_reentry_strength": 0.25, "capture_recovery_confirm_near_miss_enabled": True, "capture_recovery_confirm_near_miss_exact_base_confirms": 1, "capture_recovery_confirm_near_miss_min_box_confidence": 0.99, "capture_recovery_confirm_near_miss_min_h4_method": 0.80, "capture_recovery_confirm_near_miss_max_h4_transition": 0.06, "capture_recovery_confirm_near_miss_min_reward_to_cost": 3.0, "capture_recovery_confirm_near_miss_min_reentry_strength": 0.10}},
    ),
]


def _parse_time(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True)
    return out


def _winner_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values("net_return", ascending=False).reset_index(drop=True)
    for n in [0, 3, 5, 10]:
        trimmed = ranked.iloc[n:].copy() if n > 0 else ranked.copy()
        rows.append({"remove_top_winners": n, **_summary(trimmed.sort_values("exit_time"), 8760, return_col="net_return")})
    return pd.DataFrame(rows)


def _context_bucket(df: pd.DataFrame) -> pd.Series:
    box_conf = pd.to_numeric(df.get("box_confidence", np.nan), errors="coerce")
    h4_method = pd.to_numeric(df.get("h4_method_agreement", np.nan), errors="coerce")
    h4_transition = pd.to_numeric(df.get("h4_transition_risk", np.nan), errors="coerce")
    return np.select(
        [(box_conf >= 0.98) & (h4_method >= 0.78) & (h4_transition <= 0.06), (box_conf >= 0.90) & (h4_method >= 0.70) & (h4_transition <= 0.10)],
        ["elite", "strong"],
        default="standard",
    )


def _candidate_side(df: pd.DataFrame) -> pd.Series:
    up = df["fb_overshoot_up"].fillna(False)
    dn = df["fb_overshoot_down"].fillna(False)
    return pd.Series(np.where(dn, 1, np.where(up, -1, 0)), index=df.index)


def _reentry_strength_from_close(df: pd.DataFrame, side_col: str = "candidate_side") -> pd.Series:
    side = pd.to_numeric(df[side_col], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    mid = pd.to_numeric(df["box_midline"], errors="coerce")
    lower = pd.to_numeric(df["box_lower_edge"], errors="coerce")
    upper = pd.to_numeric(df["box_upper_edge"], errors="coerce")
    strength = pd.Series(np.nan, index=df.index, dtype=float)
    long_mask = (side > 0) & (mid > lower)
    short_mask = (side < 0) & (upper > mid)
    strength.loc[long_mask] = (close.loc[long_mask] - lower.loc[long_mask]) / (mid.loc[long_mask] - lower.loc[long_mask])
    strength.loc[short_mask] = (upper.loc[short_mask] - close.loc[short_mask]) / (upper.loc[short_mask] - mid.loc[short_mask])
    return strength.clip(lower=0.0)


def _build_trade_audit(variant_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    trades = _parse_time(pd.read_csv(variant_dir / "trades_false_break.csv"), ["entry_time", "exit_time"])
    if trades.empty:
        return trades

    execution = _parse_time(pd.read_csv(variant_dir / "execution_false_break.csv"), ["timestamp"]).rename(columns={"timestamp": "entry_time"})
    events = _parse_time(pd.read_csv(variant_dir / "events_merged.csv"), ["timestamp"]).rename(columns={"timestamp": "entry_time"})
    bt_cost = 2.0 * ((cfg["backtest"]["fee_bps_per_leg"] + cfg["backtest"]["slippage_bps_per_leg"] + cfg["backtest"]["half_spread_bps_per_leg"]) / 10000.0)
    events["atr"] = compute_atr(events[["high", "low", "close"]].copy(), cfg["risk"]["atr_window"])

    keep_exec = ["entry_time", "fb_regime", "fb_regime_ret", "box_width_pct", "event_base_confirms", "event_confidence", "reward_to_cost_ratio", "expected_reward_rate", "expected_cost_rate", "blocked_reason", "capture_recovery_used", "capture_recovery_reason", "capture_recovery_candidate_signal", "effective_signal_input", "reentry_strength"]
    keep_evt = ["entry_time", "box_midline", "box_upper_edge", "box_lower_edge", "box_width", "box_width_pct", "h4_box_width_pct", "box_confidence", "h4_method_agreement", "h4_transition_risk", "atr"]
    audit = trades.merge(execution[[c for c in keep_exec if c in execution.columns]], on="entry_time", how="left")
    audit = audit.merge(events[[c for c in keep_evt if c in events.columns]], on="entry_time", how="left", suffixes=("", "_evt"))
    audit["box_width_bucket"] = pd.cut(audit["box_width_pct"], bins=WIDTH_BINS, labels=WIDTH_LABELS, include_lowest=True, right=True)
    audit["h4_width_bucket"] = pd.cut(audit["h4_box_width_pct"], bins=H4_WIDTH_BINS, labels=H4_WIDTH_LABELS, include_lowest=True, right=True)
    audit["reward_to_midline"] = audit["side"] * (audit["box_midline"] / audit["entry_price"] - 1.0)
    opposite = np.where(audit["side"] > 0, audit["box_upper_edge"], audit["box_lower_edge"])
    audit["reward_to_opposite_edge"] = audit["side"] * (pd.Series(opposite, index=audit.index) / audit["entry_price"] - 1.0)
    audit["reward_to_cost_midline"] = audit["reward_to_midline"] / bt_cost
    stop_dist = pd.concat([cfg["risk"]["min_stop_pct"] * audit["entry_price"], cfg["risk"]["atr_mult"] * audit["atr"], cfg["risk"]["box_stop_mult"] * audit["box_width"]], axis=1).max(axis=1)
    audit["entry_stop_distance"] = stop_dist
    audit["stop_dist_box_ratio"] = stop_dist / audit["box_width"].replace(0.0, np.nan)
    return audit

def _posthoc_candidate_audit(variant_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    events = _parse_time(pd.read_csv(variant_dir / "events_merged.csv"), ["timestamp"])
    execution = _parse_time(pd.read_csv(variant_dir / "execution_false_break.csv"), ["timestamp"])
    trades = _parse_time(pd.read_csv(variant_dir / "trades_false_break.csv"), ["entry_time", "exit_time"])

    merged = events.merge(execution, on="timestamp", how="inner", suffixes=("", "_exec")).sort_values("timestamp").reset_index(drop=True)
    merged["tradable_context"] = compute_tradable_context_mask(merged)
    merged["raw_candidate"] = merged["tradable_context"] & (merged["fb_overshoot_up"].fillna(False) | merged["fb_overshoot_down"].fillna(False))
    merged["event_survivor"] = merged["tradable_context"] & (merged["fb_reenter_from_up"].fillna(False) | merged["fb_reenter_from_down"].fillna(False))
    merged["confirm_survivor"] = merged["tradable_context"] & (merged["raw_signal"].fillna(0).astype(int) != 0)
    merged["execution_survivor"] = merged["tradable_context"] & (merged["exec_signal"].fillna(0).astype(int) != 0)
    merged["candidate_side"] = _candidate_side(merged)
    merged["context_bucket"] = _context_bucket(merged)
    merged["box_width_bucket"] = pd.cut(merged["box_width_pct"], bins=WIDTH_BINS, labels=WIDTH_LABELS, include_lowest=True, right=True)
    merged["h4_width_bucket"] = pd.cut(merged["h4_box_width_pct"], bins=H4_WIDTH_BINS, labels=H4_WIDTH_LABELS, include_lowest=True, right=True)
    merged["confirm_count_bucket"] = pd.cut(merged["event_false_break_base_confirms"].fillna(0), bins=CONFIRM_COUNT_BINS, labels=CONFIRM_COUNT_LABELS, include_lowest=True, right=True)
    merged["reentry_strength_candidate"] = _reentry_strength_from_close(merged)
    merged["atr"] = compute_atr(merged[["high", "low", "close"]].copy(), cfg["risk"]["atr_window"])

    trade_keys = set()
    for row in trades.itertuples(index=False):
        side = 1 if getattr(row, "side") > 0 else -1
        trade_keys.add((pd.Timestamp(getattr(row, "entry_time")), side))

    risk_cfg = RiskConfig(**cfg["risk"])
    idx_map = {ts: i for i, ts in enumerate(merged["timestamp"].tolist())}
    rows = []
    for row in merged.loc[merged["raw_candidate"]].itertuples(index=False):
        entry_idx = idx_map.get(row.timestamp)
        side = int(row.candidate_side)
        entry_price = float(row.close)
        if entry_idx is None or side == 0 or not np.isfinite(entry_price) or entry_price <= 0.0:
            continue

        width = float(row.box_width) if np.isfinite(row.box_width) else np.nan
        atr = float(row.atr) if np.isfinite(row.atr) else np.nan
        stop_dist = max(cfg["risk"]["min_stop_pct"] * entry_price, (cfg["risk"]["atr_mult"] * atr) if np.isfinite(atr) else 0.0, (cfg["risk"]["box_stop_mult"] * width) if np.isfinite(width) else 0.0)
        stop_price = entry_price - stop_dist if side > 0 else entry_price + stop_dist
        stop_hit = False
        reached_mid = False
        reached_opp = False
        mfe = {h: 0.0 for h in POSTHOC_HORIZONS}
        mae = {h: 0.0 for h in POSTHOC_HORIZONS}
        upper = float(row.box_upper_edge)
        lower = float(row.box_lower_edge)
        mid = float(row.box_midline)
        end_idx = min(entry_idx + risk_cfg.max_hold_bars, len(merged) - 1)
        for j in range(entry_idx + 1, end_idx + 1):
            high = float(merged.at[j, "high"])
            low = float(merged.at[j, "low"])
            for horizon in POSTHOC_HORIZONS:
                if j <= entry_idx + horizon:
                    if side > 0:
                        mfe[horizon] = max(mfe[horizon], max(high / entry_price - 1.0, 0.0))
                        mae[horizon] = max(mae[horizon], max(1.0 - low / entry_price, 0.0))
                    else:
                        mfe[horizon] = max(mfe[horizon], max(1.0 - low / entry_price, 0.0))
                        mae[horizon] = max(mae[horizon], max(high / entry_price - 1.0, 0.0))
            if side > 0:
                reached_mid = reached_mid or (np.isfinite(mid) and high >= mid)
                reached_opp = reached_opp or (np.isfinite(upper) and high >= upper)
                if low <= stop_price:
                    stop_hit = True
                    break
            else:
                reached_mid = reached_mid or (np.isfinite(mid) and low <= mid)
                reached_opp = reached_opp or (np.isfinite(lower) and low <= lower)
                if high >= stop_price:
                    stop_hit = True
                    break
            if reached_mid and reached_opp:
                break

        final_trade = (pd.Timestamp(row.timestamp), side) in trade_keys
        if not bool(row.event_survivor):
            stage = "event_missed"
        elif not bool(row.confirm_survivor):
            stage = "confirm_missed"
        elif not bool(row.execution_survivor):
            stage = "execution_missed"
        elif not final_trade:
            stage = "risk_dropped"
        else:
            stage = "executed_trade"

        rows.append({
            "timestamp": row.timestamp,
            "stage": stage,
            "candidate_side": side,
            "tradable_context": bool(row.tradable_context),
            "event_survivor": bool(row.event_survivor),
            "confirm_survivor": bool(row.confirm_survivor),
            "execution_survivor": bool(row.execution_survivor),
            "final_trade": final_trade,
            "fb_regime": row.fb_regime,
            "context_bucket": row.context_bucket,
            "box_width_bucket": row.box_width_bucket,
            "h4_width_bucket": row.h4_width_bucket,
            "confirm_count_bucket": row.confirm_count_bucket,
            "event_base_confirms": float(row.event_false_break_base_confirms),
            "event_confidence": float(row.event_confidence),
            "box_confidence": float(row.box_confidence),
            "h4_method_agreement": float(row.h4_method_agreement),
            "h4_transition_risk": float(row.h4_transition_risk),
            "box_width_pct": float(row.box_width_pct),
            "h4_box_width_pct": float(row.h4_box_width_pct),
            "reward_to_cost_ratio": float(row.reward_to_cost_ratio) if pd.notna(row.reward_to_cost_ratio) else np.nan,
            "reentry_strength": float(row.reentry_strength) if pd.notna(row.reentry_strength) else float(row.reentry_strength_candidate),
            "capture_recovery_candidate_signal": int(row.capture_recovery_candidate_signal) if pd.notna(row.capture_recovery_candidate_signal) else 0,
            "capture_recovery_used": bool(row.capture_recovery_used),
            "capture_recovery_reason": str(row.capture_recovery_reason),
            "blocked_reason": str(row.blocked_reason),
            "good_mid_before_stop": bool(reached_mid and not stop_hit),
            "good_opp_before_stop": bool(reached_opp and not stop_hit),
            "entry_stop_distance": float(stop_dist),
            "stop_dist_box_ratio": float(stop_dist / width) if np.isfinite(width) and width > 0 else np.nan,
            **{f"mfe_{h}": mfe[h] for h in POSTHOC_HORIZONS},
            **{f"mae_{h}": mae[h] for h in POSTHOC_HORIZONS},
        })

    return pd.DataFrame(rows)


def _tradable_bar_count(variant_dir: Path) -> int:
    events = _parse_time(pd.read_csv(variant_dir / "events_merged.csv"), ["timestamp"])
    execution = _parse_time(pd.read_csv(variant_dir / "execution_false_break.csv"), ["timestamp"])
    merged = events.merge(execution, on="timestamp", how="inner")
    return int(compute_tradable_context_mask(merged).sum())


def _capture_funnel(candidate_df: pd.DataFrame, tradable_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = int(len(candidate_df))
    event = int(candidate_df["event_survivor"].sum())
    confirm = int(candidate_df["confirm_survivor"].sum())
    execution = int(candidate_df["execution_survivor"].sum())
    final = int(candidate_df["final_trade"].sum())
    tradable = int(tradable_count)
    funnel = pd.DataFrame([
        {"stage": "tradable_context_bars", "count": tradable, "rate_vs_prev": np.nan},
        {"stage": "raw_false_break_candidates", "count": raw, "rate_vs_prev": raw / tradable if tradable else 0.0},
        {"stage": "event_layer_survivors", "count": event, "rate_vs_prev": event / raw if raw else 0.0},
        {"stage": "confirm_layer_survivors", "count": confirm, "rate_vs_prev": confirm / event if event else 0.0},
        {"stage": "execution_layer_survivors", "count": execution, "rate_vs_prev": execution / confirm if confirm else 0.0},
        {"stage": "final_trades", "count": final, "rate_vs_prev": final / execution if execution else 0.0},
    ])
    event_attrition = pd.DataFrame([
        {"layer": "event", "reason": "missing_reentry", "reason_hits": int((candidate_df["stage"] == "event_missed").sum())},
        {"layer": "confirm", "reason": "base_confirms_0", "reason_hits": int(((candidate_df["stage"] == "confirm_missed") & (candidate_df["event_base_confirms"] <= 0)).sum())},
        {"layer": "confirm", "reason": "base_confirms_1", "reason_hits": int(((candidate_df["stage"] == "confirm_missed") & (candidate_df["event_base_confirms"] == 1)).sum())},
    ])
    counts: dict[str, int] = {}
    for text in candidate_df.loc[candidate_df["stage"] == "execution_missed", "blocked_reason"].fillna("").astype(str):
        for reason in text.split("|"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    execution_attrition = pd.DataFrame([{"layer": "execution", "reason": k, "reason_hits": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))])
    execution_attrition = pd.concat([execution_attrition, pd.DataFrame([{"layer": "post_execution", "reason": "risk_dropped_after_execution", "reason_hits": int((candidate_df["stage"] == "risk_dropped").sum())}])], ignore_index=True)
    return funnel, event_attrition, execution_attrition


def _capture_rate_table(candidate_df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    rows = []
    for bucket, group in candidate_df.groupby(bucket_col, dropna=False, observed=False):
        raw = len(group)
        event = int(group["event_survivor"].sum())
        confirm = int(group["confirm_survivor"].sum())
        execution = int(group["execution_survivor"].sum())
        final = int(group["final_trade"].sum())
        rows.append({bucket_col: str(bucket), "raw_candidates": raw, "event_survivors": event, "confirm_survivors": confirm, "execution_survivors": execution, "final_trades": final, "event_capture_rate": event / raw if raw else 0.0, "confirm_capture_rate": confirm / event if event else 0.0, "execution_capture_rate": execution / confirm if confirm else 0.0, "trade_capture_rate": final / raw if raw else 0.0})
    return pd.DataFrame(rows)


def _missed_vs_executed_comparison(candidate_df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "executed_trade": candidate_df[candidate_df["stage"] == "executed_trade"],
        "risk_dropped": candidate_df[candidate_df["stage"] == "risk_dropped"],
        "execution_missed": candidate_df[candidate_df["stage"] == "execution_missed"],
        "confirm_missed": candidate_df[candidate_df["stage"] == "confirm_missed"],
        "event_missed": candidate_df[candidate_df["stage"] == "event_missed"],
        "missed_good_mid": candidate_df[(candidate_df["stage"] != "executed_trade") & candidate_df["good_mid_before_stop"]],
    }
    cols = ["event_base_confirms", "event_confidence", "box_confidence", "h4_method_agreement", "h4_transition_risk", "reward_to_cost_ratio", "reentry_strength", "box_width_pct", "good_mid_before_stop", "good_opp_before_stop", "mfe_2", "mae_2", "mfe_4", "mae_4"]
    rows = []
    for name, group in groups.items():
        if group.empty:
            continue
        row = {"group": name, "count": int(len(group))}
        for col in cols:
            row[col] = float(group[col].mean()) if col in group.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _variant_summary_row(name: str, trades: pd.DataFrame, candidate_df: pd.DataFrame, annualization_hours: int) -> Dict[str, Any]:
    s = _summary(trades.sort_values("exit_time"), annualization_hours, return_col="net_return")
    price_stop = trades[trades["exit_reason"] == "price_stop"]
    time_stop_max = trades[trades["exit_reason"] == "time_stop_max"]
    state_stop = trades[trades["exit_reason"] == "state_stop"]
    return {
        "variant": name,
        **s,
        "price_stop_count": int(len(price_stop)),
        "price_stop_total_return": float((1.0 + price_stop["net_return"]).prod() - 1.0) if not price_stop.empty else 0.0,
        "time_stop_max_count": int(len(time_stop_max)),
        "time_stop_max_total_return": float((1.0 + time_stop_max["net_return"]).prod() - 1.0) if not time_stop_max.empty else 0.0,
        "state_stop_count": int(len(state_stop)),
        "state_stop_total_return": float((1.0 + state_stop["net_return"]).prod() - 1.0) if not state_stop.empty else 0.0,
        "capture_rate_in_tradable": float(candidate_df["final_trade"].sum() / len(candidate_df)) if len(candidate_df) else 0.0,
        "execution_capture_rate": float(candidate_df["execution_survivor"].sum() / len(candidate_df)) if len(candidate_df) else 0.0,
    }


def _trade_key(df: pd.DataFrame) -> pd.Series:
    return df["entry_time"].astype(str) + "|" + df["side"].astype(str)


def _baseline_trade_comparison(variant_name: str, variant_trades: pd.DataFrame, baseline_trades: pd.DataFrame, baseline_candidates: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame]:
    if variant_name == "baseline_master_current":
        return {"baseline_trades_retained": int(len(baseline_trades)), "baseline_trades_lost": 0, "new_trades_added": 0, "missed_good_candidate_recovery_count": 0, "newly_introduced_bad_trade_count": 0, "new_price_stop_count": 0}, pd.DataFrame()

    variant = variant_trades.copy()
    baseline = baseline_trades.copy()
    variant["trade_key"] = _trade_key(variant)
    baseline["trade_key"] = _trade_key(baseline)
    base_keys = set(baseline["trade_key"])
    variant_keys = set(variant["trade_key"])
    retained = base_keys & variant_keys
    lost = base_keys - variant_keys
    new = variant_keys - base_keys

    baseline_lookup = baseline_candidates.copy()
    baseline_lookup["trade_key"] = baseline_lookup["timestamp"].astype(str) + "|" + baseline_lookup["candidate_side"].astype(str)
    baseline_lookup = baseline_lookup.drop_duplicates("trade_key").set_index("trade_key")

    new_rows = []
    new_negative = 0
    new_price_stop = 0
    recovered_good = 0
    for row in variant[variant["trade_key"].isin(new)].itertuples(index=False):
        lookup = baseline_lookup.loc[row.trade_key] if row.trade_key in baseline_lookup.index else None
        good_mid = bool(getattr(lookup, "good_mid_before_stop", False)) if lookup is not None else False
        if good_mid:
            recovered_good += 1
        if float(row.net_return) <= 0.0:
            new_negative += 1
        if str(row.exit_reason) == "price_stop":
            new_price_stop += 1
        new_rows.append({
            "variant": variant_name,
            "entry_time": row.entry_time,
            "side": row.side,
            "net_return": row.net_return,
            "exit_reason": row.exit_reason,
            "hold_bars": row.hold_bars,
            "capture_recovery_used": bool(getattr(row, "capture_recovery_used", False)),
            "capture_recovery_reason": getattr(row, "capture_recovery_reason", ""),
            "baseline_stage": getattr(lookup, "stage", "not_in_baseline_candidates") if lookup is not None else "not_in_baseline_candidates",
            "baseline_good_mid_before_stop": good_mid,
            "baseline_good_opp_before_stop": bool(getattr(lookup, "good_opp_before_stop", False)) if lookup is not None else False,
            "baseline_blocked_reason": getattr(lookup, "blocked_reason", "") if lookup is not None else "",
            "baseline_reward_to_cost_ratio": getattr(lookup, "reward_to_cost_ratio", np.nan) if lookup is not None else np.nan,
            "baseline_reentry_strength": getattr(lookup, "reentry_strength", np.nan) if lookup is not None else np.nan,
            "baseline_event_base_confirms": getattr(lookup, "event_base_confirms", np.nan) if lookup is not None else np.nan,
        })

    return {"baseline_trades_retained": int(len(retained)), "baseline_trades_lost": int(len(lost)), "new_trades_added": int(len(new)), "missed_good_candidate_recovery_count": int(recovered_good), "newly_introduced_bad_trade_count": int(new_negative), "new_price_stop_count": int(new_price_stop)}, pd.DataFrame(new_rows)


def _best_variant_name(summary_df: pd.DataFrame) -> str:
    baseline = summary_df.loc[summary_df["variant"] == "baseline_master_current"].iloc[0]
    candidates = summary_df[~summary_df["variant"].isin(["baseline_master_current", "variant_capture_audit_only"])].copy()
    candidates = candidates[(candidates["price_stop_count"] <= baseline["price_stop_count"] + 3) & (candidates["time_stop_max_count"] >= baseline["time_stop_max_count"] - 3) & (candidates["time_stop_max_total_return"] >= baseline["time_stop_max_total_return"] * 0.85)]
    if candidates.empty:
        return "baseline_master_current"
    best = candidates.sort_values(["total_return", "sharpe"], ascending=False).iloc[0]
    if float(best["total_return"]) <= float(baseline["total_return"]):
        return "baseline_master_current"
    return str(best["variant"])


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run third-round capture audit and narrow recovery experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outdir", default="outputs/third_round_capture_recovery")
    args = parser.parse_args()

    base_cfg = _force_closed_inputs(_load_yaml(args.config))
    frozen_cfg = copy.deepcopy(base_cfg)
    _deep_merge(frozen_cfg, copy.deepcopy(FROZEN_MASTER_BASELINE_OVERRIDES))
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    annualization_hours = int(frozen_cfg["backtest"].get("annualization_hours", 8760))

    variant_rows = []
    exit_rows = []
    winner_rows = []
    funnel_rows = []
    event_attrition_rows = []
    execution_attrition_rows = []
    context_rows = []
    regime_rows = []
    width_rows = []
    confirm_rows = []
    comparison_rows = []
    candidate_rows = []
    baseline_trades = None
    baseline_candidates = None
    baseline_variant_dir = None

    for name, overrides in VARIANTS:
        variant_cfg = copy.deepcopy(frozen_cfg)
        _deep_merge(variant_cfg, copy.deepcopy(overrides))
        variant_dir = out_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _save_yaml(variant_cfg, variant_dir / "config_variant.yaml")
        run_study(config_path=str(variant_dir / "config_variant.yaml"), outdir_override=str(variant_dir))

        trades = _build_trade_audit(variant_dir, variant_cfg)
        trades.to_csv(variant_dir / "trade_audit_false_break.csv", index=False)
        candidates = _posthoc_candidate_audit(variant_dir, variant_cfg)
        candidates.insert(0, "variant", name)
        candidates.to_csv(variant_dir / "candidate_audit_false_break.csv", index=False)

        variant_rows.append(_variant_summary_row(name, trades, candidates, annualization_hours))
        exr = _exit_reason_ratio(trades)
        if not exr.empty:
            exr.insert(0, "variant", name)
            exit_rows.append(exr)
        ws = _winner_sensitivity(trades)
        ws.insert(0, "variant", name)
        winner_rows.append(ws)
        funnel, event_attrition, execution_attrition = _capture_funnel(candidates, _tradable_bar_count(variant_dir))
        funnel.insert(0, "variant", name)
        funnel_rows.append(funnel)
        event_attrition.insert(0, "variant", name)
        event_attrition.insert(1, "semantics", ATTRITION_OVERLAP_NOTE)
        event_attrition_rows.append(event_attrition)
        execution_attrition.insert(0, "variant", name)
        execution_attrition.insert(1, "semantics", ATTRITION_OVERLAP_NOTE)
        execution_attrition_rows.append(execution_attrition)
        for bucket_col, target in [("context_bucket", context_rows), ("fb_regime", regime_rows), ("box_width_bucket", width_rows), ("confirm_count_bucket", confirm_rows)]:
            table = _capture_rate_table(candidates, bucket_col)
            if not table.empty:
                table.insert(0, "variant", name)
                target.append(table)
        comp = _missed_vs_executed_comparison(candidates)
        if not comp.empty:
            comp.insert(0, "variant", name)
            comparison_rows.append(comp)
        candidate_rows.append(candidates)
        if name == "baseline_master_current":
            baseline_trades = trades.copy()
            baseline_candidates = candidates.copy()
            baseline_variant_dir = variant_dir

    variant_summary = pd.DataFrame(variant_rows)
    new_trade_rows = []
    enriched_rows = []
    for name, _ in VARIANTS:
        variant_dir = out_root / name
        cfg_variant = _load_yaml(variant_dir / "config_variant.yaml")
        trades = _build_trade_audit(variant_dir, cfg_variant)
        extras, new_audit = _baseline_trade_comparison(name, trades, baseline_trades, baseline_candidates)
        row = variant_summary.loc[variant_summary["variant"] == name].iloc[0].to_dict()
        row.update(extras)
        enriched_rows.append(row)
        if not new_audit.empty:
            new_trade_rows.append(new_audit)
    variant_summary = pd.DataFrame(enriched_rows).sort_values(["total_return", "sharpe"], ascending=False).reset_index(drop=True)

    variant_summary.to_csv(out_root / "variant_summary.csv", index=False)
    pd.concat(exit_rows, ignore_index=True).to_csv(out_root / "exit_reason_by_variant.csv", index=False)
    pd.concat(winner_rows, ignore_index=True).to_csv(out_root / "winner_sensitivity_by_variant.csv", index=False)
    pd.concat(funnel_rows, ignore_index=True).to_csv(out_root / "tradable_funnel_by_variant.csv", index=False)
    pd.concat(event_attrition_rows, ignore_index=True).to_csv(out_root / "event_attrition_by_variant.csv", index=False)
    pd.concat(execution_attrition_rows, ignore_index=True).to_csv(out_root / "execution_attrition_by_variant.csv", index=False)
    pd.concat(context_rows, ignore_index=True).to_csv(out_root / "capture_rate_by_context_bucket.csv", index=False)
    pd.concat(regime_rows, ignore_index=True).to_csv(out_root / "capture_rate_by_regime_bucket.csv", index=False)
    pd.concat(width_rows, ignore_index=True).to_csv(out_root / "capture_rate_by_width_bucket.csv", index=False)
    pd.concat(confirm_rows, ignore_index=True).to_csv(out_root / "capture_rate_by_confirm_count.csv", index=False)
    pd.concat(comparison_rows, ignore_index=True).to_csv(out_root / "missed_vs_executed_feature_comparison.csv", index=False)
    pd.concat(candidate_rows, ignore_index=True).to_csv(out_root / "missed_candidate_posthoc_quality.csv", index=False)
    (pd.concat(new_trade_rows, ignore_index=True) if new_trade_rows else pd.DataFrame()).to_csv(out_root / "new_trade_audit.csv", index=False)

    capture_rate = variant_summary[["variant", "trades", "capture_rate_in_tradable", "execution_capture_rate", "price_stop_count", "time_stop_max_count", "state_stop_count", "baseline_trades_retained", "baseline_trades_lost", "new_trades_added", "missed_good_candidate_recovery_count", "newly_introduced_bad_trade_count", "new_price_stop_count"]].copy()
    capture_rate.to_csv(out_root / "capture_rate_by_variant.csv", index=False)

    best_variant = _best_variant_name(variant_summary)
    build_html(baseline_variant_dir, out_root / "baseline_tradable_window_bs_equity.html")
    build_png(baseline_variant_dir, out_root / "baseline_tradable_window_bs_equity.png")
    build_html(out_root / best_variant, out_root / "best_variant_tradable_window_bs_equity.html")
    build_png(out_root / best_variant, out_root / "best_variant_tradable_window_bs_equity.png")

    baseline_row = variant_summary.loc[variant_summary["variant"] == "baseline_master_current"].iloc[0]
    best_row = variant_summary.loc[variant_summary["variant"] == best_variant].iloc[0]
    funnel_all = pd.concat(funnel_rows, ignore_index=True)
    audit_lines = [
        "# Third Round Audit",
        "",
        "Scope: current merged master only. This round audits capture loss inside already tradable context windows and tests only narrow execution-local recovery branches.",
        "",
        "## Baseline capture funnel",
        f"- Tradable-context bars: {int(funnel_all.query('variant == \"baseline_master_current\" and stage == \"tradable_context_bars\"')['count'].iloc[0])}",
        f"- Raw false-break candidates: {int(funnel_all.query('variant == \"baseline_master_current\" and stage == \"raw_false_break_candidates\"')['count'].iloc[0])}",
        f"- Event-layer survivors: {int(funnel_all.query('variant == \"baseline_master_current\" and stage == \"event_layer_survivors\"')['count'].iloc[0])}",
        f"- Confirm-layer survivors: {int(funnel_all.query('variant == \"baseline_master_current\" and stage == \"confirm_layer_survivors\"')['count'].iloc[0])}",
        f"- Execution-layer survivors: {int(funnel_all.query('variant == \"baseline_master_current\" and stage == \"execution_layer_survivors\"')['count'].iloc[0])}",
        f"- Final trades: {int(funnel_all.query('variant == \"baseline_master_current\" and stage == \"final_trades\"')['count'].iloc[0])}",
        "",
        "## Audit conclusions",
        "- Tradable windows are present but underfilled; the system is selective inside them.",
        "- The raw no-reentry pool remains mostly low-quality. The narrower recoverable subset sits in confirmation misses and low-confidence execution misses.",
        f"- Best in-sample third-round recovery variant: `{best_variant}`.",
        f"- Baseline total return: {baseline_row['total_return']:.2%}, Sharpe: {baseline_row['sharpe']:.3f}, MaxDD: {baseline_row['max_drawdown']:.2%}.",
        f"- Best variant total return: {best_row['total_return']:.2%}, Sharpe: {best_row['sharpe']:.3f}, MaxDD: {best_row['max_drawdown']:.2%}.",
        "",
        "## Attrition semantics",
        f"- {ATTRITION_OVERLAP_NOTE}",
    ]
    _write_markdown(out_root / "THIRD_ROUND_AUDIT.md", audit_lines)

    _write_markdown(out_root / "THIRD_ROUND_CHANGES.md", [
        "# Third Round Changes",
        "",
        "Minimal code changes in this round:",
        "- Added default-off execution-level recovery switches for two narrow miss types.",
        "- Added third-round capture/recovery research runner and outputs.",
        "- Left regime backbone, warmup exclusion, bad-width filter, TP, and risk logic unchanged.",
        "",
        "Recovery branches added:",
        "- `capture_recovery_exec_conf_low_*`: only for raw false-break signals blocked solely by low event confidence.",
        "- `capture_recovery_confirm_near_miss_*`: only for reentry candidates with exactly one base confirm in strong approved context.",
        "",
        "Both branches remain optional and default-off in `config.yaml`.",
    ])

    result_lines = ["# Third Round Experiment Results", "", "## Summary"]
    for row in variant_summary.itertuples(index=False):
        result_lines.append(f"- `{row.variant}`: trades={row.trades}, total_return={row.total_return:.2%}, sharpe={row.sharpe:.3f}, max_dd={row.max_drawdown:.2%}, price_stop_count={row.price_stop_count}, time_stop_max_count={row.time_stop_max_count}, retained={row.baseline_trades_retained}, new={row.new_trades_added}")
    result_lines.extend(["", "## Best variant", f"- Selected best variant: `{best_variant}`", f"- Missed-good candidate recoveries: {int(best_row['missed_good_candidate_recovery_count'])}", f"- Newly introduced bad trades: {int(best_row['newly_introduced_bad_trade_count'])}", f"- New price-stop trades: {int(best_row['new_price_stop_count'])}", "", "## Attrition semantics", f"- {ATTRITION_OVERLAP_NOTE}"])
    _write_markdown(out_root / "THIRD_ROUND_EXPERIMENT_RESULTS.md", result_lines)

    print(f"Saved outputs to: {out_root}")
    print(f"Best variant: {best_variant}")


if __name__ == "__main__":
    main()
