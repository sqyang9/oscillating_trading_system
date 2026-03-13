from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from backtest import BacktestConfig, run_backtest, save_backtest_tables, write_backtest_report
from event_first import BoxInitiationConfig, FalseBreakConfig, detect_box_initiation_events
from execution import ExecutionConfig, align_4h_to_1h, export_execution_tables, run_execution_engines
from local_box import (
    LocalBoxConfig,
    attach_gap_segments,
    build_local_boxes,
    export_local_boxes,
    load_ohlcv_csv,
    resample_5m_to_1h,
)
from risk import RiskConfig, apply_risk_layer, compute_atr, export_risk_tables


CLOSED_INPUT_5M = "btc_data/closed/btc_usdt_swap_5m_closed.csv"
CLOSED_INPUT_4H = "btc_data/closed/btc_usdt_swap_4h_closed.csv"

WIDTH_BINS = [0.0, 0.005, 0.010, 0.015, 0.025, 0.040, 0.080, np.inf]
WIDTH_LABELS = ["<=0.5%", "0.5-1.0%", "1.0-1.5%", "1.5-2.5%", "2.5-4.0%", "4.0-8.0%", ">8.0%"]
ATR_WIDTH_BINS = [0.0, 0.75, 1.25, 1.75, 2.5, 4.0, np.inf]
ATR_WIDTH_LABELS = ["<=0.75x", "0.75-1.25x", "1.25-1.75x", "1.75-2.5x", "2.5-4.0x", ">4.0x"]

SESSIONS = {
    "asia": list(range(0, 8)),
    "europe": list(range(8, 16)),
    "us": list(range(16, 24)),
}


FROZEN_LAYER12_CONTROL_CONFIG: dict[str, Any] = {
    "data": {
        "symbol": "BTC/USDT",
        "input_5m": CLOSED_INPUT_5M,
        "input_4h": CLOSED_INPUT_4H,
        "use_5m_for_1h": True,
        "output_dir": "outputs/layer12_structural_research/control",
    },
    "local_box": {
        "rolling_window": 72,
        "min_periods": 24,
        "lower_quantile": 0.20,
        "upper_quantile": 0.80,
        "pivot_left": 2,
        "pivot_right": 2,
        "pivot_memory": 8,
        "touch_tolerance": 0.12,
        "repeated_test_window": 24,
        "repeated_test_min_gap": 2,
        "method_weights": {"quantile": 0.50, "pivot": 0.30, "repeated_test": 0.20},
        "min_width_pct": 0.001,
        "max_width_pct": 0.08,
        "stable_change_window": 12,
        "drift_window": 24,
        "drift_threshold": 0.35,
        "squeeze_width_pct": 0.01,
        "min_methods_each_side": 1,
    },
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
    "event_box_initiation": {
        "edge_tolerance_width_frac": 0.12,
        "min_confirmations": 2,
        "momentum_lookback": 3,
        "reversal_body_frac": 0.25,
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


@dataclass(frozen=True)
class ResearchVariant:
    name: str
    anchor_policy: str = "latest"
    reversal_policy: str = "baseline"


def deep_copy_config() -> dict[str, Any]:
    return copy.deepcopy(FROZEN_LAYER12_CONTROL_CONFIG)


def save_yaml(obj: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, allow_unicode=False)


def parse_times(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True)
    return out


def load_closed_base_frames(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars_5m = load_ohlcv_csv(cfg["data"]["input_5m"])
    bars_4h = load_ohlcv_csv(cfg["data"]["input_4h"])
    bars_1h = resample_5m_to_1h(bars_5m)
    bars_1h = attach_gap_segments(bars_1h, expected_freq="1h")
    bars_4h = attach_gap_segments(bars_4h, expected_freq="4h")

    lb_cfg = LocalBoxConfig(**cfg["local_box"])
    boxes_1h = build_local_boxes(bars_1h, lb_cfg)
    boxes_4h = build_local_boxes(bars_4h, lb_cfg)
    enriched_1h = align_4h_to_1h(boxes_1h, boxes_4h)
    enriched_1h["bar_index"] = np.arange(len(enriched_1h), dtype=int)
    return bars_1h, boxes_1h, boxes_4h, enriched_1h


def _momentum_series(close: pd.Series, lookback: int) -> pd.Series:
    ret = close.pct_change(fill_method=None)
    return ret.rolling(lookback, min_periods=1).mean()


def _relative_volume(volume: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    base = volume.shift(1).rolling(lookback, min_periods=min_periods).mean()
    return volume / base.replace(0.0, np.nan)


def _volume_confirmation_metrics(
    df: pd.DataFrame,
    rel_volume: pd.Series,
    breakout_idx: int,
    reentry_idx: int,
    cfg: FalseBreakConfig,
) -> tuple[float, float, float, bool]:
    breakout_vol_ratio = float(rel_volume.iat[breakout_idx]) if breakout_idx >= 0 else np.nan
    reentry_vol_ratio = float(rel_volume.iat[reentry_idx]) if reentry_idx >= 0 else np.nan
    breakout_volume = float(df.at[breakout_idx, "volume"]) if breakout_idx >= 0 else np.nan
    reentry_volume = float(df.at[reentry_idx, "volume"]) if reentry_idx >= 0 else np.nan
    if np.isfinite(breakout_volume) and breakout_volume > 0.0 and np.isfinite(reentry_volume):
        reentry_vs_breakout = reentry_volume / breakout_volume
    else:
        reentry_vs_breakout = np.nan

    breakout_ok = np.isfinite(breakout_vol_ratio) and breakout_vol_ratio <= cfg.breakout_volume_max_ratio
    reentry_ok = (
        np.isfinite(reentry_vol_ratio) and reentry_vol_ratio >= cfg.reentry_volume_min_ratio
    ) or (
        np.isfinite(reentry_vs_breakout) and reentry_vs_breakout >= cfg.reentry_vs_breakout_min_ratio
    )
    return breakout_vol_ratio, reentry_vol_ratio, reentry_vs_breakout, bool(breakout_ok and reentry_ok)


def build_reversal_features(df: pd.DataFrame, wick_ratio: float) -> pd.DataFrame:
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_body_high = pd.concat([prev_open, prev_close], axis=1).max(axis=1)
    prev_body_low = pd.concat([prev_open, prev_close], axis=1).min(axis=1)

    wick_short = ((df["close"] < df["open"]) & (upper_wick >= wick_ratio)).fillna(False)
    wick_long = ((df["close"] > df["open"]) & (lower_wick >= wick_ratio)).fillna(False)
    engulf_short = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    ).fillna(False)
    engulf_long = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    ).fillna(False)
    outside_inside_short = (
        (prev_high >= df["high"])
        & (prev_low <= df["low"])
        & (df["close"] < df["open"])
        & (df["close"] <= df["box_upper_edge"])
    ).fillna(False)
    outside_inside_long = (
        (prev_high >= df["high"])
        & (prev_low <= df["low"])
        & (df["close"] > df["open"])
        & (df["close"] >= df["box_lower_edge"])
    ).fillna(False)
    body_reclaim_short = ((df["close"] <= prev_body_low) & (df["close"] <= df["box_upper_edge"])).fillna(False)
    body_reclaim_long = ((df["close"] >= prev_body_high) & (df["close"] >= df["box_lower_edge"])).fillna(False)
    prev_reject_short = (
        (upper_wick.shift(1) >= wick_ratio)
        & (prev_high > df["box_upper_edge"].shift(1))
        & (df["close"] <= df["box_upper_edge"])
        & (df["close"] < df["open"])
    ).fillna(False)
    prev_reject_long = (
        (lower_wick.shift(1) >= wick_ratio)
        & (prev_low < df["box_lower_edge"].shift(1))
        & (df["close"] >= df["box_lower_edge"])
        & (df["close"] > df["open"])
    ).fillna(False)

    out = pd.DataFrame(index=df.index)
    out["rev_wick_short"] = wick_short
    out["rev_wick_long"] = wick_long
    out["rev_engulf_short"] = engulf_short
    out["rev_engulf_long"] = engulf_long
    out["rev_outside_inside_short"] = outside_inside_short
    out["rev_outside_inside_long"] = outside_inside_long
    out["rev_body_reclaim_short"] = body_reclaim_short
    out["rev_body_reclaim_long"] = body_reclaim_long
    out["rev_prev_reject_short"] = prev_reject_short
    out["rev_prev_reject_long"] = prev_reject_long
    out["rev_composite_score_short"] = (
        wick_short.astype(int)
        + engulf_short.astype(int)
        + outside_inside_short.astype(int)
        + body_reclaim_short.astype(int)
        + prev_reject_short.astype(int)
    )
    out["rev_composite_score_long"] = (
        wick_long.astype(int)
        + engulf_long.astype(int)
        + outside_inside_long.astype(int)
        + body_reclaim_long.astype(int)
        + prev_reject_long.astype(int)
    )
    return out


def _research_reversal_match(features: pd.DataFrame, idx: int, side: str, policy: str) -> bool:
    if side == "short":
        wick = bool(features.at[idx, "rev_wick_short"])
        engulf = bool(features.at[idx, "rev_engulf_short"])
        reclaim = bool(features.at[idx, "rev_body_reclaim_short"])
        prev_reject = bool(features.at[idx, "rev_prev_reject_short"])
        score = int(features.at[idx, "rev_composite_score_short"])
    else:
        wick = bool(features.at[idx, "rev_wick_long"])
        engulf = bool(features.at[idx, "rev_engulf_long"])
        reclaim = bool(features.at[idx, "rev_body_reclaim_long"])
        prev_reject = bool(features.at[idx, "rev_prev_reject_long"])
        score = int(features.at[idx, "rev_composite_score_long"])
    if policy == "baseline":
        return wick
    if policy == "composite_narrow":
        return bool(wick or engulf or (reclaim and prev_reject) or score >= 2)
    raise ValueError(f"Unsupported reversal policy: {policy}")

def detect_false_break_events_research(
    df: pd.DataFrame,
    cfg: FalseBreakConfig,
    anchor_policy: str = "latest",
    reversal_policy: str = "baseline",
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    features = build_reversal_features(out, cfg.reversal_wick_ratio)
    n = len(out)

    signal = np.zeros(n, dtype=int)
    confidence = np.zeros(n, dtype=float)
    side = np.full(n, "none", dtype=object)
    confirm_count = np.zeros(n, dtype=int)
    base_confirm_count = np.zeros(n, dtype=int)
    overshoot_up = np.zeros(n, dtype=bool)
    overshoot_dn = np.zeros(n, dtype=bool)
    reenter_up = np.zeros(n, dtype=bool)
    reenter_dn = np.zeros(n, dtype=bool)
    volume_confirmation = np.zeros(n, dtype=bool)
    volume_blocked = np.zeros(n, dtype=bool)
    breakout_vol_ratio = np.full(n, np.nan)
    reentry_vol_ratio = np.full(n, np.nan)
    reentry_vs_breakout = np.full(n, np.nan)
    breakout_anchor_idx = np.full(n, -1, dtype=int)
    breakout_age_bars = np.full(n, np.nan)
    confirm_c1 = np.zeros(n, dtype=bool)
    confirm_c2 = np.zeros(n, dtype=bool)
    confirm_c3 = np.zeros(n, dtype=bool)
    confirm_c4 = np.zeros(n, dtype=bool)

    mom = _momentum_series(out["close"], cfg.momentum_lookback)
    mom_turn_down = (mom < 0.0) & (mom.shift(1) >= 0.0)
    mom_turn_up = (mom > 0.0) & (mom.shift(1) <= 0.0)
    volume_min_periods = min(max(int(cfg.volume_min_periods), 1), max(int(cfg.volume_lookback), 1))
    rel_volume = _relative_volume(out["volume"], max(int(cfg.volume_lookback), 1), volume_min_periods)
    volume_mode = cfg.normalized_volume_confirmation_mode()

    p_short = -1
    p_long = -1
    for i in range(n):
        if i > 0 and out.at[i, "segment_id"] != out.at[i - 1, "segment_id"]:
            p_short = -1
            p_long = -1

        width = out.at[i, "box_width"]
        upper = out.at[i, "box_upper_edge"]
        lower = out.at[i, "box_lower_edge"]
        if not out.at[i, "box_valid"] or not np.isfinite(width) or width <= 0.0:
            p_short = -1
            p_long = -1
            continue

        is_up_overshoot = bool(out.at[i, "high"] > (upper + cfg.overshoot_width_mult * width))
        is_dn_overshoot = bool(out.at[i, "low"] < (lower - cfg.overshoot_width_mult * width))
        if is_up_overshoot:
            overshoot_up[i] = True
            if anchor_policy == "latest" or p_short < 0:
                p_short = i
        if is_dn_overshoot:
            overshoot_dn[i] = True
            if anchor_policy == "latest" or p_long < 0:
                p_long = i

        if p_short >= 0:
            age = i - p_short
            if age > cfg.reentry_window or out.at[i, "segment_id"] != out.at[p_short, "segment_id"]:
                p_short = -1
            else:
                reenter = bool(out.at[i, "close"] <= upper)
                if reenter:
                    reenter_up[i] = True
                    c1 = bool(out.at[i, "pivot_upper_confirm"] or out.at[i, "touch_upper"] or out.at[i, "repeated_upper_count"] >= 2)
                    c2 = _research_reversal_match(features, i, "short", reversal_policy)
                    c3 = bool(mom_turn_down.iat[i])
                    c4 = bool(out.at[i, "close"] <= out.at[i, "box_midline"] + cfg.recapture_to_mid_frac * width)
                    confirm_c1[i] = c1
                    confirm_c2[i] = c2
                    confirm_c3[i] = c3
                    confirm_c4[i] = c4
                    base_csum = int(c1) + int(c2) + int(c3) + int(c4)
                    base_confirm_count[i] = base_csum
                    breakout_anchor_idx[i] = p_short
                    breakout_age_bars[i] = age
                    bvr, rvr, rvb, volume_ok = _volume_confirmation_metrics(out, rel_volume, p_short, i, cfg)
                    breakout_vol_ratio[i] = bvr
                    reentry_vol_ratio[i] = rvr
                    reentry_vs_breakout[i] = rvb
                    volume_confirmation[i] = volume_ok
                    total_csum = base_csum
                    total_possible = 4
                    if cfg.volume_confirmation_enabled and volume_mode == "confirm":
                        total_csum += int(volume_ok)
                        total_possible = 5
                    can_fire = base_csum >= cfg.min_confirmations
                    if cfg.volume_confirmation_enabled and volume_mode == "require" and can_fire and not volume_ok:
                        volume_blocked[i] = True
                        can_fire = False
                    if can_fire:
                        signal[i] = -1
                        side[i] = "short"
                        confirm_count[i] = total_csum
                        confidence[i] = min(1.0, (total_csum / total_possible) * out.at[i, "box_confidence"])
                    p_short = -1

        if p_long >= 0:
            age = i - p_long
            if age > cfg.reentry_window or out.at[i, "segment_id"] != out.at[p_long, "segment_id"]:
                p_long = -1
            else:
                reenter = bool(out.at[i, "close"] >= lower)
                if reenter:
                    reenter_dn[i] = True
                    c1 = bool(out.at[i, "pivot_lower_confirm"] or out.at[i, "touch_lower"] or out.at[i, "repeated_lower_count"] >= 2)
                    c2 = _research_reversal_match(features, i, "long", reversal_policy)
                    c3 = bool(mom_turn_up.iat[i])
                    c4 = bool(out.at[i, "close"] >= out.at[i, "box_midline"] - cfg.recapture_to_mid_frac * width)
                    confirm_c1[i] = c1
                    confirm_c2[i] = c2
                    confirm_c3[i] = c3
                    confirm_c4[i] = c4
                    base_csum = int(c1) + int(c2) + int(c3) + int(c4)
                    base_confirm_count[i] = base_csum
                    breakout_anchor_idx[i] = p_long
                    breakout_age_bars[i] = age
                    bvr, rvr, rvb, volume_ok = _volume_confirmation_metrics(out, rel_volume, p_long, i, cfg)
                    breakout_vol_ratio[i] = bvr
                    reentry_vol_ratio[i] = rvr
                    reentry_vs_breakout[i] = rvb
                    volume_confirmation[i] = volume_ok
                    total_csum = base_csum
                    total_possible = 4
                    if cfg.volume_confirmation_enabled and volume_mode == "confirm":
                        total_csum += int(volume_ok)
                        total_possible = 5
                    can_fire = base_csum >= cfg.min_confirmations and signal[i] == 0
                    if cfg.volume_confirmation_enabled and volume_mode == "require" and can_fire and not volume_ok:
                        volume_blocked[i] = True
                        can_fire = False
                    if can_fire:
                        signal[i] = 1
                        side[i] = "long"
                        confirm_count[i] = total_csum
                        confidence[i] = min(1.0, (total_csum / total_possible) * out.at[i, "box_confidence"])
                    p_long = -1

    out = out.join(features)
    out["fb_overshoot_up"] = overshoot_up
    out["fb_overshoot_down"] = overshoot_dn
    out["fb_reenter_from_up"] = reenter_up
    out["fb_reenter_from_down"] = reenter_dn
    out["fb_breakout_volume_ratio"] = breakout_vol_ratio
    out["fb_reentry_volume_ratio"] = reentry_vol_ratio
    out["fb_reentry_vs_breakout_ratio"] = reentry_vs_breakout
    out["fb_volume_confirmation"] = volume_confirmation
    out["fb_volume_blocked"] = volume_blocked
    out["fb_breakout_anchor_idx"] = breakout_anchor_idx
    out["fb_breakout_age_bars"] = breakout_age_bars
    out["fb_confirm_c1_structure"] = confirm_c1
    out["fb_confirm_c2_reversal"] = confirm_c2
    out["fb_confirm_c3_momentum"] = confirm_c3
    out["fb_confirm_c4_recapture"] = confirm_c4
    out["event_false_break_signal"] = signal
    out["event_false_break_side"] = side
    out["event_false_break_base_confirms"] = base_confirm_count
    out["event_false_break_confirms"] = confirm_count
    out["event_false_break_confidence"] = confidence
    out["fb_anchor_policy"] = anchor_policy
    out["fb_reversal_policy"] = reversal_policy
    return out


def build_event_tables_research(
    df: pd.DataFrame,
    false_break_cfg: FalseBreakConfig,
    initiation_cfg: BoxInitiationConfig,
    anchor_policy: str,
    reversal_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fb = detect_false_break_events_research(df, false_break_cfg, anchor_policy=anchor_policy, reversal_policy=reversal_policy)
    bi = detect_box_initiation_events(df, initiation_cfg)
    merged = df.copy().join(
        fb[
            [
                "rev_wick_short",
                "rev_wick_long",
                "rev_engulf_short",
                "rev_engulf_long",
                "rev_outside_inside_short",
                "rev_outside_inside_long",
                "rev_body_reclaim_short",
                "rev_body_reclaim_long",
                "rev_prev_reject_short",
                "rev_prev_reject_long",
                "rev_composite_score_short",
                "rev_composite_score_long",
                "fb_overshoot_up",
                "fb_overshoot_down",
                "fb_reenter_from_up",
                "fb_reenter_from_down",
                "fb_breakout_volume_ratio",
                "fb_reentry_volume_ratio",
                "fb_reentry_vs_breakout_ratio",
                "fb_volume_confirmation",
                "fb_volume_blocked",
                "fb_breakout_anchor_idx",
                "fb_breakout_age_bars",
                "fb_confirm_c1_structure",
                "fb_confirm_c2_reversal",
                "fb_confirm_c3_momentum",
                "fb_confirm_c4_recapture",
                "event_false_break_signal",
                "event_false_break_side",
                "event_false_break_base_confirms",
                "event_false_break_confirms",
                "event_false_break_confidence",
                "fb_anchor_policy",
                "fb_reversal_policy",
            ]
        ]
    )
    merged = merged.join(
        bi[
            [
                "event_box_init_signal",
                "event_box_init_side",
                "event_box_init_confirms",
                "event_box_init_confidence",
                "box_edge_test_upper",
                "box_edge_test_lower",
            ]
        ]
    )
    return merged, fb, bi


def run_research_variant(
    cfg: dict[str, Any],
    boxes_1h: pd.DataFrame,
    boxes_4h: pd.DataFrame,
    enriched_1h: pd.DataFrame,
    out_dir: Path,
    variant: ResearchVariant,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    variant_cfg = copy.deepcopy(cfg)
    variant_cfg["data"]["output_dir"] = str(out_dir).replace("\\", "/")
    save_yaml(variant_cfg, out_dir / "config_variant.yaml")

    export_local_boxes(boxes_1h, str(out_dir / "local_box_1h.csv"))
    export_local_boxes(boxes_4h, str(out_dir / "local_box_4h.csv"))

    fb_cfg = FalseBreakConfig(**variant_cfg["event_false_break"])
    bi_cfg = BoxInitiationConfig(**variant_cfg["event_box_initiation"])
    merged, events_fb, events_bi = build_event_tables_research(
        enriched_1h,
        fb_cfg,
        bi_cfg,
        anchor_policy=variant.anchor_policy,
        reversal_policy=variant.reversal_policy,
    )
    merged.to_csv(out_dir / "events_merged.csv", index=False)
    events_fb.to_csv(out_dir / "event_false_break.csv", index=False)
    events_bi.to_csv(out_dir / "event_box_initiation.csv", index=False)

    bt_cfg = BacktestConfig(**variant_cfg["backtest"])
    exec_cfg_dict = dict(variant_cfg["execution"])
    exec_cfg_dict["allow_states"] = tuple(exec_cfg_dict.get("allow_states", ["STABLE", "SQUEEZE"]))
    if exec_cfg_dict.get("expected_value_round_trip_cost_rate") is None:
        exec_cfg_dict["expected_value_round_trip_cost_rate"] = 2.0 * bt_cfg.leg_cost_rate()
    ex_cfg = ExecutionConfig(**exec_cfg_dict)
    execution_outputs = run_execution_engines(merged, ex_cfg)
    export_execution_tables(execution_outputs, str(out_dir))

    risk_cfg = RiskConfig(**variant_cfg["risk"])
    risk_outputs = apply_risk_layer(merged, execution_outputs, risk_cfg)
    export_risk_tables(risk_outputs, str(out_dir))

    backtest_outputs = run_backtest(risk_outputs["trades_false_break"], risk_outputs["trades_boundary"], bt_cfg)
    save_backtest_tables(backtest_outputs, str(out_dir))
    write_backtest_report(backtest_outputs["summary"], str(out_dir / "report_backtest.md"))

    return {
        "config": variant_cfg,
        "merged": merged,
        "events_fb": events_fb,
        "events_bi": events_bi,
        "execution": execution_outputs,
        "risk": risk_outputs,
        "backtest": backtest_outputs,
        "out_dir": out_dir,
        "variant": variant,
    }

def label_market_phase(ret_val: float, cfg: dict[str, Any]) -> str:
    if not np.isfinite(ret_val):
        return "warmup"
    ex = cfg["execution"]
    if ret_val <= ex["regime_down_low"]:
        return "deep_down"
    if ex["regime_down_low"] < ret_val <= ex["regime_down_high"]:
        return "down"
    if ex["regime_down_high"] < ret_val < ex["regime_up_low"]:
        return "flat"
    if ex["regime_up_low"] <= ret_val < ex["regime_up_high"]:
        return "up"
    return "strong_up"


def session_from_hour(hour: int) -> str:
    for label, hours in SESSIONS.items():
        if int(hour) in hours:
            return label
    return "other"


def add_common_bar_features(merged: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = merged.copy()
    risk_cfg = RiskConfig(**cfg["risk"])
    out["atr"] = compute_atr(out, risk_cfg.atr_window)
    out["box_width_atr_mult"] = out["box_width"] / out["atr"].replace(0.0, np.nan)
    out["market_phase_ret"] = out["close"].pct_change(cfg["execution"]["regime_lookback_hours"], fill_method=None)
    out["market_phase"] = out["market_phase_ret"].apply(lambda value: label_market_phase(float(value), cfg))
    out["hour_utc"] = out["timestamp"].dt.hour
    out["session"] = out["hour_utc"].apply(session_from_hour)
    out["volume_rel_20"] = _relative_volume(
        out["volume"],
        cfg["event_false_break"]["volume_lookback"],
        min(cfg["event_false_break"]["volume_min_periods"], cfg["event_false_break"]["volume_lookback"]),
    )
    out["volume_rank_24"] = out["volume"].rolling(24, min_periods=6).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )
    hour_medians = out.groupby("hour_utc")["volume"].transform("median")
    out["volume_hour_norm"] = out["volume"] / hour_medians.replace(0.0, np.nan)
    return out


def build_trade_audit(variant_result: dict[str, Any]) -> pd.DataFrame:
    trades = variant_result["backtest"]["trades_false_break"].copy()
    if trades.empty:
        return trades

    merged = add_common_bar_features(variant_result["merged"], variant_result["config"])
    execution = variant_result["execution"]["false_break"].copy()
    trades = parse_times(trades, ["entry_time", "exit_time"])
    execution = parse_times(execution, ["timestamp"])
    merged = parse_times(merged, ["timestamp"])

    exec_keep = [
        "timestamp",
        "event_confidence",
        "fb_regime",
        "fb_regime_ret",
        "event_base_confirms",
        "entry_warmup",
        "blocked_reason",
        "reward_to_cost_ratio",
    ]
    exec_keep = [col for col in exec_keep if col in execution.columns]
    execution = execution[exec_keep].rename(columns={"timestamp": "entry_time"})

    bar_keep = [
        "timestamp",
        "bar_index",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr",
        "box_midline",
        "box_upper_edge",
        "box_lower_edge",
        "box_width",
        "box_width_pct",
        "box_width_atr_mult",
        "market_phase",
        "market_phase_ret",
        "box_confidence",
        "method_agreement",
        "transition_risk",
        "fb_breakout_anchor_idx",
        "fb_breakout_age_bars",
        "fb_breakout_volume_ratio",
        "fb_reentry_volume_ratio",
        "fb_reentry_vs_breakout_ratio",
        "fb_confirm_c1_structure",
        "fb_confirm_c2_reversal",
        "fb_confirm_c3_momentum",
        "fb_confirm_c4_recapture",
        "rev_wick_short",
        "rev_wick_long",
        "rev_engulf_short",
        "rev_engulf_long",
        "rev_outside_inside_short",
        "rev_outside_inside_long",
        "rev_body_reclaim_short",
        "rev_body_reclaim_long",
        "rev_prev_reject_short",
        "rev_prev_reject_long",
        "rev_composite_score_short",
        "rev_composite_score_long",
        "volume_rel_20",
        "volume_rank_24",
        "volume_hour_norm",
        "session",
        "hour_utc",
    ]
    bar_keep = [col for col in bar_keep if col in merged.columns]
    entry_bars = merged[bar_keep].rename(columns={"timestamp": "entry_time"})

    audit = trades.merge(execution, on="entry_time", how="left", suffixes=("", "_exec"))
    audit = audit.merge(entry_bars, on="entry_time", how="left", suffixes=("", "_bar"))
    audit["base_confirms"] = audit["event_base_confirms"].fillna(audit.get("entry_base_confirms", np.nan))
    audit["box_width_bucket"] = pd.cut(audit["box_width_pct"], bins=WIDTH_BINS, labels=WIDTH_LABELS, include_lowest=True, right=True)
    audit["atr_width_bucket"] = pd.cut(audit["box_width_atr_mult"], bins=ATR_WIDTH_BINS, labels=ATR_WIDTH_LABELS, include_lowest=True, right=True)
    audit["edge_distance"] = np.where(audit["side"] > 0, audit["entry_price"] - audit["box_lower_edge"], audit["box_upper_edge"] - audit["entry_price"])
    audit["edge_distance_pct"] = audit["edge_distance"] / audit["entry_price"].replace(0.0, np.nan)
    audit["edge_distance_atr"] = audit["edge_distance"] / audit["atr"].replace(0.0, np.nan)
    audit["reentry_distance"] = audit["edge_distance"]
    audit["reentry_distance_atr"] = audit["reentry_distance"] / audit["atr"].replace(0.0, np.nan)
    audit["reentry_distance_pct"] = audit["reentry_distance"] / audit["entry_price"].replace(0.0, np.nan)

    audit["overshoot_distance"] = np.nan
    anchor_map = merged.set_index("bar_index")
    for row in audit.itertuples():
        anchor_idx = int(row.fb_breakout_anchor_idx) if pd.notna(row.fb_breakout_anchor_idx) else -1
        if anchor_idx < 0 or anchor_idx not in anchor_map.index:
            continue
        anchor_row = anchor_map.loc[anchor_idx]
        if row.side > 0:
            distance = float(anchor_row["box_lower_edge"] - anchor_row["low"])
        else:
            distance = float(anchor_row["high"] - anchor_row["box_upper_edge"])
        audit.at[row.Index, "overshoot_distance"] = distance

    audit["overshoot_distance_pct"] = audit["overshoot_distance"] / audit["entry_price"].replace(0.0, np.nan)
    audit["overshoot_distance_atr"] = audit["overshoot_distance"] / audit["atr"].replace(0.0, np.nan)
    audit["confirm_richness"] = (
        audit["fb_confirm_c1_structure"].fillna(False).astype(int)
        + audit["fb_confirm_c2_reversal"].fillna(False).astype(int)
        + audit["fb_confirm_c3_momentum"].fillna(False).astype(int)
        + audit["fb_confirm_c4_recapture"].fillna(False).astype(int)
    )
    audit["reversal_composite_score"] = np.where(audit["side"] > 0, audit["rev_composite_score_long"], audit["rev_composite_score_short"])
    audit["wick_reversal_used"] = np.where(audit["side"] > 0, audit["rev_wick_long"], audit["rev_wick_short"]).astype(bool)
    audit["engulfing_reversal"] = np.where(audit["side"] > 0, audit["rev_engulf_long"], audit["rev_engulf_short"]).astype(bool)
    audit["body_reclaim_reversal"] = np.where(audit["side"] > 0, audit["rev_body_reclaim_long"], audit["rev_body_reclaim_short"]).astype(bool)
    audit["prev_reject_combo"] = np.where(audit["side"] > 0, audit["rev_prev_reject_long"], audit["rev_prev_reject_short"]).astype(bool)
    return audit


def add_forward_entry_quality(audit: pd.DataFrame, merged: pd.DataFrame, horizons: tuple[int, ...] = (6, 12, 24)) -> pd.DataFrame:
    out = audit.copy()
    bars = merged.copy().reset_index(drop=True)
    ts_to_idx = {ts: idx for idx, ts in enumerate(bars["timestamp"])}
    risk_cfg = RiskConfig(**FROZEN_LAYER12_CONTROL_CONFIG["risk"])
    for horizon in horizons:
        out[f"entry_precision_{horizon}h"] = np.nan
        out[f"forward_mfe_{horizon}h"] = np.nan
        out[f"forward_mae_{horizon}h"] = np.nan

    for row in out.itertuples():
        entry_idx = ts_to_idx.get(row.entry_time)
        if entry_idx is None:
            continue
        stop_dist = max(
            risk_cfg.min_stop_pct * float(row.entry_price),
            risk_cfg.atr_mult * float(row.atr) if pd.notna(row.atr) else 0.0,
            risk_cfg.box_stop_mult * float(row.box_width) if pd.notna(row.box_width) else 0.0,
        )
        out.at[row.Index, "entry_stop_distance"] = stop_dist
        for horizon in horizons:
            window = bars.iloc[entry_idx + 1 : entry_idx + horizon + 1]
            if window.empty:
                continue
            hi = pd.to_numeric(window["high"], errors="coerce").max()
            lo = pd.to_numeric(window["low"], errors="coerce").min()
            rng = hi - lo if np.isfinite(hi) and np.isfinite(lo) else np.nan
            if row.side > 0:
                precision = (float(row.entry_price) - lo) / rng if np.isfinite(rng) and rng > 0 else np.nan
                mfe = max((hi / row.entry_price - 1.0), 0.0) if np.isfinite(hi) else np.nan
                mae = max((1.0 - lo / row.entry_price), 0.0) if np.isfinite(lo) else np.nan
            else:
                precision = (hi - float(row.entry_price)) / rng if np.isfinite(rng) and rng > 0 else np.nan
                mfe = max((1.0 - lo / row.entry_price), 0.0) if np.isfinite(lo) else np.nan
                mae = max((hi / row.entry_price - 1.0), 0.0) if np.isfinite(hi) else np.nan
            out.at[row.Index, f"entry_precision_{horizon}h"] = precision
            out.at[row.Index, f"forward_mfe_{horizon}h"] = mfe
            out.at[row.Index, f"forward_mae_{horizon}h"] = mae
    return out

def extract_overshoot_episodes(merged: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    df = add_common_bar_features(merged, cfg).reset_index(drop=True)
    fb_cfg = FalseBreakConfig(**cfg["event_false_break"])
    rows: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any] | None] = {"short": None, "long": None}
    episode_id = 0

    def finalize(side_name: str, status: str, now_idx: int | None = None) -> None:
        nonlocal episode_id
        episode = active[side_name]
        if episode is None:
            return
        row = dict(episode)
        row["episode_id"] = episode_id
        row["status"] = status
        row["final_bar_idx"] = now_idx if now_idx is not None else episode["last_anchor_idx"]
        rows.append(row)
        episode_id += 1
        active[side_name] = None

    for i in range(len(df)):
        if i > 0 and df.at[i, "segment_id"] != df.at[i - 1, "segment_id"]:
            finalize("short", "segment_reset", i)
            finalize("long", "segment_reset", i)

        width = df.at[i, "box_width"]
        if not df.at[i, "box_valid"] or not np.isfinite(width) or width <= 0.0:
            finalize("short", "box_invalid", i)
            finalize("long", "box_invalid", i)
            continue

        upper = df.at[i, "box_upper_edge"]
        lower = df.at[i, "box_lower_edge"]
        up_overshoot = bool(df.at[i, "high"] > upper + fb_cfg.overshoot_width_mult * width)
        dn_overshoot = bool(df.at[i, "low"] < lower - fb_cfg.overshoot_width_mult * width)

        if up_overshoot:
            if active["short"] is None:
                active["short"] = {
                    "side": "short",
                    "segment_id": int(df.at[i, "segment_id"]),
                    "first_anchor_idx": i,
                    "last_anchor_idx": i,
                    "first_anchor_time": df.at[i, "timestamp"],
                    "last_anchor_time": df.at[i, "timestamp"],
                    "overshoot_bars": 1,
                    "first_overshoot_distance": float(df.at[i, "high"] - upper),
                    "last_overshoot_distance": float(df.at[i, "high"] - upper),
                    "first_breakout_volume_ratio": float(df.at[i, "volume_rel_20"]),
                    "last_breakout_volume_ratio": float(df.at[i, "volume_rel_20"]),
                    "first_breakout_momentum": float(df.at[i, "close"] / df.at[max(i - 1, 0), "close"] - 1.0) if i > 0 else np.nan,
                    "last_breakout_momentum": float(df.at[i, "close"] / df.at[max(i - 1, 0), "close"] - 1.0) if i > 0 else np.nan,
                    "reentry_idx": -1,
                    "reentry_time": pd.NaT,
                }
            else:
                active["short"]["last_anchor_idx"] = i
                active["short"]["last_anchor_time"] = df.at[i, "timestamp"]
                active["short"]["overshoot_bars"] += 1
                active["short"]["last_overshoot_distance"] = float(df.at[i, "high"] - upper)
                active["short"]["last_breakout_volume_ratio"] = float(df.at[i, "volume_rel_20"])
                active["short"]["last_breakout_momentum"] = float(df.at[i, "close"] / df.at[max(i - 1, 0), "close"] - 1.0) if i > 0 else np.nan

        if dn_overshoot:
            if active["long"] is None:
                active["long"] = {
                    "side": "long",
                    "segment_id": int(df.at[i, "segment_id"]),
                    "first_anchor_idx": i,
                    "last_anchor_idx": i,
                    "first_anchor_time": df.at[i, "timestamp"],
                    "last_anchor_time": df.at[i, "timestamp"],
                    "overshoot_bars": 1,
                    "first_overshoot_distance": float(lower - df.at[i, "low"]),
                    "last_overshoot_distance": float(lower - df.at[i, "low"]),
                    "first_breakout_volume_ratio": float(df.at[i, "volume_rel_20"]),
                    "last_breakout_volume_ratio": float(df.at[i, "volume_rel_20"]),
                    "first_breakout_momentum": float(df.at[i, "close"] / df.at[max(i - 1, 0), "close"] - 1.0) if i > 0 else np.nan,
                    "last_breakout_momentum": float(df.at[i, "close"] / df.at[max(i - 1, 0), "close"] - 1.0) if i > 0 else np.nan,
                    "reentry_idx": -1,
                    "reentry_time": pd.NaT,
                }
            else:
                active["long"]["last_anchor_idx"] = i
                active["long"]["last_anchor_time"] = df.at[i, "timestamp"]
                active["long"]["overshoot_bars"] += 1
                active["long"]["last_overshoot_distance"] = float(lower - df.at[i, "low"])
                active["long"]["last_breakout_volume_ratio"] = float(df.at[i, "volume_rel_20"])
                active["long"]["last_breakout_momentum"] = float(df.at[i, "close"] / df.at[max(i - 1, 0), "close"] - 1.0) if i > 0 else np.nan

        for side_name in ("short", "long"):
            episode = active[side_name]
            if episode is None:
                continue
            first_age = i - int(episode["first_anchor_idx"])
            last_age = i - int(episode["last_anchor_idx"])
            first_alive = first_age <= fb_cfg.reentry_window
            last_alive = last_age <= fb_cfg.reentry_window
            reentry = bool(df.at[i, "close"] <= upper) if side_name == "short" else bool(df.at[i, "close"] >= lower)
            if reentry and i > int(episode["last_anchor_idx"]):
                episode["reentry_idx"] = i
                episode["reentry_time"] = df.at[i, "timestamp"]
                episode["first_anchor_age_at_reentry"] = first_age
                episode["last_anchor_age_at_reentry"] = last_age
                episode["first_within_window"] = first_alive
                episode["last_within_window"] = last_alive
                episode["anchor_shift_bars"] = int(episode["last_anchor_idx"] - episode["first_anchor_idx"])
                episode["reentry_volume_ratio"] = float(df.at[i, "volume_rel_20"])
                episode["reentry_box_width_pct"] = float(df.at[i, "box_width_pct"])
                episode["reentry_box_width_atr_mult"] = float(df.at[i, "box_width_atr_mult"])
                episode["reentry_market_phase"] = str(df.at[i, "market_phase"])
                episode["reentry_hour_utc"] = int(df.at[i, "hour_utc"])
                episode["reentry_session"] = str(df.at[i, "session"])
                finalize(side_name, "reentry", i)
                continue
            if not last_alive:
                episode["anchor_shift_bars"] = int(episode["last_anchor_idx"] - episode["first_anchor_idx"])
                episode["first_within_window"] = False
                episode["last_within_window"] = False
                finalize(side_name, "timeout", i)

    finalize("short", "end_of_data")
    finalize("long", "end_of_data")
    return pd.DataFrame(rows)


def candidate_quality_label(
    bars: pd.DataFrame,
    entry_idx: int,
    side: int,
    entry_price: float,
    target_price: float,
    stop_distance: float,
    max_hold_bars: int,
) -> str:
    window = bars.iloc[entry_idx + 1 : entry_idx + max_hold_bars + 1]
    if window.empty or not np.isfinite(entry_price):
        return "no_window"
    for row in window.itertuples():
        if side > 0:
            stop_hit = np.isfinite(row.low) and row.low <= entry_price - stop_distance
            target_hit = np.isfinite(row.high) and row.high >= target_price
        else:
            stop_hit = np.isfinite(row.high) and row.high >= entry_price + stop_distance
            target_hit = np.isfinite(row.low) and row.low <= target_price
        if stop_hit and target_hit:
            return "stop_first_same_bar"
        if stop_hit:
            return "stop_first"
        if target_hit:
            return "target_first"
    final_close = float(window.iloc[-1]["close"])
    ret = side * (final_close / entry_price - 1.0)
    return "positive_hold" if ret > 0 else "negative_hold"


def effect_size(pos: pd.Series, neg: pd.Series) -> float:
    pos = pd.to_numeric(pos, errors="coerce").dropna()
    neg = pd.to_numeric(neg, errors="coerce").dropna()
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    if pooled <= 0.0 or not np.isfinite(pooled):
        return np.nan
    return float((pos.mean() - neg.mean()) / pooled)


def quantile_hit_rate_spread(feature: pd.Series, label: pd.Series, buckets: int = 3) -> float:
    data = pd.DataFrame({"feature": pd.to_numeric(feature, errors="coerce"), "label": label.astype(float)}).dropna()
    if len(data) < buckets * 3:
        return np.nan
    try:
        data["bucket"] = pd.qcut(data["feature"], buckets, duplicates="drop")
    except ValueError:
        return np.nan
    grouped = data.groupby("bucket", observed=False)["label"].mean()
    if grouped.empty:
        return np.nan
    return float(grouped.iloc[-1] - grouped.iloc[0])


def binary_feature_diagnostics(df: pd.DataFrame, feature_cols: list[str], label_col: str, positive_value: Any = True) -> pd.DataFrame:
    rows = []
    positive_mask = df[label_col] == positive_value
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        feature_series = pd.to_numeric(df[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "label": label_col,
                "positive_value": str(positive_value),
                "n": int(feature_series.notna().sum()),
                "positive_mean": float(pd.to_numeric(df.loc[positive_mask, feature], errors="coerce").mean()),
                "negative_mean": float(pd.to_numeric(df.loc[~positive_mask, feature], errors="coerce").mean()),
                "effect_size": effect_size(df.loc[positive_mask, feature], df.loc[~positive_mask, feature]),
                "spearman_corr": float(feature_series.corr(positive_mask.astype(float), method="spearman")) if feature_series.notna().sum() >= 3 else np.nan,
                "quantile_hit_rate_spread": quantile_hit_rate_spread(feature_series, positive_mask.astype(float)),
            }
        )
    return pd.DataFrame(rows)


def equity_curve_from_returns(df: pd.DataFrame, return_col: str = "net_return") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "peak", "drawdown"])
    equity = 1.0
    peak = 1.0
    rows = []
    for row in df.sort_values("exit_time").itertuples(index=False):
        equity *= 1.0 + float(getattr(row, return_col))
        peak = max(peak, equity)
        rows.append({"timestamp": row.exit_time, "equity": equity, "peak": peak, "drawdown": equity / peak - 1.0})
    return pd.DataFrame(rows)


def summarize_returns(df: pd.DataFrame, annualization_hours: int, return_col: str = "net_return") -> dict[str, float]:
    if df.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "avg_holding_bars": 0.0,
        }
    curve = equity_curve_from_returns(df, return_col=return_col)
    std = float(df[return_col].std(ddof=1)) if len(df) >= 2 else 0.0
    if std > 0.0 and np.isfinite(std):
        mean_hold = max(float(df["hold_bars"].mean()), 1.0)
        sharpe = float(df[return_col].mean() / std * np.sqrt(annualization_hours / mean_hold))
    else:
        sharpe = 0.0
    return {
        "trades": int(len(df)),
        "win_rate": float((df[return_col] > 0).mean()),
        "avg_return": float(df[return_col].mean()),
        "total_return": float(curve["equity"].iloc[-1] - 1.0) if not curve.empty else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(curve["drawdown"].min()) if not curve.empty else 0.0,
        "avg_holding_bars": float(df["hold_bars"].mean()),
    }


def winner_sensitivity(trades: pd.DataFrame, annualization_hours: int) -> pd.DataFrame:
    rows = []
    ranked = trades.sort_values("net_return", ascending=False).reset_index(drop=True)
    for remove_top in [0, 3, 5, 10]:
        trimmed = ranked.iloc[remove_top:].copy() if remove_top > 0 else ranked.copy()
        rows.append({"remove_top_winners": remove_top, **summarize_returns(trimmed, annualization_hours)})
    return pd.DataFrame(rows)
