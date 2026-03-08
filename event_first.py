"""Event-first signal families for oscillating range trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FalseBreakConfig:
    overshoot_width_mult: float = 0.08
    reentry_window: int = 6
    min_confirmations: int = 2
    recapture_to_mid_frac: float = 0.40
    momentum_lookback: int = 3
    reversal_wick_ratio: float = 0.45
    volume_confirmation_enabled: bool = False
    volume_confirmation_mode: str = "require"  # require | confirm
    volume_lookback: int = 20
    volume_min_periods: int = 10
    breakout_volume_max_ratio: float = 1.15
    reentry_volume_min_ratio: float = 0.95
    reentry_vs_breakout_min_ratio: float = 1.05

    def normalized_volume_confirmation_mode(self) -> str:
        mode = str(self.volume_confirmation_mode).strip().lower()
        if mode in {"require", "confirm"}:
            return mode
        return "require"


@dataclass(frozen=True)
class BoxInitiationConfig:
    edge_tolerance_width_frac: float = 0.12
    min_confirmations: int = 2
    momentum_lookback: int = 3
    reversal_body_frac: float = 0.25


def _require_event_columns(df: pd.DataFrame) -> None:
    need = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "segment_id",
        "box_upper_edge",
        "box_lower_edge",
        "box_midline",
        "box_width",
        "box_valid",
        "box_confidence",
        "touch_upper",
        "touch_lower",
        "pivot_upper_confirm",
        "pivot_lower_confirm",
        "repeated_upper_count",
        "repeated_lower_count",
    }
    miss = sorted(need - set(df.columns))
    if miss:
        raise ValueError(f"Missing event input columns: {miss}")


def _momentum_series(close: pd.Series, lookback: int) -> pd.Series:
    ret = close.pct_change(fill_method=None)
    mom = ret.rolling(lookback, min_periods=1).mean()
    return mom


def _reversal_candle_flags(df: pd.DataFrame, wick_ratio: float) -> tuple[pd.Series, pd.Series]:
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    short_flag = (df["close"] < df["open"]) & (upper_wick >= wick_ratio)
    long_flag = (df["close"] > df["open"]) & (lower_wick >= wick_ratio)
    return short_flag.fillna(False), long_flag.fillna(False)


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


def detect_false_break_events(df: pd.DataFrame, cfg: FalseBreakConfig) -> pd.DataFrame:
    """Detect upside/downside false-break reversals in a causal way."""
    _require_event_columns(df)
    out = df.copy().reset_index(drop=True)

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

    mom = _momentum_series(out["close"], cfg.momentum_lookback)
    mom_turn_down = (mom < 0.0) & (mom.shift(1) >= 0.0)
    mom_turn_up = (mom > 0.0) & (mom.shift(1) <= 0.0)
    rev_short, rev_long = _reversal_candle_flags(out, cfg.reversal_wick_ratio)

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

        if out.at[i, "high"] > (upper + cfg.overshoot_width_mult * width):
            overshoot_up[i] = True
            p_short = i

        if out.at[i, "low"] < (lower - cfg.overshoot_width_mult * width):
            overshoot_dn[i] = True
            p_long = i

        if p_short >= 0:
            age = i - p_short
            if age > cfg.reentry_window or out.at[i, "segment_id"] != out.at[p_short, "segment_id"]:
                p_short = -1
            else:
                reenter = out.at[i, "close"] <= upper
                if reenter:
                    reenter_up[i] = True
                    c1 = bool(out.at[i, "pivot_upper_confirm"] or out.at[i, "touch_upper"] or out.at[i, "repeated_upper_count"] >= 2)
                    c2 = bool(rev_short.iat[i])
                    c3 = bool(mom_turn_down.iat[i])
                    c4 = bool(out.at[i, "close"] <= out.at[i, "box_midline"] + cfg.recapture_to_mid_frac * width)
                    base_csum = int(c1) + int(c2) + int(c3) + int(c4)
                    base_confirm_count[i] = base_csum

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
                reenter = out.at[i, "close"] >= lower
                if reenter:
                    reenter_dn[i] = True
                    c1 = bool(out.at[i, "pivot_lower_confirm"] or out.at[i, "touch_lower"] or out.at[i, "repeated_lower_count"] >= 2)
                    c2 = bool(rev_long.iat[i])
                    c3 = bool(mom_turn_up.iat[i])
                    c4 = bool(out.at[i, "close"] >= out.at[i, "box_midline"] - cfg.recapture_to_mid_frac * width)
                    base_csum = int(c1) + int(c2) + int(c3) + int(c4)
                    base_confirm_count[i] = base_csum

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

    out["fb_overshoot_up"] = overshoot_up
    out["fb_overshoot_down"] = overshoot_dn
    out["fb_reenter_from_up"] = reenter_up
    out["fb_reenter_from_down"] = reenter_dn
    out["fb_breakout_volume_ratio"] = breakout_vol_ratio
    out["fb_reentry_volume_ratio"] = reentry_vol_ratio
    out["fb_reentry_vs_breakout_ratio"] = reentry_vs_breakout
    out["fb_volume_confirmation"] = volume_confirmation
    out["fb_volume_blocked"] = volume_blocked
    out["event_false_break_signal"] = signal
    out["event_false_break_side"] = side
    out["event_false_break_base_confirms"] = base_confirm_count
    out["event_false_break_confirms"] = confirm_count
    out["event_false_break_confidence"] = confidence

    return out


def detect_box_initiation_events(df: pd.DataFrame, cfg: BoxInitiationConfig) -> pd.DataFrame:
    """Detect edge initiation/reversal events at box boundaries."""
    _require_event_columns(df)
    out = df.copy().reset_index(drop=True)

    n = len(out)
    signal = np.zeros(n, dtype=int)
    side = np.full(n, "none", dtype=object)
    confidence = np.zeros(n, dtype=float)
    confirms = np.zeros(n, dtype=int)

    mom = _momentum_series(out["close"], cfg.momentum_lookback)
    mom_up = (mom > 0.0) & (mom >= mom.shift(1))
    mom_dn = (mom < 0.0) & (mom <= mom.shift(1))

    rng = (out["high"] - out["low"]).replace(0.0, np.nan)
    upper_rej = (out["high"] - out["close"]) / rng
    lower_rej = (out["close"] - out["low"]) / rng

    for i in range(n):
        if not out.at[i, "box_valid"]:
            continue

        width = out.at[i, "box_width"]
        upper = out.at[i, "box_upper_edge"]
        lower = out.at[i, "box_lower_edge"]
        mid = out.at[i, "box_midline"]

        if not np.isfinite(width) or width <= 0.0:
            continue

        near_upper = out.at[i, "high"] >= upper - cfg.edge_tolerance_width_frac * width
        near_lower = out.at[i, "low"] <= lower + cfg.edge_tolerance_width_frac * width

        short_c1 = bool(near_upper)
        short_c2 = bool(upper_rej.iat[i] >= cfg.reversal_body_frac and out.at[i, "close"] < out.at[i, "open"])
        short_c3 = bool(mom_dn.iat[i])
        short_c4 = bool(out.at[i, "close"] <= mid)
        short_sum = int(short_c1) + int(short_c2) + int(short_c3) + int(short_c4)

        long_c1 = bool(near_lower)
        long_c2 = bool(lower_rej.iat[i] >= cfg.reversal_body_frac and out.at[i, "close"] > out.at[i, "open"])
        long_c3 = bool(mom_up.iat[i])
        long_c4 = bool(out.at[i, "close"] >= mid)
        long_sum = int(long_c1) + int(long_c2) + int(long_c3) + int(long_c4)

        short_conf = min(1.0, (short_sum / 4.0) * out.at[i, "box_confidence"])
        long_conf = min(1.0, (long_sum / 4.0) * out.at[i, "box_confidence"])

        if short_sum >= cfg.min_confirmations and short_conf >= long_conf:
            signal[i] = -1
            side[i] = "short"
            confirms[i] = short_sum
            confidence[i] = short_conf

        if long_sum >= cfg.min_confirmations and long_conf > short_conf:
            signal[i] = 1
            side[i] = "long"
            confirms[i] = long_sum
            confidence[i] = long_conf

    out["event_box_init_signal"] = signal
    out["event_box_init_side"] = side
    out["event_box_init_confirms"] = confirms
    out["event_box_init_confidence"] = confidence

    out["box_edge_test_upper"] = out["high"] >= (
        out["box_upper_edge"] - cfg.edge_tolerance_width_frac * out["box_width"]
    )
    out["box_edge_test_lower"] = out["low"] <= (
        out["box_lower_edge"] + cfg.edge_tolerance_width_frac * out["box_width"]
    )

    return out


def build_event_tables(
    df: pd.DataFrame,
    false_break_cfg: FalseBreakConfig,
    initiation_cfg: BoxInitiationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fb = detect_false_break_events(df, false_break_cfg)
    bi = detect_box_initiation_events(df, initiation_cfg)

    merged = df.copy()
    merged = merged.join(
        fb[
            [
                "fb_overshoot_up",
                "fb_overshoot_down",
                "fb_reenter_from_up",
                "fb_reenter_from_down",
                "fb_breakout_volume_ratio",
                "fb_reentry_volume_ratio",
                "fb_reentry_vs_breakout_ratio",
                "fb_volume_confirmation",
                "fb_volume_blocked",
                "event_false_break_signal",
                "event_false_break_side",
                "event_false_break_base_confirms",
                "event_false_break_confirms",
                "event_false_break_confidence",
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


def export_event_tables(
    merged_df: pd.DataFrame,
    false_break_df: pd.DataFrame,
    initiation_df: pd.DataFrame,
    merged_path: str,
    false_break_path: str,
    initiation_path: str,
) -> None:
    merged_df.to_csv(merged_path, index=False)
    false_break_df.to_csv(false_break_path, index=False)
    initiation_df.to_csv(initiation_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate event-first signal families.")
    parser.add_argument("--input", required=True, help="Input local-box feature CSV")
    parser.add_argument("--output-merged", required=True)
    parser.add_argument("--output-false-break", required=True)
    parser.add_argument("--output-box-init", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    merged, fb, bi = build_event_tables(df, FalseBreakConfig(), BoxInitiationConfig())
    export_event_tables(merged, fb, bi, args.output_merged, args.output_false_break, args.output_box_init)
