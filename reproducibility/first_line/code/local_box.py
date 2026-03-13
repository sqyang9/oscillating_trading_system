"""Causal 1H local box builder for range-trading features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


REQUIRED_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class LocalBoxConfig:
    rolling_window: int = 72
    min_periods: int = 24
    lower_quantile: float = 0.20
    upper_quantile: float = 0.80
    pivot_left: int = 2
    pivot_right: int = 2
    pivot_memory: int = 8
    touch_tolerance: float = 0.12
    repeated_test_window: int = 24
    repeated_test_min_gap: int = 2
    method_weights: Dict[str, float] | None = None
    min_width_pct: float = 0.001
    max_width_pct: float = 0.08
    stable_change_window: int = 12
    drift_window: int = 24
    drift_threshold: float = 0.35
    squeeze_width_pct: float = 0.01
    min_methods_each_side: int = 1

    def normalized_weights(self) -> Dict[str, float]:
        default = {"quantile": 0.50, "pivot": 0.30, "repeated_test": 0.20}
        raw = default if self.method_weights is None else {**default, **self.method_weights}
        s = sum(max(v, 0.0) for v in raw.values())
        if s <= 0:
            return default
        return {k: max(v, 0.0) / s for k, v in raw.items()}


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df[REQUIRED_OHLCV_COLUMNS].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def resample_5m_to_1h(df_5m: pd.DataFrame, expected_per_hour: int = 12) -> pd.DataFrame:
    """Resample 5m bars to 1h bars while preserving explicit gaps."""
    df = df_5m.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    indexed = df.set_index("timestamp")

    agg = indexed.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    counts = indexed["close"].resample("1h", label="left", closed="left").count()
    agg["missing_subbars"] = expected_per_hour - counts
    agg["gap"] = counts < expected_per_hour

    # Explicitly keep missing bars as gaps without interpolation.
    for col in ["open", "high", "low", "close", "volume"]:
        agg.loc[agg["gap"], col] = np.nan

    out = agg.reset_index()
    return out


def attach_gap_segments(df: pd.DataFrame, expected_freq: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    expected = pd.Timedelta(expected_freq)

    dt = out["timestamp"].diff()
    structural_gap = dt.gt(expected)
    null_gap = out[["open", "high", "low", "close"]].isna().any(axis=1)
    out["gap"] = structural_gap.fillna(False) | null_gap

    seg_break = out["gap"] | out["gap"].shift(1, fill_value=False)
    out["segment_id"] = seg_break.cumsum().astype(int)
    return out


def _segment_rolling_quantile(
    series: pd.Series,
    segment_id: pd.Series,
    window: int,
    min_periods: int,
    quantile: float,
) -> pd.Series:
    grp = series.groupby(segment_id)
    rolled = grp.rolling(window=window, min_periods=min_periods).quantile(quantile)
    return rolled.reset_index(level=0, drop=True)


def _compute_pivot_edges(df: pd.DataFrame, cfg: LocalBoxConfig) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    hi = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    seg = out["segment_id"].to_numpy(dtype=int)

    pivot_upper = np.full(n, np.nan)
    pivot_lower = np.full(n, np.nan)
    pivot_upper_confirm = np.zeros(n, dtype=bool)
    pivot_lower_confirm = np.zeros(n, dtype=bool)

    highs: list[float] = []
    lows: list[float] = []

    left = cfg.pivot_left
    right = cfg.pivot_right
    mem = max(cfg.pivot_memory, 1)

    for t in range(n):
        if t > 0 and seg[t] != seg[t - 1]:
            highs.clear()
            lows.clear()

        c = t - right
        if c >= left:
            ws = c - left
            we = c + right
            if ws >= 0 and we < n and np.all(seg[ws : we + 1] == seg[t]):
                hi_window = hi[ws : we + 1]
                lo_window = lo[ws : we + 1]

                if np.isfinite(hi[c]) and np.isfinite(hi_window).all() and hi[c] >= np.max(hi_window):
                    highs.append(float(hi[c]))
                    if len(highs) > mem:
                        highs = highs[-mem:]
                    pivot_upper_confirm[t] = True

                if np.isfinite(lo[c]) and np.isfinite(lo_window).all() and lo[c] <= np.min(lo_window):
                    lows.append(float(lo[c]))
                    if len(lows) > mem:
                        lows = lows[-mem:]
                    pivot_lower_confirm[t] = True

        if highs:
            pivot_upper[t] = float(np.median(highs))
        if lows:
            pivot_lower[t] = float(np.median(lows))

    out["pivot_upper_edge"] = pivot_upper
    out["pivot_lower_edge"] = pivot_lower
    out["pivot_upper_confirm"] = pivot_upper_confirm
    out["pivot_lower_confirm"] = pivot_lower_confirm
    return out


def _compute_repeated_test_edges(df: pd.DataFrame, cfg: LocalBoxConfig) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    hi = out["high"].to_numpy(dtype=float)
    lo = out["low"].to_numpy(dtype=float)
    q_up = out["q_upper_edge"].to_numpy(dtype=float)
    q_dn = out["q_lower_edge"].to_numpy(dtype=float)
    seg = out["segment_id"].to_numpy(dtype=int)

    rep_up = np.full(n, np.nan)
    rep_dn = np.full(n, np.nan)
    rep_up_count = np.zeros(n, dtype=int)
    rep_dn_count = np.zeros(n, dtype=int)
    touch_up = np.zeros(n, dtype=bool)
    touch_dn = np.zeros(n, dtype=bool)

    upper_touches: list[Tuple[int, float]] = []
    lower_touches: list[Tuple[int, float]] = []

    for t in range(n):
        if t > 0 and seg[t] != seg[t - 1]:
            upper_touches.clear()
            lower_touches.clear()

        width_q = q_up[t] - q_dn[t] if np.isfinite(q_up[t]) and np.isfinite(q_dn[t]) else np.nan
        if np.isfinite(width_q) and width_q > 0:
            up_trigger = np.isfinite(hi[t]) and hi[t] >= (q_up[t] - cfg.touch_tolerance * width_q)
            dn_trigger = np.isfinite(lo[t]) and lo[t] <= (q_dn[t] + cfg.touch_tolerance * width_q)

            if up_trigger:
                if not upper_touches or (t - upper_touches[-1][0]) >= cfg.repeated_test_min_gap:
                    upper_touches.append((t, float(hi[t])))
                    touch_up[t] = True

            if dn_trigger:
                if not lower_touches or (t - lower_touches[-1][0]) >= cfg.repeated_test_min_gap:
                    lower_touches.append((t, float(lo[t])))
                    touch_dn[t] = True

            min_idx = t - cfg.repeated_test_window
            upper_touches = [x for x in upper_touches if x[0] >= min_idx]
            lower_touches = [x for x in lower_touches if x[0] >= min_idx]

            rep_up_count[t] = len(upper_touches)
            rep_dn_count[t] = len(lower_touches)

            if rep_up_count[t] >= 2:
                rep_up[t] = float(np.mean([x[1] for x in upper_touches[-3:]]))
            if rep_dn_count[t] >= 2:
                rep_dn[t] = float(np.mean([x[1] for x in lower_touches[-3:]]))

    out["repeated_upper_edge"] = rep_up
    out["repeated_lower_edge"] = rep_dn
    out["repeated_upper_count"] = rep_up_count
    out["repeated_lower_count"] = rep_dn_count
    out["touch_upper"] = touch_up
    out["touch_lower"] = touch_dn
    return out


def _weighted_edge(values: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, int]:
    num = 0.0
    den = 0.0
    methods = 0
    for k, v in values.items():
        if np.isfinite(v):
            w = weights.get(k, 0.0)
            num += w * v
            den += w
            methods += 1
    if den <= 0:
        return np.nan, methods
    return num / den, methods


def _consecutive_valid_age(valid: Iterable[bool], segment_id: Iterable[int]) -> np.ndarray:
    valid_arr = np.asarray(list(valid), dtype=bool)
    seg_arr = np.asarray(list(segment_id), dtype=int)
    out = np.zeros(len(valid_arr), dtype=int)

    for i in range(len(valid_arr)):
        if not valid_arr[i]:
            out[i] = 0
            continue
        if i > 0 and valid_arr[i - 1] and seg_arr[i] == seg_arr[i - 1]:
            out[i] = out[i - 1] + 1
        else:
            out[i] = 1
    return out


def build_local_boxes(df: pd.DataFrame, cfg: LocalBoxConfig) -> pd.DataFrame:
    """Create causal local box features for each bar."""
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing columns for local box: {missing}")

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values("timestamp").reset_index(drop=True)

    if "segment_id" not in out.columns or "gap" not in out.columns:
        out = attach_gap_segments(out, expected_freq="1h")

    out["q_lower_edge"] = _segment_rolling_quantile(
        out["close"], out["segment_id"], cfg.rolling_window, cfg.min_periods, cfg.lower_quantile
    )
    out["q_upper_edge"] = _segment_rolling_quantile(
        out["close"], out["segment_id"], cfg.rolling_window, cfg.min_periods, cfg.upper_quantile
    )

    out = _compute_pivot_edges(out, cfg)
    out = _compute_repeated_test_edges(out, cfg)

    weights = cfg.normalized_weights()
    upper = np.full(len(out), np.nan)
    lower = np.full(len(out), np.nan)
    upper_methods = np.zeros(len(out), dtype=int)
    lower_methods = np.zeros(len(out), dtype=int)

    for i in range(len(out)):
        uv, um = _weighted_edge(
            {
                "quantile": float(out.at[i, "q_upper_edge"]),
                "pivot": float(out.at[i, "pivot_upper_edge"]),
                "repeated_test": float(out.at[i, "repeated_upper_edge"]),
            },
            weights,
        )
        lv, lm = _weighted_edge(
            {
                "quantile": float(out.at[i, "q_lower_edge"]),
                "pivot": float(out.at[i, "pivot_lower_edge"]),
                "repeated_test": float(out.at[i, "repeated_lower_edge"]),
            },
            weights,
        )
        upper[i] = uv
        lower[i] = lv
        upper_methods[i] = um
        lower_methods[i] = lm

    out["box_upper_edge"] = upper
    out["box_lower_edge"] = lower
    out["upper_methods"] = upper_methods
    out["lower_methods"] = lower_methods
    out["method_agreement"] = np.minimum(upper_methods, lower_methods) / 3.0

    out["box_midline"] = (out["box_upper_edge"] + out["box_lower_edge"]) / 2.0
    out["box_width"] = out["box_upper_edge"] - out["box_lower_edge"]
    out["box_width_pct"] = out["box_width"] / out["close"].replace(0.0, np.nan)

    seg = out["segment_id"]
    edge_shift = out[["box_upper_edge", "box_lower_edge"]].diff().abs().sum(axis=1)
    norm_change = edge_shift / out["box_width"].shift(1).replace(0.0, np.nan)
    smooth_change = (
        norm_change.groupby(seg)
        .rolling(cfg.stable_change_window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    out["box_stability"] = np.exp(-smooth_change.clip(lower=0.0)).clip(0.0, 1.0)

    out["box_drift"] = (
        out["box_midline"].diff(cfg.drift_window)
        / (out["box_width"].replace(0.0, np.nan) * max(cfg.drift_window, 1))
    )

    broken_up = out["close"] > out["box_upper_edge"]
    broken_dn = out["close"] < out["box_lower_edge"]
    out["broken_test_count"] = (broken_up | broken_dn).groupby(seg).cumsum()
    out["box_break_side"] = np.where(broken_up, -1, np.where(broken_dn, 1, 0))

    width_ok = out["box_width_pct"].between(cfg.min_width_pct, cfg.max_width_pct)
    methods_ok = (out["upper_methods"] >= cfg.min_methods_each_side) & (
        out["lower_methods"] >= cfg.min_methods_each_side
    )
    finite_ok = out[["box_upper_edge", "box_lower_edge", "box_midline", "box_width"]].notna().all(axis=1)
    out["box_valid"] = width_ok & methods_ok & finite_ok
    out["box_age"] = _consecutive_valid_age(out["box_valid"], out["segment_id"])

    stable = out["box_valid"] & (out["box_stability"] >= 0.70) & (out["box_drift"].abs() <= cfg.drift_threshold)
    squeeze = out["box_valid"] & (out["box_width_pct"] <= cfg.squeeze_width_pct)
    state = np.where(~out["box_valid"], "INVALID", np.where(squeeze, "SQUEEZE", np.where(stable, "STABLE", "WIDE")))
    out["box_state"] = state
    out["range_usable"] = out["box_valid"] & out["box_state"].isin(["STABLE", "SQUEEZE"])

    transition = (1.0 - out["box_stability"].fillna(0.0)) + out["box_drift"].abs().fillna(1.0)
    out["transition_risk"] = transition.clip(0.0, 1.0)

    confidence = (
        0.45 * out["method_agreement"].fillna(0.0)
        + 0.35 * out["box_stability"].fillna(0.0)
        + 0.20 * (1.0 - out["box_drift"].abs().clip(0.0, 1.0).fillna(1.0))
    )
    out["box_confidence"] = confidence.clip(0.0, 1.0)

    return out


def local_box_columns() -> list[str]:
    return [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "gap",
        "segment_id",
        "q_lower_edge",
        "q_upper_edge",
        "pivot_upper_edge",
        "pivot_lower_edge",
        "pivot_upper_confirm",
        "pivot_lower_confirm",
        "repeated_upper_edge",
        "repeated_lower_edge",
        "repeated_upper_count",
        "repeated_lower_count",
        "touch_upper",
        "touch_lower",
        "box_upper_edge",
        "box_lower_edge",
        "upper_methods",
        "lower_methods",
        "method_agreement",
        "box_midline",
        "box_width",
        "box_width_pct",
        "box_stability",
        "box_drift",
        "broken_test_count",
        "box_break_side",
        "box_valid",
        "box_age",
        "box_state",
        "range_usable",
        "transition_risk",
        "box_confidence",
    ]


def export_local_boxes(df: pd.DataFrame, output_path: str) -> None:
    cols = [c for c in local_box_columns() if c in df.columns]
    df.loc[:, cols].to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build causal local boxes.")
    parser.add_argument("--input", required=True, help="Input OHLCV CSV")
    parser.add_argument("--output", required=True, help="Output feature CSV")
    parser.add_argument("--from-5m", action="store_true", help="Resample 5m input to 1h first")
    args = parser.parse_args()

    src = load_ohlcv_csv(args.input)
    if args.from_5m:
        src = resample_5m_to_1h(src)
    src = attach_gap_segments(src, expected_freq="1h")
    features = build_local_boxes(src, LocalBoxConfig())
    export_local_boxes(features, args.output)