"""Causal risk management layer for execution events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    atr_window: int = 14
    atr_mult: float = 1.10
    box_stop_mult: float = 0.60
    min_stop_pct: float = 0.002
    max_hold_bars: int = 30
    early_progress_bars: int = 8
    early_progress_min_return: float = 0.001
    state_stop_grace_bars: int = 2
    side_cooldown_bars: int = 4
    circuit_breaker_loss_streak: int = 3
    circuit_breaker_pause_bars: int = 16
    take_profit_enabled: bool = False
    take_profit_mode: str = "midline"  # midline | opposite_edge
    early_failure_filter_enabled: bool = False
    early_failure_bars: int = 6
    early_failure_min_progress: float = 0.003
    early_failure_max_adverse: float = 0.008
    early_failure_scope: str = "all"  # all | low_confirm_only | width_bucket_only | warmup_only
    early_failure_confirm_threshold: int = 2
    early_failure_width_min_pct: float = 0.010
    early_failure_width_max_pct: float = 0.015

    def normalized_take_profit_mode(self) -> str:
        mode = str(self.take_profit_mode).strip().lower()
        if mode in {"midline", "opposite_edge"}:
            return mode
        return "midline"

    def normalized_early_failure_scope(self) -> str:
        scope = str(self.early_failure_scope).strip().lower()
        if scope in {"all", "low_confirm_only", "width_bucket_only", "warmup_only"}:
            return scope
        return "all"


def compute_atr(df: pd.DataFrame, window: int) -> pd.Series:
    hi = df["high"]
    lo = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _effective_close(close: pd.Series) -> pd.Series:
    """Causal fallback price for gap bars: last known close in-sequence."""
    return close.ffill()


def _take_profit_target(cand: dict, side: int, cfg: RiskConfig) -> float:
    if not cfg.take_profit_enabled:
        return np.nan

    entry_price = float(cand.get("price", np.nan))
    mode = cfg.normalized_take_profit_mode()
    if side > 0:
        target = float(cand.get("box_midline", np.nan)) if mode == "midline" else float(cand.get("box_upper_edge", np.nan))
        if not np.isfinite(target) or not np.isfinite(entry_price) or target <= entry_price:
            return np.nan
        return target

    if side < 0:
        target = float(cand.get("box_midline", np.nan)) if mode == "midline" else float(cand.get("box_lower_edge", np.nan))
        if not np.isfinite(target) or not np.isfinite(entry_price) or target >= entry_price:
            return np.nan
        return target

    return np.nan


def _early_failure_scope_match(active: dict, cfg: RiskConfig) -> bool:
    if not cfg.early_failure_filter_enabled:
        return False

    scope = cfg.normalized_early_failure_scope()
    if scope == "all":
        return True
    if scope == "low_confirm_only":
        confirms = float(active.get("entry_base_confirms", np.nan))
        return np.isfinite(confirms) and confirms <= float(cfg.early_failure_confirm_threshold)
    if scope == "width_bucket_only":
        width_pct = float(active.get("entry_box_width_pct", np.nan))
        return np.isfinite(width_pct) and cfg.early_failure_width_min_pct <= width_pct < cfg.early_failure_width_max_pct
    if scope == "warmup_only":
        return bool(active.get("entry_warmup", False))
    return False


def _early_failure_excursions(bars: pd.DataFrame, entry_idx: int, now_idx: int, side: int, entry_price: float) -> tuple[float, float]:
    if now_idx <= entry_idx or not np.isfinite(entry_price) or entry_price <= 0.0:
        return 0.0, 0.0

    window = bars.iloc[entry_idx + 1 : now_idx + 1]
    if window.empty:
        return 0.0, 0.0

    hi = pd.to_numeric(window["high"], errors="coerce").max()
    lo = pd.to_numeric(window["low"], errors="coerce").min()

    if side > 0:
        favorable = max((hi / entry_price - 1.0), 0.0) if np.isfinite(hi) else 0.0
        adverse = max((1.0 - lo / entry_price), 0.0) if np.isfinite(lo) else 0.0
    else:
        favorable = max((1.0 - lo / entry_price), 0.0) if np.isfinite(lo) else 0.0
        adverse = max((hi / entry_price - 1.0), 0.0) if np.isfinite(hi) else 0.0

    return favorable, adverse


def _apply_engine_risk(
    bars: pd.DataFrame,
    execution_df: pd.DataFrame,
    engine_name: str,
    cfg: RiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = bars.copy().sort_values("timestamp").reset_index(drop=True)
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b["atr"] = compute_atr(b, cfg.atr_window)
    b["effective_close"] = _effective_close(b["close"])

    edf = execution_df.copy().sort_values("timestamp").reset_index(drop=True)
    edf = edf[edf["exec_signal"] != 0].copy()
    if edf.empty:
        return pd.DataFrame(), pd.DataFrame()

    ts_to_idx = {ts: i for i, ts in enumerate(b["timestamp"].tolist())}
    edf["bar_idx"] = edf["timestamp"].map(ts_to_idx)
    edf = edf.dropna(subset=["bar_idx"]).copy()
    edf["bar_idx"] = edf["bar_idx"].astype(int)
    edf = edf.sort_values(["bar_idx", "timestamp"]).reset_index(drop=True)

    by_idx: Dict[int, List[dict]] = {}
    for row in edf.to_dict(orient="records"):
        by_idx.setdefault(int(row["bar_idx"]), []).append(row)

    trades: List[dict] = []
    logs: List[dict] = []

    active: dict | None = None
    side_cooldown_until = {1: -10_000_000, -1: -10_000_000}
    circuit_pause_until = -10_000_000
    loss_streak = 0

    for i in range(len(b)):
        now_ts = b.at[i, "timestamp"]

        if active is not None and i > active["entry_idx"]:
            side = active["side"]
            entry_price = active["entry_price"]

            atr = b.at[i, "atr"]
            width = b.at[i, "box_width"] if "box_width" in b.columns else np.nan
            stop_dist = max(
                cfg.min_stop_pct * entry_price,
                (cfg.atr_mult * atr) if np.isfinite(atr) else 0.0,
                (cfg.box_stop_mult * width) if np.isfinite(width) else 0.0,
            )
            hold = i - active["entry_idx"]
            effective_close = b.at[i, "effective_close"]

            exit_price = np.nan
            exit_reason = ""
            early_favorable, early_adverse = _early_failure_excursions(b, active["entry_idx"], i, side, entry_price)

            if side > 0 and np.isfinite(b.at[i, "low"]) and b.at[i, "low"] <= (entry_price - stop_dist):
                exit_price = entry_price - stop_dist
                exit_reason = "price_stop"
            elif side < 0 and np.isfinite(b.at[i, "high"]) and b.at[i, "high"] >= (entry_price + stop_dist):
                exit_price = entry_price + stop_dist
                exit_reason = "price_stop"

            tp_target = active.get("take_profit_target", np.nan)
            tp_mode = active.get("take_profit_mode", cfg.normalized_take_profit_mode())
            if exit_reason == "" and np.isfinite(tp_target):
                if side > 0 and np.isfinite(b.at[i, "high"]) and b.at[i, "high"] >= tp_target:
                    exit_price = tp_target
                    exit_reason = f"take_profit_{tp_mode}"
                elif side < 0 and np.isfinite(b.at[i, "low"]) and b.at[i, "low"] <= tp_target:
                    exit_price = tp_target
                    exit_reason = f"take_profit_{tp_mode}"

            state_bad = (
                (not bool(b.at[i, "box_valid"]))
                or (str(b.at[i, "box_state"]) not in {"STABLE", "SQUEEZE"})
            )
            if exit_reason == "" and hold >= cfg.state_stop_grace_bars and state_bad and np.isfinite(effective_close):
                exit_price = effective_close
                exit_reason = "state_stop"

            mark_price = effective_close if np.isfinite(effective_close) else entry_price
            unreal = side * (mark_price / entry_price - 1.0)
            if (
                exit_reason == ""
                and hold >= cfg.early_failure_bars
                and _early_failure_scope_match(active, cfg)
                and early_favorable < cfg.early_failure_min_progress
                and early_adverse > cfg.early_failure_max_adverse
                and np.isfinite(mark_price)
            ):
                exit_price = mark_price
                exit_reason = "early_failure"

            if exit_reason == "" and hold >= cfg.early_progress_bars and unreal < cfg.early_progress_min_return and np.isfinite(mark_price):
                exit_price = mark_price
                exit_reason = "time_stop_early"

            if exit_reason == "" and hold >= cfg.max_hold_bars and np.isfinite(mark_price):
                exit_price = mark_price
                exit_reason = "time_stop_max"

            if exit_reason and np.isfinite(exit_price):
                gross_return = side * (exit_price / entry_price - 1.0)
                trades.append(
                    {
                        "engine": engine_name,
                        "entry_time": active["entry_time"],
                        "exit_time": now_ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": float(exit_price),
                        "hold_bars": hold,
                        "gross_return": gross_return,
                        "exit_reason": exit_reason,
                        "entry_confidence": active["entry_confidence"],
                        "entry_base_confirms": active.get("entry_base_confirms", np.nan),
                        "entry_box_width_pct": active.get("entry_box_width_pct", np.nan),
                        "entry_warmup": active.get("entry_warmup", False),
                        "take_profit_target": active.get("take_profit_target", np.nan),
                        "early_favorable_excursion": early_favorable,
                        "early_adverse_excursion": early_adverse,
                    }
                )
                logs.append(
                    {
                        "timestamp": now_ts,
                        "engine": engine_name,
                        "event": "exit",
                        "side": side,
                        "status": "closed",
                        "reason": exit_reason,
                    }
                )

                side_cooldown_until[side] = i + cfg.side_cooldown_bars

                if gross_return <= 0:
                    loss_streak += 1
                else:
                    loss_streak = 0

                if loss_streak >= cfg.circuit_breaker_loss_streak:
                    circuit_pause_until = i + cfg.circuit_breaker_pause_bars
                    loss_streak = 0
                    logs.append(
                        {
                            "timestamp": now_ts,
                            "engine": engine_name,
                            "event": "circuit",
                            "side": 0,
                            "status": "pause",
                            "reason": "loss_streak_trigger",
                        }
                    )

                active = None

        if i in by_idx:
            for cand in by_idx[i]:
                side = int(np.sign(cand["exec_signal"]))
                if side == 0:
                    continue

                if active is not None:
                    logs.append(
                        {
                            "timestamp": now_ts,
                            "engine": engine_name,
                            "event": "entry",
                            "side": side,
                            "status": "blocked",
                            "reason": "position_overlap",
                        }
                    )
                    continue

                if i < circuit_pause_until:
                    logs.append(
                        {
                            "timestamp": now_ts,
                            "engine": engine_name,
                            "event": "entry",
                            "side": side,
                            "status": "blocked",
                            "reason": "circuit_pause",
                        }
                    )
                    continue

                if i < side_cooldown_until[side]:
                    logs.append(
                        {
                            "timestamp": now_ts,
                            "engine": engine_name,
                            "event": "entry",
                            "side": side,
                            "status": "blocked",
                            "reason": "risk_side_cooldown",
                        }
                    )
                    continue

                entry_price = float(cand.get("price", np.nan))
                if not np.isfinite(entry_price):
                    entry_price = float(b.at[i, "effective_close"]) if np.isfinite(b.at[i, "effective_close"]) else np.nan

                if not np.isfinite(entry_price):
                    logs.append(
                        {
                            "timestamp": now_ts,
                            "engine": engine_name,
                            "event": "entry",
                            "side": side,
                            "status": "blocked",
                            "reason": "entry_price_nan",
                        }
                    )
                    continue

                tp_target = _take_profit_target(cand, side, cfg)
                active = {
                    "side": side,
                    "entry_idx": i,
                    "entry_time": now_ts,
                    "entry_price": entry_price,
                    "entry_confidence": float(cand.get("event_confidence", 0.0)),
                    "entry_base_confirms": float(cand.get("event_base_confirms", np.nan)),
                    "entry_box_width_pct": float(cand.get("box_width_pct", np.nan)),
                    "entry_warmup": bool(cand.get("entry_warmup", False)),
                    "take_profit_target": tp_target,
                    "take_profit_mode": cfg.normalized_take_profit_mode(),
                }
                logs.append(
                    {
                        "timestamp": now_ts,
                        "engine": engine_name,
                        "event": "entry",
                        "side": side,
                        "status": "accepted",
                        "reason": "",
                    }
                )

    if active is not None:
        exit_slice = b.loc[active["entry_idx"] :, ["timestamp", "effective_close"]].dropna(subset=["effective_close"])
        if exit_slice.empty:
            exit_time = b.at[len(b) - 1, "timestamp"]
            exit_price = active["entry_price"]
        else:
            exit_time = exit_slice.iloc[-1]["timestamp"]
            exit_price = float(exit_slice.iloc[-1]["effective_close"])

        side = active["side"]
        entry_price = active["entry_price"]
        gross_return = side * (exit_price / entry_price - 1.0)
        trades.append(
            {
                "engine": engine_name,
                "entry_time": active["entry_time"],
                "exit_time": exit_time,
                "side": side,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "hold_bars": int((b["timestamp"] == exit_time).idxmax() - active["entry_idx"]),
                "gross_return": gross_return,
                "exit_reason": "end_of_data",
                "entry_confidence": active["entry_confidence"],
                "entry_base_confirms": active.get("entry_base_confirms", np.nan),
                "entry_box_width_pct": active.get("entry_box_width_pct", np.nan),
                "entry_warmup": active.get("entry_warmup", False),
                "take_profit_target": active.get("take_profit_target", np.nan),
                "early_favorable_excursion": np.nan,
                "early_adverse_excursion": np.nan,
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(logs)


def apply_risk_layer(
    bars_1h: pd.DataFrame,
    execution_outputs: Dict[str, pd.DataFrame],
    cfg: RiskConfig,
) -> Dict[str, pd.DataFrame]:
    """Apply risk controls and produce trade-level outputs."""
    engines = [k for k in ["false_break", "boundary"] if k in execution_outputs]

    trades_by_engine: Dict[str, pd.DataFrame] = {}
    logs_by_engine: Dict[str, pd.DataFrame] = {}

    for engine in engines:
        tdf, ldf = _apply_engine_risk(bars_1h, execution_outputs[engine], engine, cfg)
        trades_by_engine[engine] = tdf
        logs_by_engine[engine] = ldf

    trades_all = pd.concat([df for df in trades_by_engine.values() if not df.empty], ignore_index=True)
    logs_all = pd.concat([df for df in logs_by_engine.values() if not df.empty], ignore_index=True)

    out = {
        "trades_false_break": trades_by_engine.get("false_break", pd.DataFrame()),
        "trades_boundary": trades_by_engine.get("boundary", pd.DataFrame()),
        "trades_combined": trades_all.sort_values("exit_time") if not trades_all.empty else trades_all,
        "risk_log_false_break": logs_by_engine.get("false_break", pd.DataFrame()),
        "risk_log_boundary": logs_by_engine.get("boundary", pd.DataFrame()),
        "risk_log_combined": logs_all.sort_values("timestamp") if not logs_all.empty else logs_all,
    }
    return out


def export_risk_tables(risk_outputs: Dict[str, pd.DataFrame], out_dir: str) -> None:
    for name, df in risk_outputs.items():
        df.to_csv(f"{out_dir}/{name}.csv", index=False)
