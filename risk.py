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
            close_price = b.at[i, "close"]
            effective_close = b.at[i, "effective_close"]

            exit_price = np.nan
            exit_reason = ""

            if side > 0 and np.isfinite(b.at[i, "low"]) and b.at[i, "low"] <= (entry_price - stop_dist):
                exit_price = entry_price - stop_dist
                exit_reason = "price_stop"
            elif side < 0 and np.isfinite(b.at[i, "high"]) and b.at[i, "high"] >= (entry_price + stop_dist):
                exit_price = entry_price + stop_dist
                exit_reason = "price_stop"

            state_bad = (
                (not bool(b.at[i, "box_valid"]))
                or (str(b.at[i, "box_state"]) not in {"STABLE", "SQUEEZE"})
            )
            if exit_reason == "" and hold >= cfg.state_stop_grace_bars and state_bad and np.isfinite(effective_close):
                exit_price = effective_close
                exit_reason = "state_stop"

            mark_price = effective_close if np.isfinite(effective_close) else entry_price
            unreal = side * (mark_price / entry_price - 1.0)
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

                active = {
                    "side": side,
                    "entry_idx": i,
                    "entry_time": now_ts,
                    "entry_price": entry_price,
                    "entry_confidence": float(cand.get("event_confidence", 0.0)),
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
        # Exit remaining position at the last available finite close after entry.
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