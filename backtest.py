"""Sequential cost-aware backtest utilities for event-first engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    fee_bps_per_leg: float = 5.0
    slippage_bps_per_leg: float = 2.0
    half_spread_bps_per_leg: float = 1.0
    combine_mode: str = "equal_weight"
    engine_weights: Dict[str, float] | None = None
    annualization_hours: int = 8760

    def leg_cost_rate(self) -> float:
        return (self.fee_bps_per_leg + self.slippage_bps_per_leg + self.half_spread_bps_per_leg) / 10_000.0

    def normalized_weights(self) -> Dict[str, float]:
        base = {"false_break": 0.50, "boundary": 0.50}
        merged = base if self.engine_weights is None else {**base, **self.engine_weights}
        s = sum(max(v, 0.0) for v in merged.values())
        if s <= 0:
            return base
        return {k: max(v, 0.0) / s for k, v in merged.items()}


def _prepare_trade_table(trades: pd.DataFrame, cfg: BacktestConfig, weight: float = 1.0) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(
            columns=[
                "engine",
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "hold_bars",
                "gross_return",
                "net_return",
                "weighted_return",
                "exit_reason",
            ]
        )

    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True)
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)

    leg_cost = cfg.leg_cost_rate()
    t["net_return"] = t["gross_return"] - 2.0 * leg_cost
    t["weighted_return"] = t["net_return"] * weight
    return t.sort_values("exit_time").reset_index(drop=True)


def _equity_from_trades(trades: pd.DataFrame, initial_equity: float = 1.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "peak", "drawdown", "trade_return", "engine"]) 

    equity = initial_equity
    peak = initial_equity
    rows = []

    for row in trades.itertuples(index=False):
        r = float(getattr(row, "weighted_return", getattr(row, "net_return")))
        equity *= 1.0 + r
        peak = max(peak, equity)
        dd = equity / peak - 1.0
        rows.append(
            {
                "timestamp": row.exit_time,
                "equity": equity,
                "peak": peak,
                "drawdown": dd,
                "trade_return": r,
                "engine": getattr(row, "engine", "combined"),
            }
        )

    return pd.DataFrame(rows)


def _sharpe_from_trades(trades: pd.DataFrame, annualization_hours: int) -> float:
    if trades.empty or trades["weighted_return"].std(ddof=1) == 0:
        return 0.0
    mean_hold = max(float(trades["hold_bars"].mean()), 1.0)
    trades_per_year = annualization_hours / mean_hold
    return float(trades["weighted_return"].mean() / trades["weighted_return"].std(ddof=1) * np.sqrt(trades_per_year))


def _summary_from_curve(trades: pd.DataFrame, curve: pd.DataFrame, annualization_hours: int) -> Dict[str, float]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }

    win_rate = float((trades["weighted_return"] > 0).mean())
    avg_return = float(trades["weighted_return"].mean())
    total_return = float(curve["equity"].iloc[-1] - 1.0) if not curve.empty else 0.0
    max_dd = float(curve["drawdown"].min()) if not curve.empty else 0.0
    sharpe = _sharpe_from_trades(trades, annualization_hours)
    return {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
    }


def run_backtest(
    trades_false_break: pd.DataFrame,
    trades_boundary: pd.DataFrame,
    cfg: BacktestConfig,
) -> Dict[str, object]:
    """Backtest both engines and optional 50/50 combination."""
    weights = cfg.normalized_weights()

    tf = _prepare_trade_table(trades_false_break, cfg, weight=1.0)
    tb = _prepare_trade_table(trades_boundary, cfg, weight=1.0)

    cf = _equity_from_trades(tf)
    cb = _equity_from_trades(tb)

    sf = _summary_from_curve(tf, cf, cfg.annualization_hours)
    sb = _summary_from_curve(tb, cb, cfg.annualization_hours)

    combined = pd.concat(
        [
            _prepare_trade_table(trades_false_break, cfg, weight=weights["false_break"]),
            _prepare_trade_table(trades_boundary, cfg, weight=weights["boundary"]),
        ],
        ignore_index=True,
    ).sort_values("exit_time")
    cc = _equity_from_trades(combined)
    sc = _summary_from_curve(combined, cc, cfg.annualization_hours)

    exit_reason = combined.groupby(["engine", "exit_reason"]).size().reset_index(name="count")

    return {
        "trades_false_break": tf,
        "trades_boundary": tb,
        "trades_combined": combined,
        "equity_false_break": cf,
        "equity_boundary": cb,
        "equity_combined": cc,
        "summary": {
            "false_break": sf,
            "boundary": sb,
            "combined": sc,
        },
        "exit_reason": exit_reason,
    }


def build_funnel(
    execution_false_break: pd.DataFrame,
    execution_boundary: pd.DataFrame,
    risk_log_false_break: pd.DataFrame,
    risk_log_boundary: pd.DataFrame,
    trades_false_break: pd.DataFrame,
    trades_boundary: pd.DataFrame,
) -> pd.DataFrame:
    def _count(df: pd.DataFrame, cond: pd.Series | None = None) -> int:
        if df is None or df.empty:
            return 0
        if cond is None:
            return len(df)
        return int(cond.sum())

    rows = []
    for engine, ex, lg, tr in [
        ("false_break", execution_false_break, risk_log_false_break, trades_false_break),
        ("boundary", execution_boundary, risk_log_boundary, trades_boundary),
    ]:
        raw = _count(ex, ex["raw_signal"] != 0) if not ex.empty else 0
        gated = _count(ex, ex["exec_signal"] != 0) if not ex.empty else 0
        accepted = _count(lg, (lg["event"] == "entry") & (lg["status"] == "accepted")) if not lg.empty else 0
        blocked = _count(lg, (lg["event"] == "entry") & (lg["status"] == "blocked")) if not lg.empty else 0
        closed = _count(tr)
        rows.extend(
            [
                {"engine": engine, "stage": "raw_signal", "count": raw},
                {"engine": engine, "stage": "after_execution_gate", "count": gated},
                {"engine": engine, "stage": "risk_entry_accepted", "count": accepted},
                {"engine": engine, "stage": "risk_entry_blocked", "count": blocked},
                {"engine": engine, "stage": "closed_trade", "count": closed},
            ]
        )

    return pd.DataFrame(rows)


def save_backtest_tables(backtest_outputs: Dict[str, object], out_dir: str) -> None:
    for key in [
        "trades_false_break",
        "trades_boundary",
        "trades_combined",
        "equity_false_break",
        "equity_boundary",
        "equity_combined",
        "exit_reason",
    ]:
        df = backtest_outputs[key]
        if isinstance(df, pd.DataFrame):
            df.to_csv(f"{out_dir}/{key}.csv", index=False)


def plot_equity_curve(curve: pd.DataFrame, output_png: str, title: str = "Equity Curve") -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    if not curve.empty:
        ax.plot(curve["timestamp"], curve["equity"], linewidth=1.8, color="#1f77b4")
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def plot_drawdown_curve(curve: pd.DataFrame, output_png: str, title: str = "Drawdown") -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    if not curve.empty:
        ax.fill_between(curve["timestamp"], curve["drawdown"], 0.0, alpha=0.35, color="#d95f02")
        ax.plot(curve["timestamp"], curve["drawdown"], linewidth=1.0, color="#d95f02")
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def plot_family_comparison(
    curve_false_break: pd.DataFrame,
    curve_boundary: pd.DataFrame,
    curve_combined: pd.DataFrame,
    output_png: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    if not curve_false_break.empty:
        ax.plot(curve_false_break["timestamp"], curve_false_break["equity"], label="false_break", linewidth=1.4)
    if not curve_boundary.empty:
        ax.plot(curve_boundary["timestamp"], curve_boundary["equity"], label="boundary", linewidth=1.4)
    if not curve_combined.empty:
        ax.plot(curve_combined["timestamp"], curve_combined["equity"], label="combined", linewidth=1.8)
    ax.set_title("Per-family Equity Comparison")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def write_backtest_report(summary: Dict[str, Dict[str, float]], output_md: str) -> None:
    lines = [
        "# Backtest Summary",
        "",
        "| Engine | Trades | Win Rate | Avg Return | Total Return | Max Drawdown | Sharpe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for engine in ["false_break", "boundary", "combined"]:
        s = summary.get(engine, {})
        lines.append(
            "| {e} | {t} | {w:.2%} | {a:.4%} | {tr:.2%} | {dd:.2%} | {sh:.3f} |".format(
                e=engine,
                t=int(s.get("trades", 0)),
                w=float(s.get("win_rate", 0.0)),
                a=float(s.get("avg_return", 0.0)),
                tr=float(s.get("total_return", 0.0)),
                dd=float(s.get("max_drawdown", 0.0)),
                sh=float(s.get("sharpe", 0.0)),
            )
        )

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")