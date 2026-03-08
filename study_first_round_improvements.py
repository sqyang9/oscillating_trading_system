"""Run first-round audit and improvement experiments on the closed BTC/USDT dataset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from backtest import BacktestConfig, run_backtest
from study_event_first import run_study

CLOSED_INPUT_5M = "btc_data/closed/btc_usdt_swap_5m_closed.csv"
CLOSED_INPUT_4H = "btc_data/closed/btc_usdt_swap_4h_closed.csv"

WIDTH_BINS = [0.0, 0.005, 0.010, 0.015, 0.025, 0.040, 0.080, np.inf]
WIDTH_LABELS = ["<=0.5%", "0.5-1.0%", "1.0-1.5%", "1.5-2.5%", "2.5-4.0%", "4.0-8.0%", ">8.0%"]


VARIANTS = [
    ("baseline", {}),
    (
        "variant_1_expected_value",
        {
            "execution": {
                "expected_value_filter_enabled": True,
                "expected_value_mode": "to_opposite_edge",
                "min_reward_to_cost_ratio": 3.0,
            }
        },
    ),
    (
        "variant_2_volume_confirmation",
        {
            "event_false_break": {
                "volume_confirmation_enabled": True,
                "volume_confirmation_mode": "require",
                "volume_lookback": 20,
                "volume_min_periods": 10,
                "breakout_volume_max_ratio": 1.15,
                "reentry_volume_min_ratio": 0.95,
                "reentry_vs_breakout_min_ratio": 1.05,
            }
        },
    ),
    (
        "variant_3_regime_fast",
        {
            "execution": {
                "regime_lookback_hours": 168,
            }
        },
    ),
    (
        "variant_4_take_profit",
        {
            "risk": {
                "take_profit_enabled": True,
                "take_profit_mode": "midline",
            }
        },
    ),
    (
        "variant_combo",
        {
            "execution": {
                "expected_value_filter_enabled": True,
                "expected_value_mode": "to_opposite_edge",
                "min_reward_to_cost_ratio": 3.0,
                "regime_lookback_hours": 168,
            },
            "risk": {
                "take_profit_enabled": True,
                "take_profit_mode": "midline",
            },
        },
    ),
]

REGIME_SWEEP_HOURS = [96, 168, 240, 720]


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def _save_yaml(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=False)



def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst



def _force_closed_inputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("data", {})["input_5m"] = CLOSED_INPUT_5M
    cfg.setdefault("data", {})["input_4h"] = CLOSED_INPUT_4H
    cfg["data"]["use_5m_for_1h"] = True
    return cfg



def _equity_curve_from_returns(df: pd.DataFrame, return_col: str = "net_return") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "peak", "drawdown"])

    equity = 1.0
    peak = 1.0
    rows = []
    for row in df.sort_values("exit_time").itertuples(index=False):
        equity *= 1.0 + float(getattr(row, return_col))
        peak = max(peak, equity)
        rows.append(
            {
                "timestamp": row.exit_time,
                "equity": equity,
                "peak": peak,
                "drawdown": equity / peak - 1.0,
            }
        )
    return pd.DataFrame(rows)



def _sharpe(df: pd.DataFrame, annualization_hours: int, return_col: str = "net_return") -> float:
    if df.empty or len(df) < 2:
        return 0.0
    std = float(df[return_col].std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    mean_hold = max(float(df["hold_bars"].mean()), 1.0)
    return float(df[return_col].mean() / std * np.sqrt(annualization_hours / mean_hold))



def _summary(df: pd.DataFrame, annualization_hours: int, return_col: str = "net_return") -> Dict[str, float]:
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
    curve = _equity_curve_from_returns(df, return_col=return_col)
    return {
        "trades": int(len(df)),
        "win_rate": float((df[return_col] > 0).mean()),
        "avg_return": float(df[return_col].mean()),
        "total_return": float(curve["equity"].iloc[-1] - 1.0),
        "sharpe": _sharpe(df, annualization_hours, return_col=return_col),
        "max_drawdown": float(curve["drawdown"].min()),
        "avg_holding_bars": float(df["hold_bars"].mean()),
    }



def _blocked_reason_breakdown(execution_df: pd.DataFrame) -> pd.DataFrame:
    if execution_df.empty:
        return pd.DataFrame(columns=["reason", "count"])
    blocked_reason = execution_df["blocked_reason"].fillna("").astype(str)
    blocked = execution_df[(execution_df["raw_signal"] != 0) & (blocked_reason != "")].copy()
    if blocked.empty:
        return pd.DataFrame(columns=["reason", "count"])
    exploded = blocked.assign(reason=blocked["blocked_reason"].fillna("").astype(str).str.split("|"))
    exploded = exploded.explode("reason")
    exploded = exploded[exploded["reason"].fillna("").astype(str) != ""]
    if exploded.empty:
        return pd.DataFrame(columns=["reason", "count"])
    return exploded.groupby("reason").size().reset_index(name="count").sort_values("count", ascending=False)



def _merge_false_break_trade_features(variant_dir: Path) -> pd.DataFrame:
    trades = pd.read_csv(variant_dir / "trades_false_break.csv")
    if trades.empty:
        return trades

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    execution = pd.read_csv(variant_dir / "execution_false_break.csv")
    execution["timestamp"] = pd.to_datetime(execution["timestamp"], utc=True)
    events = pd.read_csv(variant_dir / "events_merged.csv")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    execution_keep = [
        "timestamp",
        "fb_regime",
        "fb_regime_ret",
        "expected_value_mode",
        "expected_target_price",
        "expected_reward_rate",
        "expected_cost_rate",
        "reward_to_cost_ratio",
        "box_midline",
        "box_upper_edge",
        "box_lower_edge",
        "box_width",
        "price",
    ]
    execution_keep = [c for c in execution_keep if c in execution.columns]
    events_keep = ["timestamp", "box_width_pct", "h4_box_width_pct", "box_state", "h4_box_state"]
    events_keep = [c for c in events_keep if c in events.columns]

    merged = trades.merge(execution[execution_keep], left_on="entry_time", right_on="timestamp", how="left")
    merged = merged.merge(events[events_keep], on="timestamp", how="left")
    return merged



def _group_stats(df: pd.DataFrame, group_col: str, annualization_hours: int) -> pd.DataFrame:
    if df.empty or group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "trades", "win_rate", "avg_return", "total_return", "sharpe", "max_drawdown", "avg_holding_bars"])

    rows = []
    for key, group in df.groupby(group_col, dropna=False):
        label = "NaN" if pd.isna(key) else str(key)
        s = _summary(group.sort_values("exit_time"), annualization_hours)
        rows.append({group_col: label, **s})
    return pd.DataFrame(rows).sort_values("trades", ascending=False).reset_index(drop=True)



def _exit_reason_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["exit_reason", "count", "ratio", "avg_return", "total_return"])
    rows = []
    total = len(df)
    for reason, group in df.groupby("exit_reason"):
        rows.append(
            {
                "exit_reason": str(reason),
                "count": int(len(group)),
                "ratio": float(len(group) / total),
                "avg_return": float(group["net_return"].mean()),
                "total_return": float((1.0 + group["net_return"]).prod() - 1.0),
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)



def _bucket_width(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "box_width_pct" not in out.columns:
        out["width_bucket"] = "NaN"
        return out
    out["width_bucket"] = pd.cut(out["box_width_pct"], bins=WIDTH_BINS, labels=WIDTH_LABELS, include_lowest=True)
    out["width_bucket"] = out["width_bucket"].astype(str)
    return out



def _plot_variant_equity(out_root: Path, variants: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in variants:
        path = out_root / name / "equity_false_break.csv"
        if not path.exists():
            continue
        curve = pd.read_csv(path)
        if curve.empty:
            continue
        ax.plot(pd.to_datetime(curve["timestamp"], utc=True), curve["equity"], linewidth=1.6, label=name)
    ax.set_title("False-break Equity by Variant")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_root / "equity_false_break_variants.png", dpi=130)
    plt.close(fig)



def _plot_regime_sweep(sweep: pd.DataFrame, out_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sweep["lookback_hours"], sweep["false_break_total_return"], marker="o", linewidth=1.6)
    ax.set_title("False-break Return by Regime Lookback")
    ax.set_xlabel("Lookback Hours")
    ax.set_ylabel("Total Return")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_root / "regime_lookback_sweep.png", dpi=130)
    plt.close(fig)



def _write_markdown_summary(summary_df: pd.DataFrame, sweep_df: pd.DataFrame, out_root: Path) -> None:
    lines = [
        "# First Round Experiment Summary",
        "",
        "## Variants",
        "",
        "| Variant | Trades | Win Rate | Avg Return | Total Return | Sharpe | Max Drawdown | Avg Hold | EV Blocked | Volume Blocked | TP Trades |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| {variant} | {trades} | {win:.2%} | {avg:.4%} | {tot:.2%} | {sh:.3f} | {dd:.2%} | {hold:.2f} | {ev} | {vol} | {tp} |".format(
                variant=row["variant"],
                trades=int(row["false_break_trades"]),
                win=float(row["false_break_win_rate"]),
                avg=float(row["false_break_avg_return"]),
                tot=float(row["false_break_total_return"]),
                sh=float(row["false_break_sharpe"]),
                dd=float(row["false_break_max_drawdown"]),
                hold=float(row["false_break_avg_holding_bars"]),
                ev=int(row["expected_value_blocked_count"]),
                vol=int(row["volume_blocked_count"]),
                tp=int(row["take_profit_trade_count"]),
            )
        )
    lines += [
        "",
        "## Regime Lookback Sweep",
        "",
        "| Lookback Hours | Trades | Total Return | Sharpe | Max Drawdown |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in sweep_df.sort_values("lookback_hours").iterrows():
        lines.append(
            "| {lb} | {t} | {tr:.2%} | {sh:.3f} | {dd:.2%} |".format(
                lb=int(row["lookback_hours"]),
                t=int(row["false_break_trades"]),
                tr=float(row["false_break_total_return"]),
                sh=float(row["false_break_sharpe"]),
                dd=float(row["false_break_max_drawdown"]),
            )
        )
    (out_root / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def _cost_audit(base_variant_dir: Path, backtest_cfg: BacktestConfig) -> pd.DataFrame:
    tf = pd.read_csv(base_variant_dir / "trades_false_break.csv")
    tb = pd.read_csv(base_variant_dir / "trades_boundary.csv")
    base = run_backtest(tf, tb, backtest_cfg)
    zero = run_backtest(
        tf,
        tb,
        BacktestConfig(
            fee_bps_per_leg=0.0,
            slippage_bps_per_leg=0.0,
            half_spread_bps_per_leg=0.0,
            engine_weights=backtest_cfg.engine_weights,
            annualization_hours=backtest_cfg.annualization_hours,
        ),
    )
    rows = []
    for name, summary in [("with_cost", base["summary"]["false_break"]), ("zero_cost", zero["summary"]["false_break"])]:
        rows.append(
            {
                "scenario": name,
                "trades": int(summary["trades"]),
                "avg_return": float(summary["avg_return"]),
                "total_return": float(summary["total_return"]),
                "max_drawdown": float(summary["max_drawdown"]),
                "sharpe": float(summary["sharpe"]),
            }
        )
    return pd.DataFrame(rows)



def run_experiments(config_path: str, outdir: str, limit_rows: int | None = None) -> Path:
    out_root = Path(outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _force_closed_inputs(_load_yaml(config_path))
    backtest_cfg = BacktestConfig(**base_cfg["backtest"])

    variant_rows = []
    width_rows = []
    regime_rows = []
    exit_rows = []
    engine_rows = []
    blocked_rows = []

    variant_names = []
    for variant_name, overrides in VARIANTS:
        variant_names.append(variant_name)
        variant_dir = out_root / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)

        vcfg = copy.deepcopy(base_cfg)
        _deep_merge(vcfg, overrides)
        cfg_path = variant_dir / "config_variant.yaml"
        _save_yaml(vcfg, cfg_path)

        run_study(config_path=str(cfg_path), outdir_override=str(variant_dir), limit_rows=limit_rows)

        tf = pd.read_csv(variant_dir / "trades_false_break.csv")
        tc = pd.read_csv(variant_dir / "trades_combined.csv")
        ef = pd.read_csv(variant_dir / "execution_false_break.csv")
        eb = pd.read_csv(variant_dir / "execution_boundary.csv")
        ev = pd.read_csv(variant_dir / "events_merged.csv")

        if not tf.empty:
            tf["entry_time"] = pd.to_datetime(tf["entry_time"], utc=True)
            tf["exit_time"] = pd.to_datetime(tf["exit_time"], utc=True)
        if not tc.empty:
            tc["entry_time"] = pd.to_datetime(tc["entry_time"], utc=True)
            tc["exit_time"] = pd.to_datetime(tc["exit_time"], utc=True)

        s_fb = _summary(tf, backtest_cfg.annualization_hours, return_col="net_return")
        s_combined = _summary(tc, backtest_cfg.annualization_hours, return_col="weighted_return" if "weighted_return" in tc.columns else "net_return")
        ev_blocked = int(ef["blocked_reason"].astype(str).str.contains("expected_value_blocked", regex=False).sum()) if not ef.empty else 0
        regime_blocked = int(ef["blocked_reason"].astype(str).str.contains("regime_blocked", regex=False).sum()) if not ef.empty else 0
        volume_blocked = int(ev["fb_volume_blocked"].sum()) if "fb_volume_blocked" in ev.columns else 0
        tp_trades = int(tf["exit_reason"].astype(str).str.startswith("take_profit_").sum()) if not tf.empty else 0

        variant_rows.append(
            {
                "variant": variant_name,
                "false_break_raw_signals": int((ef["raw_signal"] != 0).sum()) if not ef.empty else 0,
                "false_break_exec_signals": int((ef["exec_signal"] != 0).sum()) if not ef.empty else 0,
                "false_break_trades": s_fb["trades"],
                "false_break_win_rate": s_fb["win_rate"],
                "false_break_avg_return": s_fb["avg_return"],
                "false_break_total_return": s_fb["total_return"],
                "false_break_sharpe": s_fb["sharpe"],
                "false_break_max_drawdown": s_fb["max_drawdown"],
                "false_break_avg_holding_bars": s_fb["avg_holding_bars"],
                "combined_trades": s_combined["trades"],
                "combined_total_return": s_combined["total_return"],
                "combined_sharpe": s_combined["sharpe"],
                "combined_max_drawdown": s_combined["max_drawdown"],
                "expected_value_blocked_count": ev_blocked,
                "regime_blocked_count": regime_blocked,
                "volume_blocked_count": volume_blocked,
                "take_profit_trade_count": tp_trades,
            }
        )

        merged_fb = _merge_false_break_trade_features(variant_dir)
        if not merged_fb.empty:
            merged_fb = _bucket_width(merged_fb)
            wstats = _group_stats(merged_fb, "width_bucket", backtest_cfg.annualization_hours)
            if not wstats.empty:
                wstats.insert(0, "variant", variant_name)
                width_rows.append(wstats)

            rstats = _group_stats(merged_fb, "fb_regime", backtest_cfg.annualization_hours)
            if not rstats.empty:
                rstats.insert(0, "variant", variant_name)
                regime_rows.append(rstats)

            exits = _exit_reason_ratio(merged_fb)
            if not exits.empty:
                exits.insert(0, "variant", variant_name)
                exit_rows.append(exits)

        if not tc.empty:
            estats = _group_stats(tc, "engine", backtest_cfg.annualization_hours)
            if not estats.empty:
                estats.insert(0, "variant", variant_name)
                engine_rows.append(estats)

        blocked = _blocked_reason_breakdown(ef)
        if not blocked.empty:
            blocked.insert(0, "variant", variant_name)
            blocked_rows.append(blocked)

    summary_df = pd.DataFrame(variant_rows).sort_values("variant").reset_index(drop=True)
    summary_df.to_csv(out_root / "variant_summary.csv", index=False)

    if width_rows:
        pd.concat(width_rows, ignore_index=True).to_csv(out_root / "false_break_width_bucket_by_variant.csv", index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(out_root / "false_break_regime_bucket_by_variant.csv", index=False)
    if exit_rows:
        pd.concat(exit_rows, ignore_index=True).to_csv(out_root / "false_break_exit_reason_by_variant.csv", index=False)
    if engine_rows:
        pd.concat(engine_rows, ignore_index=True).to_csv(out_root / "engine_bucket_by_variant.csv", index=False)
    if blocked_rows:
        pd.concat(blocked_rows, ignore_index=True).to_csv(out_root / "blocked_reason_by_variant.csv", index=False)

    cost_audit = _cost_audit(out_root / "baseline", backtest_cfg)
    cost_audit.to_csv(out_root / "cost_sensitivity_baseline.csv", index=False)

    baseline_trade_features = _merge_false_break_trade_features(out_root / "baseline")
    if not baseline_trade_features.empty:
        baseline_trade_features["reward_to_midline"] = np.where(
            baseline_trade_features["side"] > 0,
            (baseline_trade_features["box_midline"] - baseline_trade_features["entry_price"]) / baseline_trade_features["entry_price"],
            (baseline_trade_features["entry_price"] - baseline_trade_features["box_midline"]) / baseline_trade_features["entry_price"],
        )
        baseline_trade_features["reward_to_opposite_edge"] = np.where(
            baseline_trade_features["side"] > 0,
            (baseline_trade_features["box_upper_edge"] - baseline_trade_features["entry_price"]) / baseline_trade_features["entry_price"],
            (baseline_trade_features["entry_price"] - baseline_trade_features["box_lower_edge"]) / baseline_trade_features["entry_price"],
        )
        rt_cost = 2.0 * backtest_cfg.leg_cost_rate()
        baseline_trade_features["round_trip_cost_rate"] = rt_cost
        baseline_trade_features["reward_cost_to_midline"] = baseline_trade_features["reward_to_midline"] / rt_cost
        baseline_trade_features["reward_cost_to_opposite_edge"] = baseline_trade_features["reward_to_opposite_edge"] / rt_cost
        baseline_trade_features.to_csv(out_root / "baseline_false_break_trade_features.csv", index=False)

    sweep_rows = []
    sweep_root = out_root / "regime_sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    for lookback in REGIME_SWEEP_HOURS:
        sweep_dir = sweep_root / f"lookback_{lookback}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        scfg = copy.deepcopy(base_cfg)
        scfg.setdefault("execution", {})["regime_lookback_hours"] = int(lookback)
        cfg_path = sweep_dir / "config_variant.yaml"
        _save_yaml(scfg, cfg_path)
        run_study(config_path=str(cfg_path), outdir_override=str(sweep_dir), limit_rows=limit_rows)
        tf = pd.read_csv(sweep_dir / "trades_false_break.csv")
        if not tf.empty:
            tf["entry_time"] = pd.to_datetime(tf["entry_time"], utc=True)
            tf["exit_time"] = pd.to_datetime(tf["exit_time"], utc=True)
        s = _summary(tf, backtest_cfg.annualization_hours)
        sweep_rows.append(
            {
                "lookback_hours": int(lookback),
                "false_break_trades": s["trades"],
                "false_break_total_return": s["total_return"],
                "false_break_sharpe": s["sharpe"],
                "false_break_max_drawdown": s["max_drawdown"],
                "false_break_avg_holding_bars": s["avg_holding_bars"],
            }
        )
    sweep_df = pd.DataFrame(sweep_rows).sort_values("lookback_hours").reset_index(drop=True)
    sweep_df.to_csv(out_root / "regime_lookback_sweep.csv", index=False)

    _plot_variant_equity(out_root, variant_names)
    _plot_regime_sweep(sweep_df, out_root)
    _write_markdown_summary(summary_df, sweep_df, out_root)

    return out_root



def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-round mean-reversion audit and improvements.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outdir", default="outputs/first_round_improvements")
    parser.add_argument("--limit_rows", type=int, default=None)
    args = parser.parse_args()

    out = run_experiments(args.config, args.outdir, limit_rows=args.limit_rows)
    print(f"First-round experiments complete. Outputs saved to: {out}")


if __name__ == "__main__":
    main()

