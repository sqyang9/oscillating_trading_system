"""Focused regime study for false-break strategy performance.

This script analyzes existing backtest outputs and studies two promising regimes:
1) down-only
2) down/up

Regime is defined by 30-day return on 1H closes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeConfig:
    down_low: float = -0.20
    down_high: float = -0.05
    up_low: float = 0.05
    up_high: float = 0.20
    lookback_hours: int = 24 * 30
    cost_per_roundtrip: float = 0.0016


def label_regime(ret_30d: float, cfg: RegimeConfig) -> str:
    if not np.isfinite(ret_30d):
        return "unknown"
    if ret_30d <= cfg.down_low:
        return "deep_down"
    if cfg.down_low < ret_30d <= cfg.down_high:
        return "down"
    if cfg.down_high < ret_30d < cfg.up_low:
        return "flat"
    if cfg.up_low <= ret_30d < cfg.up_high:
        return "up"
    return "strong_up"


def _equity_curve(returns: pd.Series) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame(columns=["trade_idx", "equity", "peak", "drawdown", "ret"]) 

    eq = (1.0 + returns).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return pd.DataFrame(
        {
            "trade_idx": np.arange(1, len(returns) + 1),
            "equity": eq.values,
            "peak": peak.values,
            "drawdown": dd.values,
            "ret": returns.values,
        }
    )


def _summary(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "trades": 0,
            "win_rate_net": 0.0,
            "gross_mean": 0.0,
            "net_mean": 0.0,
            "gross_median": 0.0,
            "net_median": 0.0,
            "gross_comp": 0.0,
            "net_comp": 0.0,
            "max_drawdown": 0.0,
            "avg_hold_bars": 0.0,
            "avg_conf": 0.0,
        }

    gross = df["gross_return"]
    net = df["net_return"]
    curve = _equity_curve(net)

    return {
        "trades": int(len(df)),
        "win_rate_net": float((net > 0).mean()),
        "gross_mean": float(gross.mean()),
        "net_mean": float(net.mean()),
        "gross_median": float(gross.median()),
        "net_median": float(net.median()),
        "gross_comp": float((1.0 + gross).prod() - 1.0),
        "net_comp": float((1.0 + net).prod() - 1.0),
        "max_drawdown": float(curve["drawdown"].min() if not curve.empty else 0.0),
        "avg_hold_bars": float(df["hold_bars"].mean()),
        "avg_conf": float(df["entry_confidence"].mean()) if "entry_confidence" in df.columns else np.nan,
    }


def _bucket_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(key)
        .agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda s: float((s > 0).mean())),
            gross_mean=("gross_return", "mean"),
            net_mean=("net_return", "mean"),
            gross_comp=("gross_return", lambda s: float((1.0 + s).prod() - 1.0)),
            net_comp=("net_return", lambda s: float((1.0 + s).prod() - 1.0)),
            avg_hold=("hold_bars", "mean"),
        )
        .reset_index()
        .sort_values(key)
    )


def _confidence_table(df: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    if df.empty or "entry_confidence" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["conf_bucket"] = pd.qcut(out["entry_confidence"], q=buckets, duplicates="drop")
    return (
        out.groupby("conf_bucket", observed=False)
        .agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda s: float((s > 0).mean())),
            gross_mean=("gross_return", "mean"),
            net_mean=("net_return", "mean"),
            net_comp=("net_return", lambda s: float((1.0 + s).prod() - 1.0)),
        )
        .reset_index()
    )


def _exit_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("exit_reason")
        .agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda s: float((s > 0).mean())),
            gross_mean=("gross_return", "mean"),
            net_mean=("net_return", "mean"),
            net_comp=("net_return", lambda s: float((1.0 + s).prod() - 1.0)),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )


def _side_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("side")
        .agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda s: float((s > 0).mean())),
            gross_mean=("gross_return", "mean"),
            net_mean=("net_return", "mean"),
            net_comp=("net_return", lambda s: float((1.0 + s).prod() - 1.0)),
        )
        .reset_index()
        .sort_values("side")
    )


def _plot_equity_compare(curves: Dict[str, pd.DataFrame], output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, c in curves.items():
        if c.empty:
            continue
        ax.plot(c["trade_idx"], c["equity"], linewidth=1.7, label=name)
    ax.set_title("False-break Net Equity by Regime Filter")
    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def _plot_drawdown_compare(curves: Dict[str, pd.DataFrame], output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for name, c in curves.items():
        if c.empty:
            continue
        ax.plot(c["trade_idx"], c["drawdown"], linewidth=1.3, label=name)
    ax.set_title("False-break Net Drawdown by Regime Filter")
    ax.set_xlabel("Trade Index")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def _plot_yearly_comp(all_yearly: Dict[str, pd.DataFrame], output_png: Path) -> None:
    # Merge for aligned grouped bars.
    names = list(all_yearly.keys())
    years = sorted(set().union(*[set(df["year"].tolist()) for df in all_yearly.values() if not df.empty]))
    if not years:
        return

    x = np.arange(len(years))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(names):
        ymap = {int(r["year"]): float(r["net_comp"]) for _, r in all_yearly[name].iterrows()} if not all_yearly[name].empty else {}
        ys = [ymap.get(y, 0.0) for y in years]
        ax.bar(x + (i - 1) * width, ys, width=width, label=name)

    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_title("Yearly Net Compounded Return by Regime Filter")
    ax.set_ylabel("Net Comp Return")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def _write_md_report(
    summaries: Dict[str, Dict[str, float]],
    yearly: Dict[str, pd.DataFrame],
    out_path: Path,
    cfg: RegimeConfig,
) -> None:
    lines = [
        "# False-break Regime Deep Study",
        "",
        "## Regime Definition",
        f"- Lookback: {cfg.lookback_hours}h (~30d)",
        f"- deep_down: ret_30d <= {cfg.down_low:.0%}",
        f"- down: {cfg.down_low:.0%} < ret_30d <= {cfg.down_high:.0%}",
        f"- flat: {cfg.down_high:.0%} < ret_30d < {cfg.up_low:.0%}",
        f"- up: {cfg.up_low:.0%} <= ret_30d < {cfg.up_high:.0%}",
        f"- strong_up: ret_30d >= {cfg.up_high:.0%}",
        "",
        "## Topline",
        "",
        "| Filter | Trades | Win (net) | Gross Mean | Net Mean | Gross Comp | Net Comp | MaxDD | Avg Hold | Avg Conf |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in ["baseline", "down_only", "down_up"]:
        s = summaries[name]
        lines.append(
            "| {name} | {tr} | {w:.2%} | {gm:.4%} | {nm:.4%} | {gc:.2%} | {nc:.2%} | {dd:.2%} | {h:.2f} | {c:.3f} |".format(
                name=name,
                tr=int(s["trades"]),
                w=float(s["win_rate_net"]),
                gm=float(s["gross_mean"]),
                nm=float(s["net_mean"]),
                gc=float(s["gross_comp"]),
                nc=float(s["net_comp"]),
                dd=float(s["max_drawdown"]),
                h=float(s["avg_hold_bars"]),
                c=float(s["avg_conf"]) if np.isfinite(float(s["avg_conf"])) else np.nan,
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- `down_only` isolates stronger mean-reversion behavior with lower sample size.",
            "- `down_up` offers broader coverage and often better robustness than unrestricted baseline.",
            "- Baseline degradation is usually from flat/strong-up environments where false-break edges decay.",
            "",
            "## Artifacts",
            "- See CSV tables in this same folder for yearly/quarterly/exit/side/confidence drilldowns.",
            "- See PNG charts for equity and drawdown comparisons.",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_research(outputs_dir: Path, out_dir: Path, cfg: RegimeConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(outputs_dir / "trades_false_break.csv")
    bars = pd.read_csv(outputs_dir / "events_merged.csv", usecols=["timestamp", "close"])

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)

    bars = bars.dropna(subset=["close"]).sort_values("timestamp")
    bars["ret_30d"] = bars["close"].pct_change(cfg.lookback_hours, fill_method=None)
    bars["regime"] = bars["ret_30d"].apply(lambda x: label_regime(x, cfg))

    trades = trades.sort_values("entry_time")
    trades = pd.merge_asof(
        trades,
        bars[["timestamp", "ret_30d", "regime"]].sort_values("timestamp"),
        left_on="entry_time",
        right_on="timestamp",
        direction="backward",
    )

    trades["regime"] = trades["regime"].fillna("unknown")
    trades["net_return"] = trades["gross_return"] - cfg.cost_per_roundtrip
    trades["year"] = trades["exit_time"].dt.year
    trades["quarter"] = trades["exit_time"].dt.to_period("Q").astype(str)

    subsets = {
        "baseline": trades.copy(),
        "down_only": trades[trades["regime"] == "down"].copy(),
        "down_up": trades[trades["regime"].isin(["down", "up"])].copy(),
    }

    summaries: Dict[str, Dict[str, float]] = {}
    curves: Dict[str, pd.DataFrame] = {}
    yearly_tables: Dict[str, pd.DataFrame] = {}

    for name, df in subsets.items():
        df = df.sort_values("exit_time").reset_index(drop=True)
        subsets[name] = df
        summaries[name] = _summary(df)
        curves[name] = _equity_curve(df["net_return"] if not df.empty else pd.Series(dtype=float))

        yearly = _bucket_table(df, "year")
        quarterly = _bucket_table(df, "quarter")
        side = _side_table(df)
        exit_r = _exit_table(df)
        conf = _confidence_table(df)

        yearly_tables[name] = yearly

        df.to_csv(out_dir / f"trades_{name}.csv", index=False)
        yearly.to_csv(out_dir / f"yearly_{name}.csv", index=False)
        quarterly.to_csv(out_dir / f"quarterly_{name}.csv", index=False)
        side.to_csv(out_dir / f"side_{name}.csv", index=False)
        exit_r.to_csv(out_dir / f"exit_{name}.csv", index=False)
        conf.to_csv(out_dir / f"confidence_{name}.csv", index=False)

        if not curves[name].empty:
            cdf = curves[name].copy()
            cdf["exit_time"] = df["exit_time"].values
            cdf.to_csv(out_dir / f"equity_curve_{name}.csv", index=False)
        else:
            pd.DataFrame(columns=["trade_idx", "equity", "peak", "drawdown", "ret", "exit_time"]).to_csv(
                out_dir / f"equity_curve_{name}.csv", index=False
            )

    pd.DataFrame([{"filter": k, **v} for k, v in summaries.items()]).to_csv(out_dir / "summary.csv", index=False)

    _plot_equity_compare(curves, out_dir / "equity_compare.png")
    _plot_drawdown_compare(curves, out_dir / "drawdown_compare.png")
    _plot_yearly_comp(yearly_tables, out_dir / "yearly_net_comp_compare.png")

    # Regime distribution in baseline trades.
    reg_dist = subsets["baseline"].groupby(["year", "regime"]).size().reset_index(name="trades")
    reg_dist.to_csv(out_dir / "regime_distribution_yearly.csv", index=False)

    _write_md_report(summaries, yearly_tables, out_dir / "report_false_break_regime.md", cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep regime study for false-break strategy")
    parser.add_argument("--outputs", default="outputs", help="Existing outputs folder from study_event_first")
    parser.add_argument("--outdir", default="outputs/regime_research_false_break", help="Research output folder")
    parser.add_argument("--down-low", type=float, default=-0.20)
    parser.add_argument("--down-high", type=float, default=-0.05)
    parser.add_argument("--up-low", type=float, default=0.05)
    parser.add_argument("--up-high", type=float, default=0.20)
    parser.add_argument("--lookback-hours", type=int, default=24 * 30)
    parser.add_argument("--cost-roundtrip", type=float, default=0.0016)
    args = parser.parse_args()

    cfg = RegimeConfig(
        down_low=args.down_low,
        down_high=args.down_high,
        up_low=args.up_low,
        up_high=args.up_high,
        lookback_hours=args.lookback_hours,
        cost_per_roundtrip=args.cost_roundtrip,
    )

    run_research(Path(args.outputs), Path(args.outdir), cfg)
    print(f"Regime research complete. Outputs: {args.outdir}")


if __name__ == "__main__":
    main()
