"""End-to-end runner for the event-first causal BTC/USDT range system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from backtest import (
    BacktestConfig,
    build_funnel,
    plot_drawdown_curve,
    plot_equity_curve,
    plot_family_comparison,
    run_backtest,
    save_backtest_tables,
    write_backtest_report,
)
from event_first import BoxInitiationConfig, FalseBreakConfig, build_event_tables, export_event_tables
from execution import (
    ExecutionConfig,
    align_4h_to_1h,
    build_event_family_html,
    build_master_html,
    build_overlay_png,
    export_execution_tables,
    run_execution_engines,
)
from local_box import LocalBoxConfig, attach_gap_segments, build_local_boxes, export_local_boxes, load_ohlcv_csv, resample_5m_to_1h
from risk import RiskConfig, apply_risk_layer, export_risk_tables


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_equity_kline_overlay(
    bars: pd.DataFrame,
    equity_curve: pd.DataFrame,
    output_png: str,
    max_rows: int = 1200,
) -> None:
    b = bars.sort_values("timestamp")
    if len(b) > max_rows:
        b = b.iloc[-max_rows:].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False, gridspec_kw={"height_ratios": [2.1, 1.0]})

    x = np.arange(len(b))
    up = b["close"] >= b["open"]
    dn = ~up

    axes[0].vlines(x, b["low"], b["high"], color="#7c7c7c", linewidth=0.6, alpha=0.7)
    axes[0].vlines(x[up], b.loc[up, "open"], b.loc[up, "close"], color="#2a9d8f", linewidth=2.0)
    axes[0].vlines(x[dn], b.loc[dn, "open"], b.loc[dn, "close"], color="#e76f51", linewidth=2.0)
    axes[0].plot(x, b["box_upper_edge"], color="#264653", linewidth=1.0)
    axes[0].plot(x, b["box_midline"], color="#6d597a", linewidth=0.9, linestyle="--")
    axes[0].plot(x, b["box_lower_edge"], color="#264653", linewidth=1.0)
    axes[0].set_title("K-line + Box")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.2)

    if equity_curve is not None and not equity_curve.empty:
        ec = equity_curve.copy().sort_values("timestamp")
        ec = ec[ec["timestamp"] >= b["timestamp"].min()]
        axes[1].plot(ec["timestamp"], ec["equity"], color="#1f77b4", linewidth=1.8)
    axes[1].set_title("Combined Equity")
    axes[1].set_ylabel("Equity")
    axes[1].grid(alpha=0.2)

    step = max(1, len(b) // 12)
    axes[0].set_xticks(x[::step])
    axes[0].set_xticklabels([str(t)[:16] for t in b["timestamp"].iloc[::step]], rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def _write_features_report(df_1h: pd.DataFrame, df_4h: pd.DataFrame, output_md: str) -> None:
    lines = [
        "# Features Summary",
        "",
        "## 1H Local Boxes",
        f"- Bars: {len(df_1h)}",
        f"- Valid box bars: {int(df_1h['box_valid'].sum())}",
        f"- Range-usable bars: {int(df_1h['range_usable'].sum())}",
        f"- Median width pct: {float(df_1h['box_width_pct'].median()):.4%}",
        f"- State counts: {df_1h['box_state'].value_counts().to_dict()}",
        "",
        "## 4H Local Boxes",
        f"- Bars: {len(df_4h)}",
        f"- Valid box bars: {int(df_4h['box_valid'].sum())}",
        f"- Range-usable bars: {int(df_4h['range_usable'].sum())}",
        f"- Median width pct: {float(df_4h['box_width_pct'].median()):.4%}",
        f"- State counts: {df_4h['box_state'].value_counts().to_dict()}",
        "",
        "All features are causal: rolling windows are segment-scoped and gaps are explicit.",
    ]
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_events_report(
    merged: pd.DataFrame,
    execution_outputs: Dict[str, pd.DataFrame],
    output_md: str,
) -> None:
    fb_raw = int((merged["event_false_break_signal"] != 0).sum())
    bi_raw = int((merged["event_box_init_signal"] != 0).sum())
    fb_exec = int((execution_outputs["false_break"]["exec_signal"] != 0).sum())
    bi_exec = int((execution_outputs["boundary"]["exec_signal"] != 0).sum())

    lines = [
        "# Event Summary",
        "",
        f"- False-break raw signals: {fb_raw}",
        f"- Boundary-init raw signals: {bi_raw}",
        f"- False-break execution-passing signals: {fb_exec}",
        f"- Boundary execution-passing signals: {bi_exec}",
        "",
        "Execution remains event-first: 4H gating is present but thresholded to keep opportunities.",
    ]
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_study(config_path: str = "config.yaml", outdir_override: str | None = None, limit_rows: int | None = None) -> Path:
    cfg = _load_yaml(config_path)

    data_cfg = cfg["data"]
    out_dir = Path(outdir_override or data_cfg.get("output_dir", "outputs"))
    _ensure_dir(out_dir)

    # 1) Load bars.
    bars_5m = load_ohlcv_csv(data_cfg["input_5m"])
    bars_4h = load_ohlcv_csv(data_cfg["input_4h"])

    if bool(data_cfg.get("use_5m_for_1h", True)):
        bars_1h = resample_5m_to_1h(bars_5m)
    else:
        bars_1h = load_ohlcv_csv(data_cfg["input_1h"])

    bars_1h = attach_gap_segments(bars_1h, expected_freq="1h")
    bars_4h = attach_gap_segments(bars_4h, expected_freq="4h")

    if limit_rows is not None and limit_rows > 0:
        bars_1h = bars_1h.iloc[-limit_rows:].reset_index(drop=True)

    # 2) Local boxes for 1H and 4H.
    lb_cfg = LocalBoxConfig(**cfg["local_box"])
    boxes_1h = build_local_boxes(bars_1h, lb_cfg)
    boxes_4h = build_local_boxes(bars_4h, lb_cfg)

    export_local_boxes(boxes_1h, str(out_dir / "local_box_1h.csv"))
    export_local_boxes(boxes_4h, str(out_dir / "local_box_4h.csv"))

    # 3) Causal 4H -> 1H alignment.
    enriched_1h = align_4h_to_1h(boxes_1h, boxes_4h)

    # 4) Event families.
    fb_cfg = FalseBreakConfig(**cfg["event_false_break"])
    bi_cfg = BoxInitiationConfig(**cfg["event_box_initiation"])
    merged, events_fb, events_bi = build_event_tables(enriched_1h, fb_cfg, bi_cfg)

    export_event_tables(
        merged,
        events_fb,
        events_bi,
        str(out_dir / "events_merged.csv"),
        str(out_dir / "event_false_break.csv"),
        str(out_dir / "event_box_initiation.csv"),
    )

    # 5) Execution layer.
    exec_cfg_dict = dict(cfg["execution"])
    exec_cfg_dict["allow_states"] = tuple(exec_cfg_dict.get("allow_states", ["STABLE", "SQUEEZE"]))
    ex_cfg = ExecutionConfig(**exec_cfg_dict)
    execution_outputs = run_execution_engines(merged, ex_cfg)
    export_execution_tables(execution_outputs, str(out_dir))

    # 6) Risk layer.
    risk_cfg = RiskConfig(**cfg["risk"])
    risk_outputs = apply_risk_layer(merged, execution_outputs, risk_cfg)
    export_risk_tables(risk_outputs, str(out_dir))

    # 7) Backtest.
    bt_cfg = BacktestConfig(**cfg["backtest"])
    backtest_outputs = run_backtest(
        risk_outputs["trades_false_break"],
        risk_outputs["trades_boundary"],
        bt_cfg,
    )
    save_backtest_tables(backtest_outputs, str(out_dir))

    funnel = build_funnel(
        execution_outputs["false_break"],
        execution_outputs["boundary"],
        risk_outputs["risk_log_false_break"],
        risk_outputs["risk_log_boundary"],
        backtest_outputs["trades_false_break"],
        backtest_outputs["trades_boundary"],
    )
    funnel.to_csv(out_dir / "backtest_funnel.csv", index=False)

    # 8) Visuals and HTML.
    max_plot_rows = int(cfg.get("visual", {}).get("max_plot_rows", 2500))
    build_master_html(merged, execution_outputs, str(out_dir / "master_event_first.html"))
    build_event_family_html(
        merged,
        events_fb,
        "event_false_break_signal",
        "event_false_break_confidence",
        "False-break Event Family",
        str(out_dir / "event_family_false_break.html"),
    )
    build_event_family_html(
        merged,
        events_bi,
        "event_box_init_signal",
        "event_box_init_confidence",
        "Boundary-init Event Family",
        str(out_dir / "event_family_box_initiation.html"),
    )

    build_overlay_png(merged, execution_outputs, str(out_dir / "overlay_kline_bs.png"), max_rows=max_plot_rows)
    _build_equity_kline_overlay(
        merged,
        backtest_outputs["equity_combined"],
        str(out_dir / "overlay_equity_kline.png"),
        max_rows=max_plot_rows,
    )

    plot_equity_curve(backtest_outputs["equity_combined"], str(out_dir / "equity_combined.png"), title="Combined Equity")
    plot_drawdown_curve(backtest_outputs["equity_combined"], str(out_dir / "drawdown_combined.png"), title="Combined Drawdown")
    plot_family_comparison(
        backtest_outputs["equity_false_break"],
        backtest_outputs["equity_boundary"],
        backtest_outputs["equity_combined"],
        str(out_dir / "per_family_comparison.png"),
    )

    # 9) Markdown reports.
    _write_features_report(boxes_1h, boxes_4h, str(out_dir / "report_features.md"))
    _write_events_report(merged, execution_outputs, str(out_dir / "report_events.md"))
    write_backtest_report(backtest_outputs["summary"], str(out_dir / "report_backtest.md"))

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run event-first causal study for BTC/USDT.")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml")
    parser.add_argument("--outdir", default=None, help="Optional output directory override")
    parser.add_argument("--limit_rows", type=int, default=None, help="Optional trailing-row limit for fast runs")
    args = parser.parse_args()

    out = run_study(config_path=args.config, outdir_override=args.outdir, limit_rows=args.limit_rows)
    print(f"Study complete. Outputs saved to: {out}")


if __name__ == "__main__":
    main()