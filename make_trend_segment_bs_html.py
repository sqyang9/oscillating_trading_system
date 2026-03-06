"""Standalone interactive HTML: trend segmentation + B/S markers.

This script is intentionally decoupled from the main pipeline for easier maintenance.
It reuses generated outputs (events/execution/equity) and builds one visual report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _label_regime(ret_val: float, down_low: float, down_high: float, up_low: float, up_high: float) -> str:
    if not np.isfinite(ret_val):
        return "warmup"
    if ret_val <= down_low:
        return "deep_down"
    if down_low < ret_val <= down_high:
        return "down"
    if down_high < ret_val < up_low:
        return "flat"
    if up_low <= ret_val < up_high:
        return "up"
    return "strong_up"


def _regime_color(regime: str) -> str:
    mapping = {
        "deep_down": "rgba(214,39,40,0.16)",
        "down": "rgba(255,127,14,0.16)",
        "flat": "rgba(148,103,189,0.16)",
        "up": "rgba(31,119,180,0.16)",
        "strong_up": "rgba(44,160,44,0.16)",
        "warmup": "rgba(140,86,75,0.12)",
    }
    return mapping.get(regime, "rgba(180,180,180,0.12)")


def _build_segments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["seg_id", "regime", "start_ts", "end_ts", "bars"])

    work = df[["timestamp", "regime"]].copy().sort_values("timestamp").reset_index(drop=True)
    seg_break = work["regime"] != work["regime"].shift(1)
    work["seg_id"] = seg_break.cumsum().astype(int)

    seg = (
        work.groupby("seg_id")
        .agg(
            regime=("regime", "first"),
            start_ts=("timestamp", "first"),
            end_ts=("timestamp", "last"),
            bars=("timestamp", "size"),
        )
        .reset_index()
    )
    return seg


def _fmt_ts(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "NA"
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def build_html(
    source_dir: Path,
    output_html: Path,
    output_segments_csv: Path,
    lookback_hours: int,
    down_low: float,
    down_high: float,
    up_low: float,
    up_high: float,
    max_rows: int,
) -> None:
    events_path = source_dir / "events_merged.csv"
    exec_path = source_dir / "execution_false_break.csv"
    equity_path = source_dir / "equity_false_break.csv"
    trades_path = source_dir / "trades_false_break.csv"

    if not events_path.exists() or not exec_path.exists():
        raise FileNotFoundError("Missing events_merged.csv or execution_false_break.csv in source_dir")

    bars = pd.read_csv(events_path)
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.sort_values("timestamp").reset_index(drop=True)

    bars["regime_ret_30d"] = bars["close"].pct_change(max(1, int(lookback_hours)), fill_method=None)
    bars["regime"] = bars["regime_ret_30d"].apply(
        lambda x: _label_regime(float(x), down_low=down_low, down_high=down_high, up_low=up_low, up_high=up_high)
    )

    bars_full = bars.dropna(subset=["open", "high", "low", "close"]).copy()
    full_bar_start = bars_full["timestamp"].min() if not bars_full.empty else pd.NaT
    full_bar_end = bars_full["timestamp"].max() if not bars_full.empty else pd.NaT

    trade_start = pd.NaT
    trade_end = pd.NaT
    if trades_path.exists():
        tr = pd.read_csv(trades_path)
        if not tr.empty and {"entry_time", "exit_time"}.issubset(tr.columns):
            tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
            tr["exit_time"] = pd.to_datetime(tr["exit_time"], utc=True)
            trade_start = tr["entry_time"].min()
            trade_end = tr["exit_time"].max()

    plot = bars_full.copy()
    if max_rows > 0 and len(plot) > max_rows:
        plot = plot.iloc[-max_rows:].copy()

    seg = _build_segments(plot)
    seg.to_csv(output_segments_csv, index=False)

    ex = pd.read_csv(exec_path)
    ex["timestamp"] = pd.to_datetime(ex["timestamp"], utc=True)
    ex = ex[ex["exec_signal"] != 0].copy().sort_values("timestamp")
    if not ex.empty and not plot.empty:
        ex = ex[ex["timestamp"].between(plot["timestamp"].min(), plot["timestamp"].max())]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        specs=[[{}], [{"secondary_y": True}]],
        subplot_titles=("Trend Segments + False-break B/S", "Regime Return (30d) + False-break Equity"),
    )

    fig.add_trace(
        go.Candlestick(
            x=plot["timestamp"],
            open=plot["open"],
            high=plot["high"],
            low=plot["low"],
            close=plot["close"],
            name="BTC/USDT 1H",
        ),
        row=1,
        col=1,
    )

    for col_name, trace_name, style in [
        ("box_upper_edge", "Box Upper", dict(width=1.3, color="#264653")),
        ("box_midline", "Box Mid", dict(width=1.0, color="#6d597a", dash="dash")),
        ("box_lower_edge", "Box Lower", dict(width=1.3, color="#264653")),
    ]:
        if col_name in plot.columns:
            fig.add_trace(
                go.Scatter(x=plot["timestamp"], y=plot[col_name], mode="lines", name=trace_name, line=style),
                row=1,
                col=1,
            )

    for r in seg.itertuples(index=False):
        fill = _regime_color(str(r.regime))
        fig.add_vrect(
            x0=r.start_ts,
            x1=r.end_ts,
            fillcolor=fill,
            opacity=1.0,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        if int(r.bars) >= 12:
            mid = r.start_ts + (r.end_ts - r.start_ts) / 2
            y = float(plot.loc[(plot["timestamp"] >= r.start_ts) & (plot["timestamp"] <= r.end_ts), "high"].max())
            fig.add_annotation(
                x=mid,
                y=y,
                xref="x",
                yref="y",
                text=f"{r.regime}",
                showarrow=False,
                yshift=10,
                font=dict(size=10, color="#111"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=fill.replace("0.16", "0.85").replace("0.12", "0.85"),
                borderwidth=1,
                row=1,
                col=1,
            )

    if not ex.empty:
        longs = ex[ex["exec_signal"] > 0]
        shorts = ex[ex["exec_signal"] < 0]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["timestamp"],
                    y=longs["price"],
                    mode="markers+text",
                    text=["B"] * len(longs),
                    textposition="bottom center",
                    name="B",
                    marker=dict(symbol="triangle-up", size=10, color="#2ca02c"),
                    customdata=np.column_stack(
                        [
                            longs.get("event_confidence", pd.Series([np.nan] * len(longs))).round(4).astype(str),
                            longs.get("fb_regime", pd.Series(["unknown"] * len(longs))).astype(str),
                        ]
                    ),
                    hovertemplate="<b>B</b><br>%{x}<br>price=%{y:.2f}<br>conf=%{customdata[0]} | regime=%{customdata[1]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts["timestamp"],
                    y=shorts["price"],
                    mode="markers+text",
                    text=["S"] * len(shorts),
                    textposition="top center",
                    name="S",
                    marker=dict(symbol="triangle-down", size=10, color="#d62728"),
                    customdata=np.column_stack(
                        [
                            shorts.get("event_confidence", pd.Series([np.nan] * len(shorts))).round(4).astype(str),
                            shorts.get("fb_regime", pd.Series(["unknown"] * len(shorts))).astype(str),
                        ]
                    ),
                    hovertemplate="<b>S</b><br>%{x}<br>price=%{y:.2f}<br>conf=%{customdata[0]} | regime=%{customdata[1]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    regime_line = plot[["timestamp", "regime_ret_30d"]].copy()
    fig.add_trace(
        go.Scatter(
            x=regime_line["timestamp"],
            y=regime_line["regime_ret_30d"] * 100.0,
            mode="lines",
            name="ret_30d (%)",
            line=dict(color="#1f77b4", width=1.6),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    for level, color in [
        (down_low * 100.0, "#d62728"),
        (down_high * 100.0, "#ff7f0e"),
        (up_low * 100.0, "#2ca02c"),
        (up_high * 100.0, "#17becf"),
    ]:
        fig.add_hline(y=level, row=2, col=1, line_dash="dot", line_color=color, opacity=0.5, secondary_y=False)

    if equity_path.exists():
        eq = pd.read_csv(equity_path)
        if not eq.empty and {"timestamp", "equity"}.issubset(eq.columns):
            eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True)
            eq = eq.sort_values("timestamp")
            eq = eq[eq["timestamp"].between(plot["timestamp"].min(), plot["timestamp"].max())]
            fig.add_trace(
                go.Scatter(
                    x=eq["timestamp"],
                    y=eq["equity"],
                    mode="lines",
                    name="False-break Equity",
                    line=dict(width=1.8, color="#111111"),
                ),
                row=2,
                col=1,
                secondary_y=True,
            )

    title = (
        f"Trend Segmentation + B/S ({source_dir.name})"
        f"<br><sup>Bar Window: {_fmt_ts(full_bar_start)} ~ {_fmt_ts(full_bar_end)}"
        f" | False-break Trades: {_fmt_ts(trade_start)} ~ {_fmt_ts(trade_end)}</sup>"
    )

    fig.update_layout(
        template="plotly_white",
        height=980,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        title=title,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ret_30d (%)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Equity", row=2, col=1, secondary_y=True)
    fig.write_html(str(output_html), include_plotlyjs="cdn", full_html=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build standalone trend-segment + BS interactive HTML")
    parser.add_argument("--source-dir", type=str, required=True, help="Folder containing events/execution/equity CSVs")
    parser.add_argument("--output-html", type=str, default=None, help="Output HTML path")
    parser.add_argument("--output-segments", type=str, default=None, help="Optional output segment CSV path")
    parser.add_argument("--lookback-hours", type=int, default=720)
    parser.add_argument("--down-low", type=float, default=-0.20)
    parser.add_argument("--down-high", type=float, default=-0.05)
    parser.add_argument("--up-low", type=float, default=0.05)
    parser.add_argument("--up-high", type=float, default=0.20)
    parser.add_argument("--max-rows", type=int, default=3500)
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser()
    out_html = Path(args.output_html).expanduser() if args.output_html else source_dir / "trend_segment_bs.html"
    out_seg = Path(args.output_segments).expanduser() if args.output_segments else source_dir / "trend_segments.csv"

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_seg.parent.mkdir(parents=True, exist_ok=True)

    build_html(
        source_dir=source_dir,
        output_html=out_html,
        output_segments_csv=out_seg,
        lookback_hours=max(int(args.lookback_hours), 1),
        down_low=float(args.down_low),
        down_high=float(args.down_high),
        up_low=float(args.up_low),
        up_high=float(args.up_high),
        max_rows=max(int(args.max_rows), 200),
    )

    print(f"html={out_html}")
    print(f"segments={out_seg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
