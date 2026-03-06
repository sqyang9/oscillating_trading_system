#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manual_box_roundX import load_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined auto-box BS + equity html.")
    p.add_argument("--csv", type=str, default="OKX_BTCUSDT.P, 240.csv")
    p.add_argument("--boxes-csv", type=str, default="auto_detected_boxes.csv")
    p.add_argument("--summary-csv", type=str, default="manual_box/auto_detected_boxes_exec_summary_relaxed.csv")
    p.add_argument("--output-html", type=str, default="manual_box/auto_detected_boxes_exec_summary_relaxed_combined_bs_equity.html")
    p.add_argument("--output-equity-csv", type=str, default="manual_box/auto_detected_boxes_exec_summary_relaxed_combined_equity.csv")
    p.add_argument("--output-trades-csv", type=str, default="manual_box/auto_detected_boxes_exec_summary_relaxed_combined_trades.csv")
    p.add_argument("--output-bars-csv", type=str, default="manual_box/auto_detected_boxes_exec_summary_relaxed_combined_bars.csv")
    return p.parse_args()


def _read_trades(path: Path) -> pd.DataFrame:
    if (not path.exists()) or path.stat().st_size == 0:
        return pd.DataFrame(columns=["time", "idx", "action", "qty", "raw_price", "exec_price", "liquidity", "commission", "reason", "tag", "cash_after"])
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["time", "idx", "action", "qty", "raw_price", "exec_price", "liquidity", "commission", "reason", "tag", "cash_after"])
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _read_equity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _resolve_artifact_path(root: Path, summary_dir: Path, box_id: str, suffix: str) -> Path:
    candidates = [
        summary_dir / f"manual_box_roundX_{box_id}_{suffix}",
        root / "manual_box" / f"manual_box_roundX_{box_id}_{suffix}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _build_stitched_equity(summary: pd.DataFrame, root: Path, summary_dir: Path) -> pd.DataFrame:
    summary = summary.copy()
    summary["start_ts"] = pd.to_datetime(summary["box_id"].str.extract(r"auto_box_(\d{8}_\d{4})_")[0], format="%Y%m%d_%H%M", utc=True, errors="coerce")
    summary = summary.sort_values("start_ts", kind="stable")

    pieces: List[pd.DataFrame] = []
    current_equity = 10000.0
    prev_end = None

    for _, row in summary.iterrows():
        box_id = str(row["box_id"])
        eq_path = _resolve_artifact_path(root, summary_dir, box_id, "best_equity.csv")
        eq = _read_equity(eq_path)
        if eq.empty:
            continue
        eq0 = float(eq["equity"].iloc[0]) if float(eq["equity"].iloc[0]) != 0 else 10000.0
        scale = current_equity / eq0
        eq = eq.copy()
        eq["equity_chained"] = eq["equity"] * scale
        eq["box_id"] = box_id
        if prev_end is not None and pd.Timestamp(eq["time"].iloc[0]) > prev_end:
            gap = pd.DataFrame(
                {
                    "time": [prev_end, pd.Timestamp(eq["time"].iloc[0])],
                    "equity": [current_equity, current_equity],
                    "equity_chained": [current_equity, current_equity],
                    "box_id": ["GAP", "GAP"],
                }
            )
            pieces.append(gap)
        pieces.append(eq)
        current_equity = float(eq["equity_chained"].iloc[-1])
        prev_end = pd.Timestamp(eq["time"].iloc[-1])

    if not pieces:
        return pd.DataFrame(columns=["time", "equity", "equity_chained", "box_id"])
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values("time", kind="stable").drop_duplicates(subset=["time", "box_id"], keep="last")
    return out


def _build_combined_trades(summary: pd.DataFrame, root: Path, summary_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, row in summary.iterrows():
        box_id = str(row["box_id"])
        trade_path = _resolve_artifact_path(root, summary_dir, box_id, "best_trades.csv")
        tr = _read_trades(trade_path)
        if tr.empty:
            continue
        tr = tr.copy()
        tr["box_id"] = box_id
        frames.append(tr)
    if not frames:
        return pd.DataFrame(columns=["time", "action", "exec_price", "reason", "box_id"])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("time", kind="stable")


def main() -> int:
    args = parse_args()
    root = Path(".").resolve()
    csv_path = Path(args.csv).expanduser()
    boxes_path = Path(args.boxes_csv).expanduser()
    summary_path = Path(args.summary_csv).expanduser()
    out_html = Path(args.output_html).expanduser()
    out_eq = Path(args.output_equity_csv).expanduser()
    out_tr = Path(args.output_trades_csv).expanduser()
    out_bars = Path(args.output_bars_csv).expanduser()
    summary_dir = summary_path.resolve().parent

    boxes = pd.read_csv(boxes_path)
    boxes["start_ts"] = pd.to_datetime(boxes["start_ts"], utc=True)
    boxes["end_ts"] = pd.to_datetime(boxes["end_ts"], utc=True)
    boxes = boxes.sort_values("start_ts", kind="stable")

    summary = pd.read_csv(summary_path)
    bars = load_data(csv_path).reset_index()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    if not boxes.empty:
        start = boxes["start_ts"].min() - pd.Timedelta(days=5)
        end = boxes["end_ts"].max() + pd.Timedelta(days=5)
        bars = bars.loc[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)].copy()

    eq = _build_stitched_equity(summary, root, summary_dir)
    trades = _build_combined_trades(summary, root, summary_dir)

    bars.rename(columns={"timestamp": "time"}, inplace=True)
    bars.to_csv(out_bars, index=False)
    eq.to_csv(out_eq, index=False)
    trades.to_csv(out_tr, index=False)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=("Auto Detected Zones + Trades", "Chained Equity"),
    )
    fig.add_trace(
        go.Candlestick(
            x=bars["time"],
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="BTC 4H",
        ),
        row=1,
        col=1,
    )

    palette = [
        "rgba(31,119,180,0.16)",
        "rgba(255,127,14,0.16)",
        "rgba(44,160,44,0.16)",
        "rgba(214,39,40,0.16)",
        "rgba(148,103,189,0.16)",
        "rgba(140,86,75,0.16)",
    ]
    for idx, row in boxes.iterrows():
        fill = palette[idx % len(palette)]
        fig.add_vrect(
            x0=row["start_ts"],
            x1=row["end_ts"],
            fillcolor=fill,
            opacity=1.0,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_shape(
            type="rect",
            x0=row["start_ts"],
            x1=row["end_ts"],
            y0=float(row["box_low"]),
            y1=float(row["box_high"]),
            xref="x",
            yref="y",
            line=dict(color=fill.replace("0.16", "0.85"), width=2),
            fillcolor=fill.replace("0.16", "0.08"),
            row=1,
            col=1,
        )
        mid_ts = row["start_ts"] + (row["end_ts"] - row["start_ts"]) / 2
        fig.add_annotation(
            x=mid_ts,
            y=float(row["box_high"]),
            text=str(row["box_id"]),
            showarrow=False,
            yshift=10,
            font=dict(size=10, color="#111"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=fill.replace("0.16", "0.85"),
            borderwidth=1,
            row=1,
            col=1,
        )

    if not trades.empty:
        sl = trades.loc[trades["reason"] == "HARD_STOP"]
        b = trades.loc[(trades["action"] == "buy") & (trades["reason"] != "HARD_STOP")]
        s = trades.loc[(trades["action"] == "sell") & (trades["reason"] != "HARD_STOP")]
        common_hover = "<b>%{text}</b><br>%{x}<br>price=%{y:.2f}<extra></extra>"
        if not b.empty:
            fig.add_trace(
                go.Scatter(
                    x=b["time"],
                    y=b["exec_price"],
                    mode="markers+text",
                    text=["B"] * len(b),
                    textposition="bottom center",
                    name="B",
                    marker=dict(symbol="triangle-up", size=10, color="#2ca02c"),
                    customdata=b[["box_id", "reason", "tag"]].astype(str).agg(" | ".join, axis=1),
                    hovertemplate="<b>B</b><br>%{x}<br>price=%{y:.2f}<br>%{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if not sl.empty:
            fig.add_trace(
                go.Scatter(
                    x=sl["time"],
                    y=sl["exec_price"],
                    mode="markers+text",
                    text=["SL"] * len(sl),
                    textposition="top center",
                    name="SL",
                    marker=dict(symbol="x", size=11, color="#111111"),
                    customdata=sl[["box_id", "reason", "tag"]].astype(str).agg(" | ".join, axis=1),
                    hovertemplate="<b>SL</b><br>%{x}<br>price=%{y:.2f}<br>%{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if not s.empty:
            fig.add_trace(
                go.Scatter(
                    x=s["time"],
                    y=s["exec_price"],
                    mode="markers+text",
                    text=["S"] * len(s),
                    textposition="top center",
                    name="S",
                    marker=dict(symbol="triangle-down", size=10, color="#d62728"),
                    customdata=s[["box_id", "reason", "tag"]].astype(str).agg(" | ".join, axis=1),
                    hovertemplate="<b>S</b><br>%{x}<br>price=%{y:.2f}<br>%{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if not eq.empty:
        fig.add_trace(
            go.Scatter(
                x=eq["time"],
                y=eq["equity_chained"],
                mode="lines",
                name="Chained Equity",
                line=dict(width=2, color="#111111"),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=980,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        title=f"Auto Box Combined View ({len(boxes)} detected zones)",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)

    print(f"html={out_html}")
    print(f"equity={out_eq}")
    print(f"trades={out_tr}")
    print(f"bars={out_bars}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
