"""Standalone chart: tradable context windows + B/S points + equity curve."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research_context import compute_tradable_context_mask


TOP_HEIGHT = 0.72
BOTTOM_HEIGHT = 0.28



TRADEABLE_COLOR = "rgba(55, 126, 184, 0.14)"
BUY_COLOR = "#1b9e77"
SELL_COLOR = "#d95f02"
BOX_COLOR = "#264653"
MID_COLOR = "#6d597a"


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _context_tradeable(execution_df: pd.DataFrame) -> pd.DataFrame:
    df = execution_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["context_tradeable"] = compute_tradable_context_mask(df)
    return df


def _segments(df: pd.DataFrame, flag_col: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df.empty or flag_col not in df.columns:
        return []

    work = df[["timestamp", flag_col]].copy().sort_values("timestamp").reset_index(drop=True)
    segs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    last_ts = None

    for row in work.itertuples(index=False):
        ts = row.timestamp
        flag = bool(getattr(row, flag_col))
        if flag and start is None:
            start = ts
        if not flag and start is not None and last_ts is not None:
            segs.append((start, last_ts))
            start = None
        last_ts = ts

    if start is not None and last_ts is not None:
        segs.append((start, last_ts))
    return segs


def _equity_with_origin(curve: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return curve.copy()
    if curve.empty:
        return pd.DataFrame({"timestamp": [bars["timestamp"].iloc[0]], "equity": [1.0]})

    out = curve.copy().sort_values("timestamp").reset_index(drop=True)
    first_bar = bars["timestamp"].iloc[0]
    if pd.Timestamp(out["timestamp"].iloc[0]) > first_bar:
        prefix = pd.DataFrame({"timestamp": [first_bar], "equity": [1.0]})
        out = pd.concat([prefix, out[["timestamp", "equity"]]], ignore_index=True)
    return out


def _equity_markers(entries: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    if entries.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "exec_signal", "price", "event_confidence"])

    eq = equity.copy().sort_values("timestamp").reset_index(drop=True)
    if eq.empty:
        out = entries.copy()
        out["equity"] = 1.0
        return out

    left = entries.sort_values("timestamp").reset_index(drop=True).copy()
    merged = pd.merge_asof(left, eq[["timestamp", "equity"]], on="timestamp", direction="backward")
    merged["equity"] = merged["equity"].fillna(1.0)
    return merged


def build_html(source_dir: Path, output_html: Path, max_rows: int = 0) -> None:
    bars = _load_csv(source_dir / "events_merged.csv")
    execution = _context_tradeable(_load_csv(source_dir / "execution_false_break.csv"))
    equity = _load_csv(source_dir / "equity_false_break.csv")

    if bars.empty or execution.empty:
        raise ValueError("Missing events/execution rows for chart")

    plot = bars.sort_values("timestamp").reset_index(drop=True)
    execution = execution.sort_values("timestamp").reset_index(drop=True)
    if max_rows > 0 and len(plot) > max_rows:
        plot = plot.iloc[-max_rows:].reset_index(drop=True)
        execution = execution[execution["timestamp"] >= plot["timestamp"].iloc[0]].reset_index(drop=True)
        equity = equity[equity["timestamp"] >= plot["timestamp"].iloc[0]].reset_index(drop=True)

    tradeable = execution[["timestamp", "context_tradeable"]].copy()
    segs = _segments(tradeable, "context_tradeable")

    entries = execution[execution["exec_signal"] != 0].copy()
    equity_plot = _equity_with_origin(equity, plot)
    entry_equity = _equity_markers(entries, equity_plot)

    title = (
        "Tradable Context + False-break B/S + Equity "
        f"({plot['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M UTC')} to "
        f"{plot['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M UTC')})"
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[TOP_HEIGHT, BOTTOM_HEIGHT],
        subplot_titles=("K-line + Tradable Windows + B/S", "False-break Equity"),
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

    for col_name, trace_name, line in [
        ("box_upper_edge", "Box Upper", dict(width=1.2, color=BOX_COLOR)),
        ("box_midline", "Box Mid", dict(width=1.0, color=MID_COLOR, dash="dash")),
        ("box_lower_edge", "Box Lower", dict(width=1.2, color=BOX_COLOR)),
    ]:
        if col_name in plot.columns:
            fig.add_trace(
                go.Scatter(x=plot["timestamp"], y=plot[col_name], mode="lines", name=trace_name, line=line),
                row=1,
                col=1,
            )

    for start, end in segs:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=TRADEABLE_COLOR,
            opacity=1.0,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )

    if not entries.empty:
        longs = entries[entries["exec_signal"] > 0]
        shorts = entries[entries["exec_signal"] < 0]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["timestamp"],
                    y=longs["price"],
                    mode="markers+text",
                    text=["B"] * len(longs),
                    textposition="bottom center",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=10, color=BUY_COLOR),
                    customdata=np.column_stack(
                        [
                            longs["event_confidence"].round(4),
                            longs["fb_regime"].astype(str),
                            longs["reward_to_cost_ratio"].round(2),
                        ]
                    ),
                    hovertemplate="<b>B</b><br>%{x}<br>price=%{y:.2f}<br>conf=%{customdata[0]}<br>regime=%{customdata[1]}<br>R/C=%{customdata[2]}<extra></extra>",
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
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=10, color=SELL_COLOR),
                    customdata=np.column_stack(
                        [
                            shorts["event_confidence"].round(4),
                            shorts["fb_regime"].astype(str),
                            shorts["reward_to_cost_ratio"].round(2),
                        ]
                    ),
                    hovertemplate="<b>S</b><br>%{x}<br>price=%{y:.2f}<br>conf=%{customdata[0]}<br>regime=%{customdata[1]}<br>R/C=%{customdata[2]}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Scatter(
            x=equity_plot["timestamp"],
            y=equity_plot["equity"],
            mode="lines",
            line=dict(width=1.8, color="#1f77b4"),
            name="Equity",
            hovertemplate="%{x}<br>equity=%{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if not entry_equity.empty:
        eq_longs = entry_equity[entry_equity["exec_signal"] > 0]
        eq_shorts = entry_equity[entry_equity["exec_signal"] < 0]
        if not eq_longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=eq_longs["timestamp"],
                    y=eq_longs["equity"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=8, color=BUY_COLOR),
                    name="Buy on Equity",
                    hovertemplate="%{x}<br>equity=%{y:.4f}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        if not eq_shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=eq_shorts["timestamp"],
                    y=eq_shorts["equity"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=8, color=SELL_COLOR),
                    name="Sell on Equity",
                    hovertemplate="%{x}<br>equity=%{y:.4f}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Equity", row=2, col=1)
    fig.write_html(output_html, include_plotlyjs="cdn")


def build_png(source_dir: Path, output_png: Path, max_rows: int = 3000) -> None:
    bars = _load_csv(source_dir / "events_merged.csv")
    execution = _context_tradeable(_load_csv(source_dir / "execution_false_break.csv"))
    equity = _load_csv(source_dir / "equity_false_break.csv")

    plot = bars.sort_values("timestamp").reset_index(drop=True)
    execution = execution.sort_values("timestamp").reset_index(drop=True)
    if max_rows > 0 and len(plot) > max_rows:
        plot = plot.iloc[-max_rows:].reset_index(drop=True)
        execution = execution[execution["timestamp"] >= plot["timestamp"].iloc[0]].reset_index(drop=True)
        equity = equity[equity["timestamp"] >= plot["timestamp"].iloc[0]].reset_index(drop=True)

    segs = _segments(execution[["timestamp", "context_tradeable"]], "context_tradeable")
    entries = execution[execution["exec_signal"] != 0].copy()
    equity_plot = _equity_with_origin(equity, plot)
    entry_equity = _equity_markers(entries, equity_plot)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [TOP_HEIGHT, BOTTOM_HEIGHT]})
    x = np.arange(len(plot))
    up = plot["close"] >= plot["open"]
    dn = ~up

    for start, end in segs:
        mask = (plot["timestamp"] >= start) & (plot["timestamp"] <= end)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        axes[0].axvspan(idx[0], idx[-1], color="#377eb8", alpha=0.10)

    axes[0].vlines(x, plot["low"], plot["high"], color="#777777", linewidth=0.6, alpha=0.7)
    axes[0].vlines(x[up], plot.loc[up, "open"], plot.loc[up, "close"], color="#2a9d8f", linewidth=2.0)
    axes[0].vlines(x[dn], plot.loc[dn, "open"], plot.loc[dn, "close"], color="#e76f51", linewidth=2.0)
    axes[0].plot(x, plot["box_upper_edge"], color=BOX_COLOR, linewidth=1.0)
    axes[0].plot(x, plot["box_midline"], color=MID_COLOR, linewidth=0.9, linestyle="--")
    axes[0].plot(x, plot["box_lower_edge"], color=BOX_COLOR, linewidth=1.0)

    ts_to_idx = {ts: i for i, ts in enumerate(plot["timestamp"])}
    if not entries.empty:
        longs = entries[entries["exec_signal"] > 0]
        shorts = entries[entries["exec_signal"] < 0]
        if not longs.empty:
            xs = [ts_to_idx[t] for t in longs["timestamp"] if t in ts_to_idx]
            ys = [p for t, p in zip(longs["timestamp"], longs["price"]) if t in ts_to_idx]
            axes[0].scatter(xs, ys, color=BUY_COLOR, s=30, marker="^", label="B")
        if not shorts.empty:
            xs = [ts_to_idx[t] for t in shorts["timestamp"] if t in ts_to_idx]
            ys = [p for t, p in zip(shorts["timestamp"], shorts["price"]) if t in ts_to_idx]
            axes[0].scatter(xs, ys, color=SELL_COLOR, s=30, marker="v", label="S")

    axes[0].set_title("Tradable Context Windows + False-break B/S")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.2)

    axes[1].plot(equity_plot["timestamp"], equity_plot["equity"], color="#1f77b4", linewidth=1.8)
    if not entry_equity.empty:
        eq_longs = entry_equity[entry_equity["exec_signal"] > 0]
        eq_shorts = entry_equity[entry_equity["exec_signal"] < 0]
        if not eq_longs.empty:
            axes[1].scatter(eq_longs["timestamp"], eq_longs["equity"], color=BUY_COLOR, s=22, marker="^")
        if not eq_shorts.empty:
            axes[1].scatter(eq_shorts["timestamp"], eq_shorts["equity"], color=SELL_COLOR, s=22, marker="v")
    axes[1].set_title("False-break Equity")
    axes[1].set_ylabel("Equity")
    axes[1].grid(alpha=0.2)

    step = max(1, len(plot) // 12)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels([str(ts)[:16] for ts in plot["timestamp"].iloc[::step]], rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tradable-window + B/S + equity charts.")
    parser.add_argument("--source-dir", default="outputs/first_round_improvements/baseline")
    parser.add_argument("--output-html", default=None)
    parser.add_argument("--output-png", default=None)
    parser.add_argument("--max-rows-html", type=int, default=0)
    parser.add_argument("--max-rows-png", type=int, default=3000)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_html = Path(args.output_html) if args.output_html else source_dir / "tradable_window_bs_equity.html"
    output_png = Path(args.output_png) if args.output_png else source_dir / "tradable_window_bs_equity.png"

    build_html(source_dir, output_html, max_rows=args.max_rows_html)
    build_png(source_dir, output_png, max_rows=args.max_rows_png)
    print(f"Saved HTML: {output_html}")
    print(f"Saved PNG: {output_png}")


if __name__ == "__main__":
    main()
