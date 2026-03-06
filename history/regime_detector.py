#!/usr/bin/env python3
"""Sequential wide-range regime detector for 4H BTC data.

This detector is intentionally separated from the executor:
- Layer 1: detect tradeable ranging regimes
- Layer 2: run the MR executor only inside detected windows

No-lookahead policy:
- Indicators use only bar t and earlier data
- Regime ON is confirmed after N consecutive qualifying bars and becomes active
  from the next bar timestamp
- Regime OFF is confirmed on bar t close and ends the regime at bar t
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from manual_box_roundX import calc_adx14, calc_atr14, calc_bollinger, calc_choppiness, load_data, parse_ts_utc


@dataclass
class DetectedRegime:
    box_id: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    box_low: float
    box_high: float
    seed_low: float
    seed_high: float
    anchor_atr: float
    risk_high_candidates: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect wide-ranging consolidation regimes and export boxes-csv.")
    parser.add_argument("--csv", type=str, required=True, help="4H OHLCV csv path")
    parser.add_argument("--output", type=str, default="auto_detected_boxes.csv", help="output boxes csv path")
    parser.add_argument("--html", type=str, default="auto_detected_boxes.html", help="output html path")
    parser.add_argument("--start", type=str, default="", help="optional UTC start timestamp")
    parser.add_argument("--end", type=str, default="", help="optional UTC end timestamp")
    parser.add_argument("--confirm-bars", type=int, default=3, help="consecutive bars required to switch ON")
    parser.add_argument("--off-confirm-bars", type=int, default=2, help="consecutive OFF bars required to switch OFF")
    parser.add_argument("--min-regime-days", type=float, default=10.0, help="drop regimes shorter than this")
    parser.add_argument("--seed-anchor-bars", type=int, default=24, help="bars used to lock seed geometry at confirmation")
    parser.add_argument("--expand-atr-mult", type=float, default=3.0, help="ATR multiple used to expand seed zone")
    parser.add_argument("--merge-gap-days", type=float, default=5.0, help="merge adjacent regimes when the gap is short")
    parser.add_argument("--merge-overlap-min", type=float, default=0.25, help="minimum box overlap ratio for merge")
    parser.add_argument("--on-adx-max", type=float, default=28.0, help="regime ON requires ADX14 < threshold")
    parser.add_argument("--on-bbwidth-min", type=float, default=0.05, help="regime ON requires BBWidth20 > threshold")
    parser.add_argument("--on-chop-min", type=float, default=48.0, help="regime ON requires CHOP72 > threshold")
    parser.add_argument("--breakdown-adx-min", type=float, default=40.0, help="regime breaks when ADX14 exceeds this")
    parser.add_argument("--id-prefix", type=str, default="auto_box", help="box id prefix")
    return parser


def _box_overlap_ratio(a: DetectedRegime, b: DetectedRegime) -> float:
    overlap = max(0.0, min(a.box_high, b.box_high) - max(a.box_low, b.box_low))
    width_ref = max(min(a.box_high - a.box_low, b.box_high - b.box_low), 1e-9)
    return float(overlap / width_ref)


def _merge_regimes(
    regimes: List[DetectedRegime],
    max_gap_days: float,
    min_overlap_ratio: float,
    id_prefix: str,
) -> List[DetectedRegime]:
    if not regimes:
        return []
    merged: List[DetectedRegime] = [regimes[0]]
    for reg in regimes[1:]:
        prev = merged[-1]
        gap_days = float((reg.start_ts - prev.end_ts).total_seconds() / 86400.0)
        overlap_ratio = _box_overlap_ratio(prev, reg)
        if gap_days <= float(max_gap_days) and overlap_ratio >= float(min_overlap_ratio):
            merged[-1] = DetectedRegime(
                box_id=prev.box_id,
                start_ts=prev.start_ts,
                end_ts=max(prev.end_ts, reg.end_ts),
                box_low=min(prev.box_low, reg.box_low),
                box_high=max(prev.box_high, reg.box_high),
                seed_low=min(prev.seed_low, reg.seed_low),
                seed_high=max(prev.seed_high, reg.seed_high),
                anchor_atr=max(prev.anchor_atr, reg.anchor_atr),
                risk_high_candidates="",
            )
        else:
            merged.append(reg)
    out: List[DetectedRegime] = []
    for idx, reg in enumerate(merged, start=1):
        out.append(
            DetectedRegime(
                box_id=f"{id_prefix}_{reg.start_ts.strftime('%Y%m%d_%H%M')}_{idx:03d}",
                start_ts=reg.start_ts,
                end_ts=reg.end_ts,
                box_low=reg.box_low,
                box_high=reg.box_high,
                seed_low=reg.seed_low,
                seed_high=reg.seed_high,
                anchor_atr=reg.anchor_atr,
                risk_high_candidates="",
            )
        )
    return out


def detect_regimes(
    df: pd.DataFrame,
    confirm_bars: int,
    off_confirm_bars: int,
    min_regime_days: float,
    seed_anchor_bars: int,
    expand_atr_mult: float,
    merge_gap_days: float,
    merge_overlap_min: float,
    on_adx_max: float,
    on_bbwidth_min: float,
    on_chop_min: float,
    breakdown_adx_min: float,
    id_prefix: str,
) -> List[DetectedRegime]:
    work = df.copy()
    work["atr14"] = calc_atr14(work)
    work["adx14"] = calc_adx14(work)
    _, _, _, bb_width20 = calc_bollinger(work, window=20, num_std=2.0)
    work["bb_width20"] = bb_width20
    work["chop72"] = calc_choppiness(work, window=72)

    ts = work.index.to_numpy()
    high = work["high"].to_numpy(dtype=float)
    low = work["low"].to_numpy(dtype=float)
    adx = work["adx14"].to_numpy(dtype=float)
    atr = work["atr14"].to_numpy(dtype=float)
    bb_width = work["bb_width20"].to_numpy(dtype=float)
    chop = work["chop72"].to_numpy(dtype=float)
    close = work["close"].to_numpy(dtype=float)

    regimes: List[DetectedRegime] = []
    in_regime = False
    on_streak = 0
    off_streak = 0
    regime_start_idx: Optional[int] = None
    regime_start_ts: Optional[pd.Timestamp] = None
    seed_low: Optional[float] = None
    seed_high: Optional[float] = None
    anchor_atr: Optional[float] = None
    regime_seq = 0

    for i in range(len(work)):
        on_cond = bool(
            np.isfinite(adx[i])
            and np.isfinite(atr[i])
            and np.isfinite(bb_width[i])
            and np.isfinite(chop[i])
            and (adx[i] < float(on_adx_max))
            and (bb_width[i] > float(on_bbwidth_min))
            and (chop[i] > float(on_chop_min))
        )

        if not in_regime:
            on_streak = (on_streak + 1) if on_cond else 0
            if on_streak >= max(int(confirm_bars), 1):
                next_i = i + 1
                if next_i >= len(work):
                    break
                anchor_start = max(0, i - max(int(seed_anchor_bars), 1) + 1)
                anchor_end = i + 1
                seed_low_locked = float(np.nanmin(low[anchor_start:anchor_end]))
                seed_high_locked = float(np.nanmax(high[anchor_start:anchor_end]))
                anchor_atr_locked = float(atr[i])
                if (
                    (not np.isfinite(seed_low_locked))
                    or (not np.isfinite(seed_high_locked))
                    or (not np.isfinite(anchor_atr_locked))
                    or (seed_high_locked <= seed_low_locked)
                ):
                    on_streak = 0
                    continue
                in_regime = True
                regime_start_idx = next_i
                regime_start_ts = pd.Timestamp(ts[next_i])
                seed_low = seed_low_locked
                seed_high = seed_high_locked
                anchor_atr = anchor_atr_locked
                on_streak = 0
                off_streak = 0
            continue

        assert seed_low is not None and seed_high is not None and anchor_atr is not None
        tol_low = float(seed_low - float(expand_atr_mult) * anchor_atr)
        tol_high = float(seed_high + float(expand_atr_mult) * anchor_atr)
        breakdown_cond = bool(
            (np.isfinite(close[i]) and ((close[i] < tol_low) or (close[i] > tol_high)))
            or (np.isfinite(adx[i]) and (adx[i] > float(breakdown_adx_min)))
        )

        off_streak = (off_streak + 1) if breakdown_cond else 0
        if off_streak >= max(int(off_confirm_bars), 1):
            if regime_start_idx is not None and regime_start_ts is not None:
                end_ts = pd.Timestamp(ts[i])
                regime_days = float((end_ts - regime_start_ts).total_seconds() / 86400.0)
                if regime_days >= float(min_regime_days):
                    if tol_high > tol_low:
                        regime_seq += 1
                        regimes.append(
                            DetectedRegime(
                                box_id=f"{id_prefix}_{regime_start_ts.strftime('%Y%m%d_%H%M')}_{regime_seq:03d}",
                                start_ts=regime_start_ts,
                                end_ts=end_ts,
                                box_low=tol_low,
                                box_high=tol_high,
                                seed_low=float(seed_low),
                                seed_high=float(seed_high),
                                anchor_atr=float(anchor_atr),
                                risk_high_candidates="",
                            )
                        )
            in_regime = False
            regime_start_idx = None
            regime_start_ts = None
            seed_low = None
            seed_high = None
            anchor_atr = None
            on_streak = 0
            off_streak = 0

    if in_regime and regime_start_idx is not None and regime_start_ts is not None and seed_low is not None and seed_high is not None and anchor_atr is not None:
        end_ts = pd.Timestamp(ts[-1])
        regime_days = float((end_ts - regime_start_ts).total_seconds() / 86400.0)
        tol_low = float(seed_low - float(expand_atr_mult) * anchor_atr)
        tol_high = float(seed_high + float(expand_atr_mult) * anchor_atr)
        if regime_days >= float(min_regime_days):
            if tol_high > tol_low:
                regime_seq += 1
                regimes.append(
                    DetectedRegime(
                        box_id=f"{id_prefix}_{regime_start_ts.strftime('%Y%m%d_%H%M')}_{regime_seq:03d}",
                        start_ts=regime_start_ts,
                        end_ts=end_ts,
                        box_low=tol_low,
                        box_high=tol_high,
                        seed_low=float(seed_low),
                        seed_high=float(seed_high),
                        anchor_atr=float(anchor_atr),
                        risk_high_candidates="",
                    )
                )
    return _merge_regimes(
        regimes=regimes,
        max_gap_days=merge_gap_days,
        min_overlap_ratio=merge_overlap_min,
        id_prefix=id_prefix,
    )


def build_visual(df: pd.DataFrame, regimes: List[DetectedRegime], html_path: Path) -> None:
    work = df.copy()
    work["adx14"] = calc_adx14(work)
    _, _, _, bb_width20 = calc_bollinger(work, window=20, num_std=2.0)
    work["bb_width20"] = bb_width20
    work["chop72"] = calc_choppiness(work, window=72)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=("Price + Auto Detected Boxes", "Regime Inputs"),
    )
    fig.add_trace(
        go.Candlestick(
            x=work.index,
            open=work["open"],
            high=work["high"],
            low=work["low"],
            close=work["close"],
            name="BTC 4H",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=work.index, y=work["adx14"], mode="lines", name="ADX14", line=dict(color="#d62728")), row=2, col=1)
    fig.add_trace(go.Scatter(x=work.index, y=work["bb_width20"] * 100.0, mode="lines", name="BBWidth20 %", line=dict(color="#1f77b4")), row=2, col=1)
    fig.add_trace(go.Scatter(x=work.index, y=work["chop72"], mode="lines", name="CHOP72", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_hline(y=25.0, row=2, col=1, line_dash="dot", line_color="#d62728", opacity=0.5)
    fig.add_hline(y=50.0, row=2, col=1, line_dash="dot", line_color="#2ca02c", opacity=0.5)
    fig.add_hline(y=5.0, row=2, col=1, line_dash="dot", line_color="#1f77b4", opacity=0.5)

    palette = [
        "rgba(31,119,180,0.16)",
        "rgba(255,127,14,0.16)",
        "rgba(44,160,44,0.16)",
        "rgba(214,39,40,0.16)",
        "rgba(148,103,189,0.16)",
        "rgba(140,86,75,0.16)",
    ]
    for idx, regime in enumerate(regimes):
        fill = palette[idx % len(palette)]
        fig.add_vrect(
            x0=regime.start_ts,
            x1=regime.end_ts,
            fillcolor=fill,
            opacity=1.0,
            layer="below",
            line_width=0,
            row=1,
            col=1,
        )
        fig.add_shape(
            type="rect",
            x0=regime.start_ts,
            x1=regime.end_ts,
            y0=regime.box_low,
            y1=regime.box_high,
            xref="x",
            yref="y",
            line=dict(color=fill.replace("0.16", "0.85"), width=2),
            fillcolor=fill.replace("0.16", "0.08"),
            row=1,
            col=1,
        )
        mid_ts = regime.start_ts + (regime.end_ts - regime.start_ts) / 2
        fig.add_annotation(
            x=mid_ts,
            y=regime.box_high,
            xref="x",
            yref="y",
            text=regime.box_id,
            showarrow=False,
            yshift=10,
            font=dict(size=10, color="#111"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=fill.replace("0.16", "0.85"),
            borderwidth=1,
        )
        fig.add_shape(
            type="rect",
            x0=regime.start_ts,
            x1=regime.end_ts,
            y0=regime.seed_low,
            y1=regime.seed_high,
            xref="x",
            yref="y",
            line=dict(color=fill.replace("0.16", "1.0"), width=1, dash="dot"),
            fillcolor="rgba(0,0,0,0)",
            row=1,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=980,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        title=f"Auto Detected Ranging Regimes ({len(regimes)} boxes)",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="ADX / BBWidth% / CHOP", row=2, col=1)
    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)


def main() -> int:
    args = build_parser().parse_args()
    csv_path = Path(args.csv).expanduser()
    out_path = Path(args.output).expanduser()
    html_path = Path(args.html).expanduser()

    df = load_data(csv_path)
    if args.start:
        start_utc = parse_ts_utc(args.start)
        df = df.loc[df.index >= start_utc].copy()
    if args.end:
        end_utc = parse_ts_utc(args.end)
        df = df.loc[df.index <= end_utc].copy()
    if df.empty:
        raise SystemExit("no data in selected window")

    regimes = detect_regimes(
        df=df,
        confirm_bars=max(int(args.confirm_bars), 1),
        off_confirm_bars=max(int(args.off_confirm_bars), 1),
        min_regime_days=max(float(args.min_regime_days), 0.0),
        seed_anchor_bars=max(int(args.seed_anchor_bars), 1),
        expand_atr_mult=max(float(args.expand_atr_mult), 0.0),
        merge_gap_days=max(float(args.merge_gap_days), 0.0),
        merge_overlap_min=min(max(float(args.merge_overlap_min), 0.0), 1.0),
        on_adx_max=float(args.on_adx_max),
        on_bbwidth_min=max(float(args.on_bbwidth_min), 0.0),
        on_chop_min=float(args.on_chop_min),
        breakdown_adx_min=float(args.breakdown_adx_min),
        id_prefix=str(args.id_prefix).strip() or "auto_box",
    )

    rows = [
        {
            "box_id": r.box_id,
            "start_ts": r.start_ts.isoformat(),
            "end_ts": r.end_ts.isoformat(),
            "box_low": round(r.box_low, 6),
            "box_high": round(r.box_high, 6),
            "risk_high_candidates": r.risk_high_candidates,
        }
        for r in regimes
    ]
    out_df = pd.DataFrame(rows, columns=["box_id", "start_ts", "end_ts", "box_low", "box_high", "risk_high_candidates"])
    out_df.to_csv(out_path, index=False)
    build_visual(df=df, regimes=regimes, html_path=html_path)

    print(f"detected_regimes={len(out_df)}")
    print(f"output={out_path}")
    print(f"html={html_path}")
    if not out_df.empty:
        print(out_df.head(min(10, len(out_df))).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
