from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from research_context import compute_tradable_context_mask

BUY_COLOR = '#1b9e77'
SELL_COLOR = '#d95f02'
WIN_COLOR = '#2a9d8f'
LOSS_COLOR = '#c1121f'
EARLY_COLOR = '#e9c46a'
STATE_COLOR = '#6a4c93'
BOX_COLOR = '#264653'
MID_COLOR = '#6d597a'
WINDOW_COLOR = 'rgba(69, 123, 157, 0.14)'
WINDOW_FACE = '#d8eef3'
WINDOW_EDGE = '#8fb9c9'


def _load_csv(path: Path, time_cols: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def _contiguous_segments(flag_df: pd.DataFrame, flag_col: str, freq_hours: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    work = flag_df[['timestamp', flag_col]].copy().sort_values('timestamp').reset_index(drop=True)
    segs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev_ts = None
    max_gap = pd.Timedelta(hours=freq_hours)
    for row in work.itertuples(index=False):
        ts = row.timestamp
        flag = bool(getattr(row, flag_col))
        if flag:
            if start is None:
                start = ts
            elif prev_ts is not None and ts - prev_ts > max_gap:
                segs.append((start, prev_ts))
                start = ts
        else:
            if start is not None and prev_ts is not None:
                segs.append((start, prev_ts))
                start = None
        prev_ts = ts
    if start is not None and prev_ts is not None:
        segs.append((start, prev_ts))
    return segs


def _expand_and_merge_segments(segs: list[tuple[pd.Timestamp, pd.Timestamp]], pad_hours: int = 24, merge_gap_hours: int = 24) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not segs:
        return []
    expanded = [(s - pd.Timedelta(hours=pad_hours), e + pd.Timedelta(hours=pad_hours)) for s, e in segs]
    expanded.sort(key=lambda x: x[0])
    merged: list[list[pd.Timestamp]] = [[expanded[0][0], expanded[0][1]]]
    merge_gap = pd.Timedelta(hours=merge_gap_hours)
    for s, e in expanded[1:]:
        cur = merged[-1]
        if s <= cur[1] + merge_gap:
            cur[1] = max(cur[1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _exit_color(reason: str, net_return: float) -> str:
    if reason == 'time_stop_max':
        return WIN_COLOR
    if reason == 'price_stop':
        return LOSS_COLOR
    if reason == 'time_stop_early':
        return EARLY_COLOR
    if reason == 'state_stop':
        return STATE_COLOR
    return WIN_COLOR if net_return >= 0 else LOSS_COLOR


def _candles_matplotlib(ax: plt.Axes, bars: pd.DataFrame) -> None:
    x = mdates.date2num(bars['timestamp'].dt.to_pydatetime())
    width = 0.028
    for xv, o, h, l, c in zip(x, bars['open'], bars['high'], bars['low'], bars['close']):
        color = BUY_COLOR if c >= o else SELL_COLOR
        ax.vlines(xv, l, h, color=color, linewidth=0.6, alpha=0.9)
        body_low = min(o, c)
        body_h = max(abs(c - o), 1e-8)
        ax.add_patch(Rectangle((xv - width / 2, body_low), width, body_h, facecolor=color, edgecolor=color, linewidth=0.4, alpha=0.8))
    ax.xaxis_date()


def _add_trade_lines_plotly(fig: go.Figure, trades: pd.DataFrame, row: int | None = 1, col: int | None = 1) -> None:
    for trade in trades.itertuples(index=False):
        color = WIN_COLOR if trade.net_return >= 0 else LOSS_COLOR
        trace = go.Scatter(
            x=[trade.entry_time, trade.exit_time],
            y=[trade.entry_price, trade.exit_price],
            mode='lines',
            line=dict(color=color, width=1.2, dash='dot'),
            opacity=0.45,
            hoverinfo='skip',
            showlegend=False,
        )
        if row is None or col is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)


def _add_trade_lines_matplotlib(ax: plt.Axes, trades: pd.DataFrame) -> None:
    for trade in trades.itertuples(index=False):
        color = WIN_COLOR if trade.net_return >= 0 else LOSS_COLOR
        ax.plot([trade.entry_time, trade.exit_time], [trade.entry_price, trade.exit_price], color=color, linewidth=0.9, alpha=0.35, linestyle=':')


def _segment_summary(segment_id: str, start: pd.Timestamp, end: pd.Timestamp, context_rows: pd.DataFrame, trades: pd.DataFrame) -> dict:
    exit_counts = trades['exit_reason'].value_counts().to_dict() if not trades.empty else {}
    return {
        'segment_id': segment_id,
        'start': start,
        'end': end,
        'bars': len(context_rows),
        'tradable_window_count': int(context_rows['context_tradeable'].astype(bool).sum()),
        'trade_count': int(len(trades)),
        'win_count': int((trades['net_return'] > 0).sum()) if not trades.empty else 0,
        'loss_count': int((trades['net_return'] <= 0).sum()) if not trades.empty else 0,
        'price_stop_count': int(exit_counts.get('price_stop', 0)),
        'time_stop_max_count': int(exit_counts.get('time_stop_max', 0)),
        'time_stop_early_count': int(exit_counts.get('time_stop_early', 0)),
        'state_stop_count': int(exit_counts.get('state_stop', 0)),
    }


def _prepare_review_data(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars = _load_csv(source_dir / 'local_box_1h.csv', ['timestamp']).sort_values('timestamp').reset_index(drop=True)
    execution = _load_csv(source_dir / 'execution_false_break.csv', ['timestamp']).sort_values('timestamp').reset_index(drop=True)
    trades = _load_csv(source_dir / 'trades_false_break.csv', ['entry_time', 'exit_time']).sort_values('entry_time').reset_index(drop=True)
    equity = _load_csv(source_dir / 'equity_false_break.csv', ['timestamp']).sort_values('timestamp').reset_index(drop=True)
    execution['context_tradeable'] = compute_tradable_context_mask(execution)
    return bars, execution, trades, equity


def _overview_html(out_path: Path, bars: pd.DataFrame, execution: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.74, 0.26], subplot_titles=('Price + Tradable Windows + Executed Trades', 'False-break Equity'))

    fig.add_trace(go.Scattergl(x=bars['timestamp'], y=bars['close'], mode='lines', line=dict(color='#1f2937', width=1.0), name='Close'), row=1, col=1)
    for col_name, trace_name, line in [
        ('box_upper_edge', 'Box Upper', dict(width=0.9, color=BOX_COLOR)),
        ('box_midline', 'Box Mid', dict(width=0.8, color=MID_COLOR, dash='dash')),
        ('box_lower_edge', 'Box Lower', dict(width=0.9, color=BOX_COLOR)),
    ]:
        fig.add_trace(go.Scattergl(x=bars['timestamp'], y=bars[col_name], mode='lines', name=trace_name, line=line, opacity=0.65), row=1, col=1)

    segs = _contiguous_segments(execution[['timestamp', 'context_tradeable']], 'context_tradeable')
    for start, end in segs:
        fig.add_vrect(x0=start, x1=end, fillcolor=WINDOW_COLOR, line_width=0, layer='below', row=1, col=1)

    _add_trade_lines_plotly(fig, trades, row=1, col=1)

    longs = trades[trades['side'] > 0]
    shorts = trades[trades['side'] < 0]
    if not longs.empty:
        fig.add_trace(go.Scattergl(x=longs['entry_time'], y=longs['entry_price'], mode='markers+text', text=['B'] * len(longs), textposition='bottom center', name='Long Entry', marker=dict(symbol='triangle-up', size=9, color=BUY_COLOR), customdata=np.column_stack([longs['net_return'].round(4), longs['exit_reason'].astype(str), longs['hold_bars']]), hovertemplate='<b>Long Entry</b><br>%{x}<br>price=%{y:.2f}<br>net=%{customdata[0]}<br>exit=%{customdata[1]}<br>hold=%{customdata[2]} bars<extra></extra>'), row=1, col=1)
    if not shorts.empty:
        fig.add_trace(go.Scattergl(x=shorts['entry_time'], y=shorts['entry_price'], mode='markers+text', text=['S'] * len(shorts), textposition='top center', name='Short Entry', marker=dict(symbol='triangle-down', size=9, color=SELL_COLOR), customdata=np.column_stack([shorts['net_return'].round(4), shorts['exit_reason'].astype(str), shorts['hold_bars']]), hovertemplate='<b>Short Entry</b><br>%{x}<br>price=%{y:.2f}<br>net=%{customdata[0]}<br>exit=%{customdata[1]}<br>hold=%{customdata[2]} bars<extra></extra>'), row=1, col=1)

    for reason in ['time_stop_max', 'price_stop', 'time_stop_early', 'state_stop']:
        sub = trades[trades['exit_reason'] == reason]
        if sub.empty:
            continue
        fig.add_trace(go.Scattergl(x=sub['exit_time'], y=sub['exit_price'], mode='markers', name=f'Exit: {reason}', marker=dict(symbol='x', size=8, color=_exit_color(reason, 1.0)), customdata=np.column_stack([sub['net_return'].round(4), sub['hold_bars']]), hovertemplate=f'<b>{reason}</b><br>%{{x}}<br>price=%{{y:.2f}}<br>net=%{{customdata[0]}}<br>hold=%{{customdata[1]}} bars<extra></extra>'), row=1, col=1)

    eq = equity.copy()
    if not eq.empty:
        first_ts = bars['timestamp'].iloc[0]
        if eq['timestamp'].iloc[0] > first_ts:
            eq = pd.concat([pd.DataFrame({'timestamp': [first_ts], 'equity': [1.0]}), eq[['timestamp', 'equity']]], ignore_index=True)
    fig.add_trace(go.Scattergl(x=eq['timestamp'], y=eq['equity'], mode='lines', line=dict(color='#1f77b4', width=1.8), name='Equity'), row=2, col=1)

    fig.update_layout(title='Best Baseline Tradable Windows Review', template='plotly_white', height=1100, width=1800, legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0.0), hovermode='x unified')
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Equity', row=2, col=1)
    fig.write_html(str(out_path), include_plotlyjs='cdn')


def _overview_png(out_path: Path, bars: pd.DataFrame, execution: pd.DataFrame, trades: pd.DataFrame, equity: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 13), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(bars['timestamp'], bars['close'], color='#1f2937', linewidth=0.9, label='Close')
    for col_name, label, color, style in [('box_upper_edge', 'Box Upper', BOX_COLOR, '-'), ('box_midline', 'Box Mid', MID_COLOR, '--'), ('box_lower_edge', 'Box Lower', BOX_COLOR, '-')]:
        ax1.plot(bars['timestamp'], bars[col_name], color=color, linewidth=0.8, linestyle=style, alpha=0.65, label=label)
    for start, end in _contiguous_segments(execution[['timestamp', 'context_tradeable']], 'context_tradeable'):
        ax1.axvspan(start, end, color=WINDOW_FACE, alpha=0.55, ec=WINDOW_EDGE, lw=0.0)
    _add_trade_lines_matplotlib(ax1, trades)
    longs = trades[trades['side'] > 0]
    shorts = trades[trades['side'] < 0]
    if not longs.empty:
        ax1.scatter(longs['entry_time'], longs['entry_price'], marker='^', s=35, color=BUY_COLOR, label='Long Entry', zorder=4)
    if not shorts.empty:
        ax1.scatter(shorts['entry_time'], shorts['entry_price'], marker='v', s=35, color=SELL_COLOR, label='Short Entry', zorder=4)
    for reason in ['time_stop_max', 'price_stop', 'time_stop_early', 'state_stop']:
        sub = trades[trades['exit_reason'] == reason]
        if sub.empty:
            continue
        ax1.scatter(sub['exit_time'], sub['exit_price'], marker='x', s=28, color=_exit_color(reason, 1.0), label=f'Exit: {reason}', zorder=4)
    eq = equity.copy()
    if not eq.empty and eq['timestamp'].iloc[0] > bars['timestamp'].iloc[0]:
        eq = pd.concat([pd.DataFrame({'timestamp': [bars['timestamp'].iloc[0]], 'equity': [1.0]}), eq[['timestamp', 'equity']]], ignore_index=True)
    ax2.plot(eq['timestamp'], eq['equity'], color='#1f77b4', linewidth=1.6)
    ax1.set_title('Best Baseline Tradable Windows Review')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Equity')
    ax2.set_xlabel('Time')
    ax1.legend(loc='upper left', ncol=4, fontsize=8)
    ax1.grid(alpha=0.15)
    ax2.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def _segment_html(out_path: Path, bars: pd.DataFrame, context_rows: pd.DataFrame, trades: pd.DataFrame, segment_id: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=bars['timestamp'], open=bars['open'], high=bars['high'], low=bars['low'], close=bars['close'], name='BTC 1H'))
    for col_name, trace_name, line in [('box_upper_edge', 'Box Upper', dict(width=1.1, color=BOX_COLOR)), ('box_midline', 'Box Mid', dict(width=1.0, color=MID_COLOR, dash='dash')), ('box_lower_edge', 'Box Lower', dict(width=1.1, color=BOX_COLOR))]:
        fig.add_trace(go.Scatter(x=bars['timestamp'], y=bars[col_name], mode='lines', name=trace_name, line=line, opacity=0.8))
    for start, end in _contiguous_segments(context_rows[['timestamp', 'context_tradeable']], 'context_tradeable'):
        fig.add_vrect(x0=start, x1=end, fillcolor=WINDOW_COLOR, line_width=0, layer='below')
    _add_trade_lines_plotly(fig, trades, row=None, col=None)
    if not trades.empty:
        longs = trades[trades['side'] > 0]
        shorts = trades[trades['side'] < 0]
        if not longs.empty:
            fig.add_trace(go.Scatter(x=longs['entry_time'], y=longs['entry_price'], mode='markers+text', text=['B'] * len(longs), textposition='bottom center', name='Long Entry', marker=dict(symbol='triangle-up', size=10, color=BUY_COLOR), customdata=np.column_stack([longs['net_return'].round(4), longs['exit_reason'].astype(str), longs['hold_bars']]), hovertemplate='<b>Long Entry</b><br>%{x}<br>price=%{y:.2f}<br>net=%{customdata[0]}<br>exit=%{customdata[1]}<br>hold=%{customdata[2]} bars<extra></extra>'))
        if not shorts.empty:
            fig.add_trace(go.Scatter(x=shorts['entry_time'], y=shorts['entry_price'], mode='markers+text', text=['S'] * len(shorts), textposition='top center', name='Short Entry', marker=dict(symbol='triangle-down', size=10, color=SELL_COLOR), customdata=np.column_stack([shorts['net_return'].round(4), shorts['exit_reason'].astype(str), shorts['hold_bars']]), hovertemplate='<b>Short Entry</b><br>%{x}<br>price=%{y:.2f}<br>net=%{customdata[0]}<br>exit=%{customdata[1]}<br>hold=%{customdata[2]} bars<extra></extra>'))
        for reason in ['time_stop_max', 'price_stop', 'time_stop_early', 'state_stop']:
            sub = trades[trades['exit_reason'] == reason]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(x=sub['exit_time'], y=sub['exit_price'], mode='markers', name=f'Exit: {reason}', marker=dict(symbol='x', size=9, color=_exit_color(reason, 1.0)), customdata=np.column_stack([sub['net_return'].round(4), sub['hold_bars']]), hovertemplate=f'<b>{reason}</b><br>%{{x}}<br>price=%{{y:.2f}}<br>net=%{{customdata[0]}}<br>hold=%{{customdata[1]}} bars<extra></extra>'))
    fig.update_layout(title=f'{segment_id} Tradable Windows + Entries/Exits', template='plotly_white', height=900, width=1700, legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0.0), hovermode='x unified')
    fig.update_yaxes(title_text='Price')
    fig.write_html(str(out_path), include_plotlyjs='cdn')


def _segment_png(out_path: Path, bars: pd.DataFrame, context_rows: pd.DataFrame, trades: pd.DataFrame, segment_id: str) -> None:
    fig, ax = plt.subplots(figsize=(20, 8.5))
    for start, end in _contiguous_segments(context_rows[['timestamp', 'context_tradeable']], 'context_tradeable'):
        ax.axvspan(start, end, color=WINDOW_FACE, alpha=0.58, ec=WINDOW_EDGE, lw=0.0)
    _candles_matplotlib(ax, bars)
    ax.plot(bars['timestamp'], bars['box_upper_edge'], color=BOX_COLOR, linewidth=0.9, alpha=0.85)
    ax.plot(bars['timestamp'], bars['box_midline'], color=MID_COLOR, linewidth=0.8, linestyle='--', alpha=0.9)
    ax.plot(bars['timestamp'], bars['box_lower_edge'], color=BOX_COLOR, linewidth=0.9, alpha=0.85)
    _add_trade_lines_matplotlib(ax, trades)
    if not trades.empty:
        longs = trades[trades['side'] > 0]
        shorts = trades[trades['side'] < 0]
        if not longs.empty:
            ax.scatter(longs['entry_time'], longs['entry_price'], marker='^', s=58, color=BUY_COLOR, zorder=5, label='Long Entry')
        if not shorts.empty:
            ax.scatter(shorts['entry_time'], shorts['entry_price'], marker='v', s=58, color=SELL_COLOR, zorder=5, label='Short Entry')
        for reason in ['time_stop_max', 'price_stop', 'time_stop_early', 'state_stop']:
            sub = trades[trades['exit_reason'] == reason]
            if sub.empty:
                continue
            ax.scatter(sub['exit_time'], sub['exit_price'], marker='x', s=42, color=_exit_color(reason, 1.0), zorder=5, label=f'Exit: {reason}')
    ax.set_title(f'{segment_id} Tradable Windows + Entries/Exits')
    ax.set_ylabel('Price')
    ax.grid(alpha=0.14)
    ax.legend(loc='upper left', ncol=4, fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def build_visual_review(source_dir: Path, out_root: Path) -> None:
    bars, execution, trades, equity = _prepare_review_data(source_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seg_dir = out_root / 'best_baseline_tradable_windows_segmented'
    seg_dir.mkdir(parents=True, exist_ok=True)

    _overview_html(out_root / 'best_baseline_tradable_windows_overview.html', bars, execution, trades, equity)
    _overview_png(out_root / 'best_baseline_tradable_windows_overview.png', bars, execution, trades, equity)

    base_segs = _contiguous_segments(execution[['timestamp', 'context_tradeable']], 'context_tradeable')
    inspect_segs = _expand_and_merge_segments(base_segs, pad_hours=24, merge_gap_hours=24)
    summaries: list[dict] = []
    for idx, (start, end) in enumerate(inspect_segs, start=1):
        segment_id = f'segment_{idx:03d}'
        bars_seg = bars[(bars['timestamp'] >= start) & (bars['timestamp'] <= end)].copy()
        ctx_seg = execution[(execution['timestamp'] >= start) & (execution['timestamp'] <= end)].copy()
        trades_seg = trades[(trades['entry_time'] <= end) & (trades['exit_time'] >= start)].copy()
        if bars_seg.empty:
            continue
        _segment_html(seg_dir / f'{segment_id}.html', bars_seg, ctx_seg, trades_seg, segment_id)
        _segment_png(seg_dir / f'{segment_id}.png', bars_seg, ctx_seg, trades_seg, segment_id)
        summaries.append(_segment_summary(segment_id, start, end, ctx_seg, trades_seg))
    pd.DataFrame(summaries).to_csv(out_root / 'visualization_index.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate close-out visual review package for the best baseline tradable windows.')
    parser.add_argument('--source-dir', default='outputs/third_round_capture_recovery/baseline_master_current')
    parser.add_argument('--outdir', default='outputs/third_round_capture_recovery')
    args = parser.parse_args()
    build_visual_review(Path(args.source_dir), Path(args.outdir))


if __name__ == '__main__':
    main()


