#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _decorator(func):
            return func

        return _decorator


DEFAULT_CSV = "/mnt/data/OKX_BTCUSDT.P, 240.csv"
FALLBACK_CSV = "OKX_BTCUSDT.P, 240.csv"
DEFAULT_START = "2026-02-06 00:00:00+00:00"
DEFAULT_END = "2026-02-10 00:00:00+00:00"
DEFAULT_OUTPUT_DIR = "/mnt/data"
FALLBACK_OUTPUT_DIR = "manual_box"
__version__ = "v4.2-stable"
__git_hash__ = ""


def frange(start: float, end: float, step: float) -> List[float]:
    n = int(round((end - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


LONG_L_GRID = frange(0.20, 0.40, 0.02)
MID1_GRID = frange(0.50, 0.60, 0.02)
BASE_RISK_GRID = frange(0.15, 0.45, 0.05)
ATR_STOP_MULT_GRID = frange(1.2, 1.8, 0.2)
MACRO_LOOKBACK_GRID = [56, 70, 84, 98, 112]

# Local grid around the current best neighborhood.
LONG_L_LOCAL = frange(0.34, 0.44, 0.02)
MID1_LOCAL = frange(0.56, 0.64, 0.02)
BASE_RISK_LOCAL = [0.25, 0.30, 0.35, 0.40]
ATR_STOP_MULT_LOCAL = [1.3, 1.5, 1.7]
MACRO_LOOKBACK_LOCAL = [70, 84, 98]

BOX_LOW = 60000.0
BOX_HIGH = 71700.0
BOX_WIDTH = BOX_HIGH - BOX_LOW
PROBE_MIN_ORDERS_TOTAL = 1
_PROBE_SELECTION_CACHE: Dict[Tuple[object, ...], Dict[str, object]] = {}


@dataclass(frozen=True)
class Params:
    long_L: float
    mid1: float
    base_risk_pct: float
    atr_stop_mult: float
    macro_lookback_bars: int
    layer_weights_value: Tuple[float, float, float] = (1.0, 1.4, 2.0)
    max_leverage_value: float = 2.0

    @property
    def mid2(self) -> float:
        return min(self.mid1 + 0.04, 0.95)

    @property
    def layer_weights(self) -> Tuple[float, float, float]:
        return self.layer_weights_value

    @property
    def layer_step(self) -> float:
        return 0.08

    @property
    def max_leverage(self) -> float:
        return self.max_leverage_value

    @property
    def hard_stop_pct(self) -> float:
        return 0.05

    @property
    def cooldown_rearm_bars(self) -> int:
        return 1

    @property
    def cooldown_stop_bars(self) -> int:
        return 24

    @property
    def time_stop_bars(self) -> int:
        return 10

    @property
    def maker_ttl_bars(self) -> int:
        return 2

    @property
    def maker_fallback_taker(self) -> bool:
        return True

    @property
    def improve_mult(self) -> float:
        return 0.10

    @property
    def min_improve_bps(self) -> float:
        return 1.0

    @property
    def trade_box_mode(self) -> str:
        return "rolling_quantile"

    @property
    def trade_box_lookback(self) -> int:
        return 60

    @property
    def trade_box_q_low(self) -> float:
        return 0.20

    @property
    def trade_box_q_high(self) -> float:
        return 0.80

    @property
    def trade_box_ema_len(self) -> int:
        return 34

    @property
    def trade_box_atr_mult(self) -> float:
        return 1.0


@dataclass(frozen=True)
class DynamicBoxConfig:
    mode: str
    break_confirm_closes: int
    break_buffer_mode: str
    break_buffer_pct: float
    break_buffer_atr_mult: float
    expand_atr_mult: float
    invalidate_force_close: bool
    max_expands: int
    freeze_after_expand_bars: int


@dataclass(frozen=True)
class RunConfig:
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    commission_pct: float
    slippage_pct: float
    box_low: float
    box_high: float
    dyn_cfg: DynamicBoxConfig
    side_mode: str
    trade_box_mode_default: str
    trade_box_lookback_default: int
    trade_box_q_low_default: float
    trade_box_q_high_default: float
    trade_box_ema_len_default: int
    trade_box_atr_mult_default: float
    risk_high_candidates: Tuple[float, ...]
    probe_days: int
    probe_metric: str
    probe_execution_mode: str
    fallback_no_signal_mode: str
    probe_tie_eps: float
    probe_min_round_trips: int
    probe_min_entry_fill_rate: float
    probe_max_hard_stop_ratio: float
    early_stop_first_k_trades: int
    early_stop_hard_stop_threshold: int
    early_stop_first_m_bars: int
    enable_early_stop: bool
    entry_execution_mode: str
    maker_fill_prob: float
    maker_queue_delay_bars: int
    seed: int
    min_round_trips: int
    pf_unreliable_penalty: float
    degrade_on_trend: bool
    trend_slope_thresh: float
    atr_expand_thresh: float
    degrade_risk_lookback_bars: int
    short_enable_rule: str = "none"
    short_enable_lookback_days: int = 10
    short_enable_min_rejects: int = 2
    short_enable_touch_band: float = 0.10
    short_enable_reject_close_gap: float = 0.02
    start_gate_mode: str = "on"
    gate_adx_thresh: float = 25.0
    gate_ema_slope_thresh: float = 0.0015
    gate_chop_thresh: float = 55.0
    gate_edge_reject_lookback_bars: int = 60
    gate_edge_reject_min_count: int = 2
    gate_edge_reject_atr_mult: float = 0.5
    invalidate_mode: str = "on"
    invalidate_m: int = 2
    invalidate_buffer_mode: str = "atr"
    invalidate_buffer_atr_mult: float = 1.0
    invalidate_buffer_pct: float = 0.15
    invalidate_action: str = "disable_only"
    cooldown_after_invalidate_bars: int = 24
    perf_stop_mode: str = "on"
    perf_window_trades: int = 12
    perf_min_profit_factor: float = 1.0
    perf_max_hard_stop_ratio: float = 0.30
    perf_action: str = "disable_only"
    cooldown_after_perf_stop_bars: int = 24
    box_source: str = "dynamic"
    clip_dynamic_to_manual: bool = False
    macro_box_mode: str = "donchian"
    macro_lookback_bars: int = 84
    macro_bb_len: int = 84
    macro_bb_std: float = 2.0
    regime_gate_mode: str = "off"
    regime_gate_adx_thresh: float = 22.0
    regime_gate_bbwidth_min: float = 0.05
    regime_gate_chop_thresh: float = 50.0
    regime_gate_bbwidth_q_thresh: float = 0.30
    regime_gate_slope_thresh: float = 0.0015
    circuit_breaker_mode: str = "off"
    circuit_break_adx_thresh: float = 25.0
    circuit_break_bbwidth_q_thresh: float = 0.50
    circuit_break_outside_consecutive: int = 2
    cooldown_cb_bars: int = 8
    cb_force_flatten: bool = False
    atr_stop_mode: str = "off"
    atr_stop_mult: float = 1.5
    structural_cooldown_bars: int = 6
    local_time_stop_bars: int = 30
    enable_runner: bool = False
    runner_pct: float = 0.20
    runner_atr_mult: float = 2.0
    complexity_penalty_lambda: float = 0.35
    override_base_risk_pct: Optional[float] = None
    override_layer_weights: Optional[Tuple[float, float, float]] = None
    override_max_leverage: Optional[float] = None
    history_start_utc: Optional[pd.Timestamp] = None
    initial_equity: float = 10000.0


def resolve_csv(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    fb = Path(FALLBACK_CSV)
    if fb.exists():
        return fb
    raise FileNotFoundError(f"CSV not found: {p} (fallback {fb} missing)")


def resolve_output_dir(path: str) -> Tuple[Path, Optional[str]]:
    out = Path(path)
    try:
        out.mkdir(parents=True, exist_ok=True)
        return out, None
    except Exception as exc:  # noqa: BLE001
        fb = Path(FALLBACK_OUTPUT_DIR)
        fb.mkdir(parents=True, exist_ok=True)
        return fb, f"Output dir {out} unavailable ({exc}); fallback to {fb.resolve()}"


def infer_column(df: pd.DataFrame, keys: List[str]) -> str:
    norm_map = {}
    for c in df.columns:
        k = "".join(ch for ch in str(c).lower() if ch.isalnum())
        norm_map[k] = c
    for key in keys:
        if key in norm_map:
            return norm_map[key]
    for k, c in norm_map.items():
        for key in keys:
            if key in k:
                return c
    raise KeyError(f"Cannot infer column from keys={keys}, columns={list(df.columns)}")


def load_data(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)

    c_time = infer_column(raw, ["time", "timestamp", "datetime", "date"])
    c_open = infer_column(raw, ["open", "o"])
    c_high = infer_column(raw, ["high", "h"])
    c_low = infer_column(raw, ["low", "l"])
    c_close = infer_column(raw, ["close", "c"])
    c_vol = infer_column(raw, ["volume", "vol", "v"])

    df = raw[[c_time, c_open, c_high, c_low, c_close, c_vol]].copy()
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return df.set_index("timestamp")


def calc_atr14(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()


def calc_adx14(df: pd.DataFrame) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean() / atr.replace(0.0, np.nan)
    minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean() / atr.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()


def calc_choppiness(df: pd.DataFrame, window: int = 72) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_sum = tr.rolling(window=window, min_periods=window).sum()
    hh = df["high"].rolling(window=window, min_periods=window).max()
    ll = df["low"].rolling(window=window, min_periods=window).min()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log10(tr_sum / denom) / np.log10(float(window))


def calc_bollinger(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = df["close"].rolling(window=window, min_periods=window).mean()
    std = df["close"].rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    return mid, upper, lower, width


def calc_max_dd_pct(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0.0, np.nan)
    return float(dd.max() * 100.0)


def parse_ts_utc(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def parse_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def parse_layer_weights(s: str) -> Tuple[float, float, float]:
    txt = str(s).strip()
    if txt.startswith(("(", "[")) and txt.endswith((")", "]")):
        txt = txt[1:-1]
    vals = [float(x.strip()) for x in txt.split(",")]
    if len(vals) != 3:
        raise ValueError(f"layer_weights must have 3 floats, got: {s}")
    return vals[0], vals[1], vals[2]


def parse_float_list(s: str) -> Tuple[float, ...]:
    txt = str(s).strip()
    if txt.lower() in {"", "nan", "none", "null"}:
        return tuple()
    vals: List[float] = []
    for part in txt.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    return tuple(vals)


def clone_cfg(cfg: RunConfig, **updates: object) -> RunConfig:
    data = dict(cfg.__dict__)
    data.update(updates)
    return RunConfig(**data)


def required_history_bars(cfg: RunConfig, p: Optional[Params] = None) -> int:
    macro_lb = int(p.macro_lookback_bars) if p is not None else int(cfg.macro_lookback_bars)
    base_lookbacks = [
        max(int(cfg.trade_box_lookback_default), 2),
        max(int(cfg.macro_bb_len), 2),
        max(macro_lb, 2),
        120,
        72,
        50,
        20,
        max(int(cfg.gate_edge_reject_lookback_bars), 1),
        max(int(cfg.degrade_risk_lookback_bars), 1),
    ]
    return max(base_lookbacks) + 12


def resolve_history_start_utc(df: pd.DataFrame, active_start_utc: pd.Timestamp, history_bars: int) -> pd.Timestamp:
    if df.empty:
        return active_start_utc
    active_pos = int(df.index.searchsorted(active_start_utc, side="left"))
    history_pos = max(active_pos - max(int(history_bars), 0), 0)
    return pd.Timestamp(df.index[history_pos])


def derive_wfa_output_paths(out_path: Path) -> Dict[str, Path]:
    suffix = "_walk_forward_summary.csv"
    if out_path.name.endswith(suffix):
        stem = out_path.name[: -len(suffix)]
    else:
        stem = out_path.stem
    return {
        "equity": out_path.with_name(f"{stem}_wfa_combined_equity.csv"),
        "trades": out_path.with_name(f"{stem}_wfa_combined_trades.csv"),
        "bars": out_path.with_name(f"{stem}_wfa_combined_bars.csv"),
        "visual": out_path.with_name(f"{stem}_wfa_combined_visual.html"),
    }


def resolve_git_hash() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fixed_defaults_summary() -> Dict[str, object]:
    return {
        "layer_weights": [1.0, 1.4, 2.0],
        "max_leverage": 2.0,
        "cooldown_rearm_bars": 1,
        "cooldown_stop_bars": 24,
        "time_stop_bars": 30,
        "maker_fallback_taker": True,
        "improve_mult": 0.10,
        "box_source": "dynamic",
        "clip_dynamic_to_manual": False,
        "macro_box_mode": "donchian",
        "macro_lookback_bars": 84,
        "regime_gate": "off",
        "regime_gate_adx_thresh": 22.0,
        "regime_gate_bbwidth_min": 0.05,
        "regime_gate_chop_thresh": 50.0,
        "circuit_breaker": "off",
        "atr_stop": "off",
        "structural_cooldown_bars": 6,
        "enable_runner": False,
        "runner_pct": 0.20,
        "runner_atr_mult": 2.0,
    }


def build_archive_manifest(
    args: argparse.Namespace,
    run_tag: str,
    cfg: RunConfig,
    best_params: Optional[Params],
    best_stats: Optional[Dict[str, object]],
    release_gate_status: str,
    release_gate_reasons: List[str],
    files: List[Path],
) -> Dict[str, object]:
    selected_high = None
    selected_source = ""
    tie_reason = ""
    if best_stats is not None:
        selected_high = best_stats.get("selected_risk_high")
        tie_reason = str(best_stats.get("probe_tie_reason", ""))
        selected_source = "fallback" if tie_reason.startswith("FALLBACK") else ("probe" if tie_reason else "unknown")
    file_entries = []
    for p in files:
        if not p.exists():
            continue
        file_entries.append(
            {
                "path": str(p),
                "size_bytes": int(p.stat().st_size),
                "sha256": sha256_file(p),
            }
        )
    return {
        "version": __version__,
        "git_hash": __git_hash__,
        "run_tag": run_tag,
        "utc_generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cli_args": vars(args),
        "fixed_defaults": fixed_defaults_summary(),
        "data_window_utc": {"start": str(cfg.start_utc), "end": str(cfg.end_utc)},
        "box_source": str(cfg.box_source),
        "clip_dynamic_to_manual": bool(cfg.clip_dynamic_to_manual),
        "macro_box": {
            "mode": str(cfg.macro_box_mode),
            "lookback_bars": int(cfg.macro_lookback_bars),
            "bb_len": int(cfg.macro_bb_len),
            "bb_std": float(cfg.macro_bb_std),
        },
        "box": {"low": float(cfg.box_low), "high": float(cfg.box_high)},
        "risk_high_candidates": [float(x) for x in cfg.risk_high_candidates],
        "probe_days": int(cfg.probe_days),
        "selected_high": selected_high,
        "selected_source": selected_source,
        "probe_tie_reason": tie_reason,
        "release_gate": {
            "status": release_gate_status,
            "not_robust": release_gate_status == "NOT_ROBUST",
            "reasons": release_gate_reasons,
        },
        "best_params": dict(best_params.__dict__) if best_params is not None else None,
        "files": file_entries,
    }


def create_archive_zip(
    out_dir: Path,
    run_tag: str,
    manifest: Dict[str, object],
    files: List[Path],
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    zip_name = f"manual_box_v4_2_{run_tag}_{ts}.zip"
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("run_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for p in files:
            if not p.exists():
                continue
            arcname = p.name if p.name != "manual_box_roundX.py" else "manual_box_roundX.py"
            zf.write(p, arcname=arcname)
    return zip_path


def build_run_seed(cfg: RunConfig, p: Params) -> int:
    base = (
        f"{cfg.seed}|{cfg.start_utc}|{cfg.end_utc}|{cfg.box_low}|{cfg.box_high}|"
        f"{cfg.box_source}|{cfg.macro_box_mode}|{cfg.macro_lookback_bars}|{cfg.macro_bb_len}|{cfg.macro_bb_std}|"
        f"{cfg.side_mode}|{cfg.entry_execution_mode}|{cfg.maker_fill_prob}|{cfg.maker_queue_delay_bars}|"
        f"{cfg.regime_gate_mode}|{cfg.regime_gate_adx_thresh}|{cfg.regime_gate_bbwidth_min}|{cfg.regime_gate_chop_thresh}|"
        f"{cfg.circuit_breaker_mode}|{cfg.atr_stop_mode}|{cfg.atr_stop_mult}|{cfg.structural_cooldown_bars}|"
        f"{cfg.enable_runner}|{cfg.runner_pct}|{cfg.runner_atr_mult}|"
        f"{p.long_L}|{p.mid1}|{p.mid2}|{p.base_risk_pct}|{p.atr_stop_mult}|{p.macro_lookback_bars}|{p.layer_step}|{p.hard_stop_pct}|"
        f"{p.maker_ttl_bars}|{p.maker_fallback_taker}|{p.improve_mult}|{p.min_improve_bps}|"
        f"{p.trade_box_mode}|{p.trade_box_lookback}|{p.trade_box_q_low}|{p.trade_box_q_high}|{p.trade_box_ema_len}|{p.trade_box_atr_mult}"
    )
    digest = hashlib.sha256(base.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def param_complexity_score(p: Params) -> float:
    # L1-style complexity around conservative defaults; lower is simpler.
    c_long = abs(float(p.long_L) - 0.36) / 0.12
    c_mid1 = abs(float(p.mid1) - 0.60) / 0.10
    c_risk = abs(float(p.base_risk_pct) - 0.30) / 0.20
    c_atr = abs(float(p.atr_stop_mult) - 1.5) / 0.6
    c_macro = abs(float(p.macro_lookback_bars) - 84.0) / 56.0
    return float((c_long + c_mid1 + c_risk + c_atr + c_macro) / 5.0)


def build_param_grid(max_combos: int, scan_mode: str, cfg: RunConfig) -> Tuple[List[Params], int]:
    base_risk_dim_local = [float(cfg.override_base_risk_pct)] if cfg.override_base_risk_pct is not None else BASE_RISK_LOCAL
    base_risk_dim_grid = [float(cfg.override_base_risk_pct)] if cfg.override_base_risk_pct is not None else BASE_RISK_GRID
    if scan_mode == "local":
        dims: List[List] = [
            LONG_L_LOCAL,
            MID1_LOCAL,
            base_risk_dim_local,
            ATR_STOP_MULT_LOCAL,
            MACRO_LOOKBACK_LOCAL,
        ]
    else:
        dims = [
            LONG_L_GRID,
            MID1_GRID,
            base_risk_dim_grid,
            ATR_STOP_MULT_GRID,
            MACRO_LOOKBACK_GRID,
        ]

    total = 1
    for d in dims:
        total *= len(d)

    def make_param(vals: Tuple) -> Params:
        (long_l, mid1, base_risk, atr_stop_mult, macro_lb) = vals
        return Params(
            long_L=float(long_l),
            mid1=float(mid1),
            base_risk_pct=float(base_risk),
            atr_stop_mult=float(atr_stop_mult),
            macro_lookback_bars=int(macro_lb),
            layer_weights_value=cfg.override_layer_weights if cfg.override_layer_weights is not None else (1.0, 1.4, 2.0),
            max_leverage_value=float(cfg.override_max_leverage) if cfg.override_max_leverage is not None else 2.0,
        )

    if total <= max_combos:
        return [make_param(vals) for vals in product(*dims)], total

    sampled_params: List[Params] = []
    seen: set[Params] = set()
    idxs = np.linspace(0, total - 1, max_combos, dtype=np.int64)
    for flat in idxs:
        rem = int(flat)
        picks = [0] * len(dims)
        for j in range(len(dims) - 1, -1, -1):
            dim_n = len(dims[j])
            picks[j] = rem % dim_n
            rem //= dim_n
        vals = tuple(dims[j][picks[j]] for j in range(len(dims)))
        p_obj = make_param(vals)
        if p_obj not in seen:
            seen.add(p_obj)
            sampled_params.append(p_obj)
    return sampled_params, total


def build_causal_clip_bounds(wdf: pd.DataFrame, cfg: RunConfig) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wdf)
    clip_low = np.full(n, np.nan, dtype=float)
    clip_high = np.full(n, np.nan, dtype=float)
    if cfg.box_source != "dynamic" or (not bool(cfg.clip_dynamic_to_manual)) or n == 0:
        return clip_low, clip_high

    lows = wdf["low"].to_numpy(dtype=float)
    highs = wdf["high"].to_numpy(dtype=float)
    active_mask = np.asarray(wdf.index >= cfg.start_utc, dtype=bool)
    hist_low = np.nan
    hist_high = np.nan

    for i in range(n):
        if not active_mask[i]:
            continue
        # Causal clip prior: bounds active on bar i only know bars through i-1.
        clip_low[i] = hist_low
        clip_high[i] = hist_high
        lo = lows[i]
        hi = highs[i]
        if np.isfinite(lo):
            hist_low = lo if (not np.isfinite(hist_low) or lo < hist_low) else hist_low
        if np.isfinite(hi):
            hist_high = hi if (not np.isfinite(hist_high) or hi > hist_high) else hist_high
    return clip_low, clip_high


def clip_arrays_to_causal_box(
    low: np.ndarray,
    high: np.ndarray,
    clip_low: np.ndarray,
    clip_high: np.ndarray,
    cfg: RunConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.box_source != "dynamic" or (not bool(cfg.clip_dynamic_to_manual)):
        return low, high
    if len(low) != len(clip_low) or len(high) != len(clip_high):
        raise ValueError("clip arrays length mismatch")
    low_clipped = np.where(np.isfinite(clip_low), np.maximum(low, clip_low), low)
    high_clipped = np.where(np.isfinite(clip_high), np.minimum(high, clip_high), high)
    invalid = ~(np.isfinite(low_clipped) & np.isfinite(high_clipped) & (high_clipped > low_clipped))
    if np.any(invalid):
        low_clipped = low_clipped.copy()
        high_clipped = high_clipped.copy()
        low_clipped[invalid] = np.nan
        high_clipped[invalid] = np.nan
    return low_clipped, high_clipped


def build_macro_box_arrays(
    wdf: pd.DataFrame,
    cfg: RunConfig,
    lookback_override: Optional[int] = None,
    clip_low_arr: Optional[np.ndarray] = None,
    clip_high_arr: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(wdf)
    if cfg.box_source != "dynamic":
        low = np.full(n, float(cfg.box_low), dtype=float)
        high = np.full(n, float(cfg.box_high), dtype=float)
        return low, high

    mode = str(cfg.macro_box_mode)
    if mode == "donchian":
        lb_src = int(lookback_override) if lookback_override is not None else int(cfg.macro_lookback_bars)
        lb = max(lb_src, 2)
        low = wdf["low"].rolling(window=lb, min_periods=lb).min().shift(1).to_numpy(dtype=float)
        high = wdf["high"].rolling(window=lb, min_periods=lb).max().shift(1).to_numpy(dtype=float)
    elif mode == "bb":
        bb_len = max(int(cfg.macro_bb_len), 2)
        mid = wdf["close"].rolling(window=bb_len, min_periods=bb_len).mean().shift(1)
        std = wdf["close"].rolling(window=bb_len, min_periods=bb_len).std(ddof=0).shift(1)
        low = (mid - float(cfg.macro_bb_std) * std).to_numpy(dtype=float)
        high = (mid + float(cfg.macro_bb_std) * std).to_numpy(dtype=float)
    else:
        raise ValueError(f"unsupported macro_box_mode: {mode}")

    if clip_low_arr is not None and clip_high_arr is not None:
        low, high = clip_arrays_to_causal_box(low, high, clip_low_arr, clip_high_arr, cfg)

    return low, high


@njit(cache=True)
def _core_loop_numba_fast(
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    x: np.ndarray,
    trade_valid: np.ndarray,
    regime_allowed: np.ndarray,
    box_low: np.ndarray,
    box_high: np.ndarray,
    long_L: float,
    mid1: float,
    mid2: float,
    base_risk_pct: float,
    atr_stop_mult: float,
    layer_step: float,
    hard_stop_pct: float,
    max_leverage: float,
    time_stop_bars: int,
    maker_ttl_bars: int,
    improve_mult: float,
    min_improve_bps: float,
    commission_pct: float,
    slippage_pct: float,
    allow_short: int,
    structural_cooldown_bars: int,
    enable_runner: int,
    runner_pct: float,
    runner_atr_mult: float,
    days_span: float,
) -> Tuple[float, float, float, float, int, int, int, float, float, int, int, int, int, int, float, int, int, int, float]:
    n = len(o)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0.0)

    comm = commission_pct / 100.0
    slp = slippage_pct / 100.0

    cash = 10000.0
    side = 0
    qty = 0.0
    avg_px = 0.0
    entry_idx = -1
    round_start_cash = 0.0
    tp1_done = 0
    layers_used = 0
    runner_active = 0
    runner_trail_px = np.nan

    long_block_until = 0
    short_block_until = 0
    long_need_center = 0
    short_need_center = 0

    # 0:none, 1:prepost, 2:active maker
    pstate = 0
    p_side = 0
    p_layer = -1
    p_qty = 0.0
    p_signal_close = 0.0
    p_improve_px = 0.0
    p_cash_before = 0.0
    p_post_idx = -1
    p_expire_idx = -1
    p_limit_px = 0.0
    p_taker_open = 0.0
    p_queue_trigger = -1

    pending_exit = 0
    exit_exec_idx = -1
    exit_reason = 0  # 1 HARD_STOP, 2 ATR_STOP, 3 TP2, 4 TP1, 5 TIME_STOP, 6 RUNNER_STOP, 7 TP2_PARTIAL
    exit_full = 1
    exit_qty = 0.0
    exit_runner_activate = 0
    exit_runner_trail_px = np.nan

    maker_orders = 0
    maker_fills = 0
    missed_entry = 0
    improve_sum_bps = 0.0
    improve_cnt = 0
    layer_add_count = 0
    signal_count_total = 0
    signal_count_long = 0
    signal_count_short = 0

    rounds_cnt = 0
    hard_stop_count = 0
    runner_stop_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    pnl_sum = 0.0
    layers_sum = 0.0
    hold_bars_sum = 0.0
    long_trade_count = 0
    short_trade_count = 0
    orders_total = 0

    eq_peak = 10000.0
    max_dd = 0.0
    max_margin_utilization = 0.0
    structural_cooldown_left = 0

    for i in range(n):
        bx_low = box_low[i]
        bx_high = box_high[i]
        bx_w = bx_high - bx_low
        box_ready = np.isfinite(bx_low) and np.isfinite(bx_high) and (bx_w > 0.0)
        structural_cooldown_now = structural_cooldown_left > 0

        if long_need_center == 1 and np.isfinite(x[i]) and (x[i] >= 0.45 and x[i] <= 0.55):
            long_need_center = 0
        if short_need_center == 1 and np.isfinite(x[i]) and (x[i] >= 0.45 and x[i] <= 0.55):
            short_need_center = 0

        if pending_exit == 1 and i == exit_exec_idx and side != 0:
            q_close = qty if exit_full == 1 else min(qty, exit_qty)
            if q_close > 0.0:
                if side > 0:
                    px = o[i] * (1.0 - slp)
                    notional = q_close * px
                    fee = notional * comm
                    cash += notional - fee
                    orders_total += 1
                else:
                    px = o[i] * (1.0 + slp)
                    notional = q_close * px
                    fee = notional * comm
                    cash -= notional + fee
                    orders_total += 1
                qty -= q_close
                if exit_reason == 7 and qty > 1e-12 and exit_runner_activate == 1:
                    runner_active = 1
                    runner_trail_px = exit_runner_trail_px
                    pstate = 0
                    tp1_done = 1
                if exit_full == 1 or qty <= 1e-12:
                    pnl = cash - round_start_cash
                    rounds_cnt += 1
                    pnl_sum += pnl
                    if pnl > 0.0:
                        gross_profit += pnl
                    else:
                        gross_loss += -pnl
                    if exit_reason == 1:
                        hard_stop_count += 1
                    elif exit_reason == 6:
                        runner_stop_count += 1
                    layers_sum += layers_used
                    hold_bars_sum += max(i - entry_idx, 0)
                    if side > 0:
                        long_trade_count += 1
                    else:
                        short_trade_count += 1

                    if exit_reason in (1, 2):
                        if side > 0:
                            long_block_until = i + 24
                            long_need_center = 1
                        else:
                            short_block_until = i + 24
                            short_need_center = 1
                    else:
                        if side > 0:
                            long_block_until = i + 1
                            long_need_center = 0
                        else:
                            short_block_until = i + 1
                            short_need_center = 0
                    side = 0
                    qty = 0.0
                    avg_px = 0.0
                    entry_idx = -1
                    round_start_cash = 0.0
                    layers_used = 0
                    tp1_done = 0
                    runner_active = 0
                    runner_trail_px = np.nan
                elif exit_reason == 4:
                    tp1_done = 1
            pending_exit = 0
            exit_runner_activate = 0
            exit_runner_trail_px = np.nan

        if pstate == 1 and i == p_post_idx:
            p_taker_open = o[i]
            if p_side > 0:
                raw_limit = p_signal_close - p_improve_px
                min_limit = o[i] * (1.0 - min_improve_bps / 10000.0)
                p_limit_px = raw_limit if raw_limit < min_limit else min_limit
            else:
                raw_limit = p_signal_close + p_improve_px
                min_limit = o[i] * (1.0 + min_improve_bps / 10000.0)
                p_limit_px = raw_limit if raw_limit > min_limit else min_limit
            if p_limit_px < 1e-9:
                p_limit_px = 1e-9
            pstate = 2
            p_queue_trigger = -1
            maker_orders += 1

        if pstate == 2 and i >= p_post_idx:
            if side != 0 and side != p_side:
                missed_entry += 1
                pstate = 0
            else:
                eps = 1e-6
                can_be_maker = abs(p_limit_px - p_taker_open) > 1e-12
                touched = 0
                if can_be_maker:
                    if p_side > 0 and l[i] < (p_limit_px - eps):
                        touched = 1
                    if p_side < 0 and h[i] > (p_limit_px + eps):
                        touched = 1
                if touched == 1 and p_queue_trigger < 0:
                    p_queue_trigger = i
                delay_ok = p_queue_trigger >= 0 and i >= p_queue_trigger
                if touched == 1 and delay_ok:
                    if p_side > 0:
                        fill_px = p_limit_px
                        notional = p_qty * fill_px
                        fee = notional * comm
                        cash -= notional + fee
                        orders_total += 1
                    else:
                        fill_px = p_limit_px
                        notional = p_qty * fill_px
                        fee = notional * comm
                        cash += notional - fee
                        orders_total += 1
                    maker_fills += 1
                    if p_side > 0:
                        improve_sum_bps += (p_taker_open - p_limit_px) / p_taker_open * 10000.0
                    else:
                        improve_sum_bps += (p_limit_px - p_taker_open) / p_taker_open * 10000.0
                    improve_cnt += 1

                    if side == 0:
                        side = p_side
                        qty = p_qty
                        avg_px = fill_px
                        entry_idx = i
                        round_start_cash = p_cash_before
                        layers_used = 1
                        tp1_done = 0
                    else:
                        new_qty = qty + p_qty
                        avg_px = (avg_px * qty + fill_px * p_qty) / new_qty
                        qty = new_qty
                        if p_layer + 1 > layers_used:
                            layers_used = p_layer + 1
                        layer_add_count += 1
                    pstate = 0
                elif i >= p_expire_idx:
                    missed_entry += 1
                    pstate = 0

        if side != 0 and pending_exit == 0 and box_ready:
            bars_held = i - entry_idx if entry_idx >= 0 else 0
            long_stop = bx_low - hard_stop_pct * bx_w
            short_stop = bx_high + hard_stop_pct * bx_w
            reason = 0
            full = 1
            q_part = qty
            runner_activate = 0
            runner_init_trail = np.nan
            if runner_active == 1:
                runner_hit = np.isfinite(runner_trail_px) and (
                    (side > 0 and l[i] <= runner_trail_px) or (side < 0 and h[i] >= runner_trail_px)
                )
                if runner_hit:
                    reason = 6
                elif np.isfinite(atr[i]):
                    if side > 0:
                        next_trail = c[i] - runner_atr_mult * atr[i]
                        if (not np.isfinite(runner_trail_px)) or next_trail > runner_trail_px:
                            runner_trail_px = next_trail
                    else:
                        next_trail = c[i] + runner_atr_mult * atr[i]
                        if (not np.isfinite(runner_trail_px)) or next_trail < runner_trail_px:
                            runner_trail_px = next_trail
            else:
                atr_stop_hit = (
                    np.isfinite(atr[i])
                    and (
                        (side > 0 and l[i] <= (avg_px - atr_stop_mult * atr[i]))
                        or (side < 0 and h[i] >= (avg_px + atr_stop_mult * atr[i]))
                    )
                )
                hard_stop_hit = (side > 0 and l[i] <= long_stop) or (side < 0 and h[i] >= short_stop)
                tp2_hit = np.isfinite(x[i]) and ((side > 0 and x[i] >= mid2) or (side < 0 and x[i] <= 1.0 - mid2))
                tp1_hit = np.isfinite(x[i]) and ((side > 0 and x[i] >= mid1) or (side < 0 and x[i] <= 1.0 - mid1))
                time_hit = bars_held >= time_stop_bars

                if atr_stop_hit:
                    reason = 2
                elif hard_stop_hit:
                    reason = 1
                elif tp2_hit:
                    if enable_runner == 1 and runner_pct > 0.0 and runner_pct < 1.0:
                        keep_qty = qty * runner_pct
                        q_part = qty - keep_qty
                        if keep_qty > 1e-12 and q_part > 1e-12 and np.isfinite(atr[i]):
                            reason = 7
                            full = 0
                            runner_activate = 1
                            if side > 0:
                                runner_init_trail = c[i] - runner_atr_mult * atr[i]
                            else:
                                runner_init_trail = c[i] + runner_atr_mult * atr[i]
                        else:
                            reason = 3
                            q_part = qty
                    else:
                        reason = 3
                elif tp1_done == 0 and tp1_hit:
                    reason = 4
                    full = 0
                    q_part = qty * 0.5
                elif time_hit:
                    reason = 5

            if reason != 0 and i + 1 < n:
                pending_exit = 1
                exit_exec_idx = i + 1
                exit_reason = reason
                exit_full = full
                exit_qty = q_part
                exit_runner_activate = runner_activate
                exit_runner_trail_px = runner_init_trail

        if (
            pending_exit == 0
            and pstate == 0
            and runner_active == 0
            and trade_valid[i]
            and regime_allowed[i]
            and np.isfinite(x[i])
            and box_ready
            and (not structural_cooldown_now)
            and i + 1 < n
        ):
            long_ready = (i >= long_block_until) and (long_need_center == 0)
            short_ready = (i >= short_block_until) and (short_need_center == 0)
            short_L = 1.0 - long_L
            long_l1 = long_L
            long_l2 = long_L - layer_step
            long_l3 = long_L - 2.0 * layer_step
            short_l1 = short_L
            short_l2 = short_L + layer_step
            short_l3 = short_L + 2.0 * layer_step

            cand_side = 0
            cand_layer = -1
            if side == 0:
                if long_ready and x[i] <= long_l1:
                    cand_side = 1
                    cand_layer = 0
                elif allow_short == 1 and short_ready and x[i] >= short_l1:
                    cand_side = -1
                    cand_layer = 0
            elif side > 0 and layers_used < 3:
                if (layers_used == 1 and x[i] <= long_l2) or (layers_used == 2 and x[i] <= long_l3):
                    cand_side = 1
                    cand_layer = layers_used
            elif allow_short == 1 and side < 0 and layers_used < 3:
                if (layers_used == 1 and x[i] >= short_l2) or (layers_used == 2 and x[i] >= short_l3):
                    cand_side = -1
                    cand_layer = layers_used

            if cand_side != 0:
                improve_px = improve_mult * abs(c[i] - o[i]) * 0.01
                signal_close = c[i]
                prelim_limit = signal_close - improve_px if cand_side > 0 else signal_close + improve_px
                if prelim_limit < 1e-9:
                    prelim_limit = 1e-9
                eq_now = cash + (qty * c[i] if side > 0 else (-qty * c[i] if side < 0 else 0.0))
                long_stop = bx_low - hard_stop_pct * bx_w
                short_stop = bx_high + hard_stop_pct * bx_w
                stop_px = long_stop if cand_side > 0 else short_stop
                per_unit_risk = abs(prelim_limit - stop_px)
                if per_unit_risk > 1e-12:
                    base_qty = eq_now * (base_risk_pct / 100.0) / per_unit_risk
                    lw = 1.0
                    if cand_layer == 1:
                        lw = 1.4
                    elif cand_layer == 2:
                        lw = 2.0
                    target_qty = base_qty * lw
                    curr_qty = qty if side == cand_side else 0.0
                    max_qty_allowed = max(eq_now * max_leverage / prelim_limit, 0.0)
                    add_cap = max(max_qty_allowed - curr_qty, 0.0)
                    qx = target_qty if target_qty < add_cap else add_cap
                    if qx > 1e-10:
                        signal_count_total += 1
                        if cand_side > 0:
                            signal_count_long += 1
                        else:
                            signal_count_short += 1
                        pstate = 1
                        p_side = cand_side
                        p_layer = cand_layer
                        p_qty = qx
                        p_signal_close = signal_close
                        p_improve_px = improve_px
                        p_cash_before = cash if side == 0 else round_start_cash
                        p_post_idx = i + 1
                        p_expire_idx = i + maker_ttl_bars

        eq = cash
        if side > 0:
            eq += qty * c[i]
        elif side < 0:
            eq -= qty * c[i]
        if eq > 1e-12 and max_leverage > 1e-12 and qty > 0.0 and np.isfinite(c[i]):
            margin_util = abs(qty * c[i]) / (eq * max_leverage)
            if margin_util > max_margin_utilization:
                max_margin_utilization = margin_util
        if eq > eq_peak:
            eq_peak = eq
        if eq_peak > 0.0:
            dd = (eq_peak - eq) / eq_peak * 100.0
            if dd > max_dd:
                max_dd = dd
        if box_ready and ((c[i] > bx_high) or (c[i] < bx_low)):
            structural_cooldown_left = structural_cooldown_bars
        elif structural_cooldown_left > 0:
            structural_cooldown_left -= 1

    if side != 0 and n > 0:
        q_close = qty
        if side > 0:
            px = c[n - 1] * (1.0 - slp)
            notional = q_close * px
            fee = notional * comm
            cash += notional - fee
            orders_total += 1
        else:
            px = c[n - 1] * (1.0 + slp)
            notional = q_close * px
            fee = notional * comm
            cash -= notional + fee
            orders_total += 1
        pnl = cash - round_start_cash
        rounds_cnt += 1
        pnl_sum += pnl
        if pnl > 0.0:
            gross_profit += pnl
        else:
            gross_loss += -pnl
        layers_sum += layers_used
        hold_bars_sum += max(n - 1 - entry_idx, 0)
        if side > 0:
            long_trade_count += 1
        else:
            short_trade_count += 1

    ret_pct = (cash / 10000.0 - 1.0) * 100.0
    pf = gross_profit / gross_loss if gross_loss > 1e-12 else (999.0 if gross_profit > 0.0 else 0.0)
    expectancy = pnl_sum / rounds_cnt if rounds_cnt > 0 else 0.0
    hard_stop_ratio = hard_stop_count / rounds_cnt if rounds_cnt > 0 else 0.0
    avg_layers = layers_sum / rounds_cnt if rounds_cnt > 0 else 0.0
    avg_hold = hold_bars_sum / rounds_cnt if rounds_cnt > 0 else 0.0
    entry_fill_rate = maker_fills / maker_orders if maker_orders > 0 else 0.0
    avg_improve = improve_sum_bps / improve_cnt if improve_cnt > 0 else 0.0
    daily_orders = orders_total / days_span if days_span > 1e-12 else 0.0
    return (
        ret_pct,
        max_dd,
        pf,
        expectancy,
        rounds_cnt,
        hard_stop_count,
        runner_stop_count,
        avg_layers,
        avg_hold,
        long_trade_count,
        short_trade_count,
        orders_total,
        maker_orders,
        maker_fills,
        daily_orders,
        missed_entry,
        signal_count_total,
        signal_count_long + signal_count_short,
        max_margin_utilization,
    )


def _can_use_numba_fast_path(cfg: RunConfig) -> bool:
    return bool(
        NUMBA_AVAILABLE
        and cfg.box_source == "dynamic"
        and cfg.dyn_cfg.mode == "none"
        and cfg.start_gate_mode == "off"
        and cfg.invalidate_mode == "off"
        and cfg.perf_stop_mode == "off"
        and cfg.circuit_breaker_mode == "off"
        and (not cfg.degrade_on_trend)
        and cfg.short_enable_rule == "none"
    )


def run_backtest_fast_stats(df: pd.DataFrame, p: Params, cfg: RunConfig) -> Dict[str, float]:
    slice_start = cfg.history_start_utc if cfg.history_start_utc is not None else cfg.start_utc
    wdf = df.loc[(df.index >= slice_start) & (df.index <= cfg.end_utc)].copy()
    active_wdf = wdf.loc[wdf.index >= cfg.start_utc]
    if active_wdf.empty:
        raise ValueError("No bars inside active trade window")

    o = wdf["open"].to_numpy(dtype=float)
    h = wdf["high"].to_numpy(dtype=float)
    l = wdf["low"].to_numpy(dtype=float)
    c = wdf["close"].to_numpy(dtype=float)
    adx = wdf["adx14"].to_numpy(dtype=float) if "adx14" in wdf.columns else np.full(len(wdf), np.nan, dtype=float)
    bb_width = wdf["bb_width20"].to_numpy(dtype=float) if "bb_width20" in wdf.columns else np.full(len(wdf), np.nan, dtype=float)
    chop = wdf["chop72"].to_numpy(dtype=float) if "chop72" in wdf.columns else np.full(len(wdf), np.nan, dtype=float)
    active_mask = np.asarray(wdf.index >= cfg.start_utc, dtype=bool)
    clip_low_arr, clip_high_arr = build_causal_clip_bounds(wdf, cfg)

    trade_mode = cfg.trade_box_mode_default
    trade_lb = max(int(cfg.trade_box_lookback_default), 2)
    if trade_mode == "rolling_hilo":
        trade_low_raw = wdf["low"].rolling(window=trade_lb, min_periods=trade_lb).min().shift(1).to_numpy(dtype=float)
        trade_high_raw = wdf["high"].rolling(window=trade_lb, min_periods=trade_lb).max().shift(1).to_numpy(dtype=float)
    elif trade_mode == "rolling_quantile":
        trade_low_raw = (
            wdf["close"]
            .rolling(window=trade_lb, min_periods=trade_lb)
            .quantile(cfg.trade_box_q_low_default)
            .shift(1)
            .to_numpy(dtype=float)
        )
        trade_high_raw = (
            wdf["close"]
            .rolling(window=trade_lb, min_periods=trade_lb)
            .quantile(cfg.trade_box_q_high_default)
            .shift(1)
            .to_numpy(dtype=float)
        )
    else:
        ema_len = max(int(cfg.trade_box_ema_len_default), 2)
        ema = wdf["close"].ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().shift(1)
        atr_shift = wdf["atr14"].shift(1)
        trade_low_raw = (ema - cfg.trade_box_atr_mult_default * atr_shift).to_numpy(dtype=float)
        trade_high_raw = (ema + cfg.trade_box_atr_mult_default * atr_shift).to_numpy(dtype=float)
    trade_low_raw, trade_high_raw = clip_arrays_to_causal_box(
        trade_low_raw, trade_high_raw, clip_low_arr, clip_high_arr, cfg
    )

    box_low, box_high = build_macro_box_arrays(
        wdf,
        cfg,
        lookback_override=p.macro_lookback_bars,
        clip_low_arr=clip_low_arr,
        clip_high_arr=clip_high_arr,
    )
    x = np.full(len(wdf), np.nan, dtype=float)
    trade_valid = np.zeros(len(wdf), dtype=np.bool_)
    for i in range(len(wdf)):
        if not (np.isfinite(box_low[i]) and np.isfinite(box_high[i]) and box_high[i] > box_low[i]):
            continue
        bw = box_high[i] - box_low[i]
        buf = cfg.dyn_cfg.break_buffer_pct * bw if cfg.dyn_cfg.break_buffer_mode == "pct" else cfg.dyn_cfg.break_buffer_atr_mult * float(wdf["atr14"].iloc[i])
        risk_low = box_low[i] - buf
        risk_high = box_high[i] + buf
        tl = trade_low_raw[i]
        th = trade_high_raw[i]
        if not (np.isfinite(tl) and np.isfinite(th)):
            continue
        tl = max(tl, risk_low)
        th = min(th, risk_high)
        if th <= tl:
            continue
        trade_valid[i] = True
        x[i] = (c[i] - tl) / (th - tl)
    trade_valid = trade_valid & active_mask
    x = np.where(active_mask, x, np.nan)
    if cfg.regime_gate_mode == "on":
        regime_allowed = (
            active_mask
            & np.isfinite(adx)
            & (adx < float(cfg.regime_gate_adx_thresh))
            & np.isfinite(bb_width)
            & (bb_width > float(cfg.regime_gate_bbwidth_min))
        )
        if "chop72" in wdf.columns:
            regime_allowed = regime_allowed & np.isfinite(chop) & (chop > float(cfg.regime_gate_chop_thresh))
    else:
        regime_allowed = active_mask.copy()
    regime_start_gate_pass_rate = float(regime_allowed[active_mask].mean()) if active_mask.any() else 0.0

    days_span = max((active_wdf.index[-1] - active_wdf.index[0]).total_seconds() / 86400.0, 1e-9)
    out = _core_loop_numba_fast(
        o=o,
        h=h,
        l=l,
        c=c,
        atr=wdf["atr14"].to_numpy(dtype=float),
        x=x,
        trade_valid=trade_valid,
        regime_allowed=regime_allowed,
        box_low=box_low,
        box_high=box_high,
        long_L=float(p.long_L),
        mid1=float(p.mid1),
        mid2=float(p.mid2),
        base_risk_pct=float(p.base_risk_pct),
        atr_stop_mult=float(p.atr_stop_mult),
        layer_step=float(p.layer_step),
        hard_stop_pct=float(p.hard_stop_pct),
        max_leverage=float(p.max_leverage),
        time_stop_bars=int(cfg.local_time_stop_bars),
        maker_ttl_bars=int(p.maker_ttl_bars),
        improve_mult=float(p.improve_mult),
        min_improve_bps=float(p.min_improve_bps),
        commission_pct=float(cfg.commission_pct),
        slippage_pct=float(cfg.slippage_pct),
        allow_short=1 if cfg.side_mode == "both" else 0,
        structural_cooldown_bars=max(int(cfg.structural_cooldown_bars), 0),
        enable_runner=1 if cfg.enable_runner else 0,
        runner_pct=float(cfg.runner_pct),
        runner_atr_mult=float(cfg.runner_atr_mult),
        days_span=float(days_span),
    )
    (
        ret_pct,
        max_dd,
        pf,
        expectancy,
        round_trips,
        hard_stop_count,
        runner_stop_count,
        avg_layers,
        avg_hold,
        long_trade_count,
        short_trade_count,
        orders_total,
        maker_orders,
        maker_fills,
        daily_orders,
        missed_entry,
        signal_count_total,
        _signal_count_join,
        max_margin_utilization,
    ) = out
    hard_stop_ratio = float(hard_stop_count / round_trips) if round_trips > 0 else 0.0
    entry_fill_rate = float(maker_fills / maker_orders) if maker_orders > 0 else 0.0
    initial_equity = max(float(cfg.initial_equity), 1e-12)
    ret_pct = float(((1.0 + (ret_pct / 100.0)) * (10000.0 / initial_equity) - 1.0) * 100.0)
    objective = float(ret_pct - 0.6 * max_dd - 0.2 * ((missed_entry / maker_orders) if maker_orders > 0 else 0.0) * 100.0)
    return {
        "return_pct": float(ret_pct),
        "max_dd_pct": float(max_dd),
        "profit_factor": float(pf),
        "pf_reliable": bool(pf < 900.0),
        "expectancy_after_cost": float(expectancy),
        "round_trips": int(round_trips),
        "hard_stop_count": int(hard_stop_count),
        "runner_stop_count": int(runner_stop_count),
        "hard_stop_ratio": float(hard_stop_ratio),
        "avg_layers": float(avg_layers),
        "avg_hold_bars": float(avg_hold),
        "long_trade_count": int(long_trade_count),
        "short_trade_count": int(short_trade_count),
        "orders_total": int(orders_total),
        "daily_orders": float(daily_orders),
        "entry_fill_rate": float(entry_fill_rate),
        "missed_entry": int(missed_entry),
        "signal_count_total": int(signal_count_total),
        "objective": float(objective),
        "filled_entries": int(maker_fills),
        "filled_exits": int(round_trips),
        "trade_box_invalid_ratio": float((~trade_valid).mean() if len(trade_valid) > 0 else 0.0),
        "regime_start_gate_pass_rate": float(regime_start_gate_pass_rate),
        "risk_expand_count": 0,
        "box_invalid_count": 0,
        "selected_risk_high": np.nan,
        "probe_tie_reason": "PROBE_SKIPPED_DYNAMIC_BOX",
        "max_margin_utilization": float(max_margin_utilization),
    }


def run_backtest(df: pd.DataFrame, p: Params, cfg: RunConfig) -> Dict[str, object]:
    slice_start = cfg.history_start_utc if cfg.history_start_utc is not None else cfg.start_utc
    wdf = df.loc[(df.index >= slice_start) & (df.index <= cfg.end_utc)].copy()
    active_wdf = wdf.loc[wdf.index >= cfg.start_utc]
    if active_wdf.empty:
        raise ValueError("No bars inside active trade window")

    times = list(wdf.index)
    active_time_mask = np.array([ts >= cfg.start_utc for ts in times], dtype=bool)
    active_bar_count = int(active_time_mask.sum())
    o = wdf["open"].to_numpy(dtype=float)
    h = wdf["high"].to_numpy(dtype=float)
    l = wdf["low"].to_numpy(dtype=float)
    c = wdf["close"].to_numpy(dtype=float)
    atr = wdf["atr14"].to_numpy(dtype=float)
    n = len(wdf)
    ma_slow_arr = wdf["close"].ewm(span=20, adjust=False, min_periods=20).mean().shift(1).to_numpy(dtype=float)
    adx_arr = wdf["adx14"].to_numpy(dtype=float) if "adx14" in wdf.columns else np.full(n, np.nan, dtype=float)
    ema_slope_abs_arr = (
        wdf["ema50_slope_abs"].to_numpy(dtype=float) if "ema50_slope_abs" in wdf.columns else np.full(n, np.nan, dtype=float)
    )
    chop_arr = wdf["chop72"].to_numpy(dtype=float) if "chop72" in wdf.columns else np.full(n, np.nan, dtype=float)
    ema50_slope20_abs_arr = (
        wdf["ema50_slope20_abs"].to_numpy(dtype=float)
        if "ema50_slope20_abs" in wdf.columns
        else np.full(n, np.nan, dtype=float)
    )
    bb_upper_arr = wdf["bb_upper20"].to_numpy(dtype=float) if "bb_upper20" in wdf.columns else np.full(n, np.nan, dtype=float)
    bb_lower_arr = wdf["bb_lower20"].to_numpy(dtype=float) if "bb_lower20" in wdf.columns else np.full(n, np.nan, dtype=float)
    bb_width_arr = wdf["bb_width20"].to_numpy(dtype=float) if "bb_width20" in wdf.columns else np.full(n, np.nan, dtype=float)
    clip_low_arr, clip_high_arr = build_causal_clip_bounds(wdf, cfg)
    if "bb_width20" in wdf.columns:
        bb_width_q30_arr = (
            wdf["bb_width20"]
            .rolling(window=120, min_periods=120)
            .quantile(float(cfg.regime_gate_bbwidth_q_thresh))
            .to_numpy(dtype=float)
        )
        bb_width_q50_arr = (
            wdf["bb_width20"]
            .rolling(window=120, min_periods=120)
            .quantile(float(cfg.circuit_break_bbwidth_q_thresh))
            .to_numpy(dtype=float)
        )
    else:
        bb_width_q30_arr = np.full(n, np.nan, dtype=float)
        bb_width_q50_arr = np.full(n, np.nan, dtype=float)

    trade_mode = cfg.trade_box_mode_default
    trade_lb = max(int(cfg.trade_box_lookback_default), 2)
    if trade_mode == "rolling_hilo":
        trade_low_raw_arr = wdf["low"].rolling(window=trade_lb, min_periods=trade_lb).min().shift(1).to_numpy(dtype=float)
        trade_high_raw_arr = wdf["high"].rolling(window=trade_lb, min_periods=trade_lb).max().shift(1).to_numpy(dtype=float)
    elif trade_mode == "rolling_quantile":
        trade_low_raw_arr = (
            wdf["close"]
            .rolling(window=trade_lb, min_periods=trade_lb)
            .quantile(cfg.trade_box_q_low_default)
            .shift(1)
            .to_numpy(dtype=float)
        )
        trade_high_raw_arr = (
            wdf["close"]
            .rolling(window=trade_lb, min_periods=trade_lb)
            .quantile(cfg.trade_box_q_high_default)
            .shift(1)
            .to_numpy(dtype=float)
        )
    elif trade_mode == "ema_atr":
        ema_len = max(int(cfg.trade_box_ema_len_default), 2)
        ema = wdf["close"].ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().shift(1)
        atr_shift = wdf["atr14"].shift(1)
        trade_low_raw_arr = (ema - cfg.trade_box_atr_mult_default * atr_shift).to_numpy(dtype=float)
        trade_high_raw_arr = (ema + cfg.trade_box_atr_mult_default * atr_shift).to_numpy(dtype=float)
    else:
        trade_low_raw_arr = np.full(n, np.nan, dtype=float)
        trade_high_raw_arr = np.full(n, np.nan, dtype=float)
    trade_low_raw_arr, trade_high_raw_arr = clip_arrays_to_causal_box(
        trade_low_raw_arr, trade_high_raw_arr, clip_low_arr, clip_high_arr, cfg
    )
    macro_low_arr, macro_high_arr = build_macro_box_arrays(
        wdf,
        cfg,
        lookback_override=p.macro_lookback_bars,
        clip_low_arr=clip_low_arr,
        clip_high_arr=clip_high_arr,
    )
    macro_dynamic = bool(cfg.box_source == "dynamic")

    comm = cfg.commission_pct / 100.0
    slp = cfg.slippage_pct / 100.0

    initial_equity = float(cfg.initial_equity)
    cash = initial_equity
    side = 0
    qty = 0.0
    avg_px = 0.0
    entry_idx = -1
    round_start_cash = 0.0
    tp1_done = False
    layers_used = 0
    runner_active = False
    runner_trail_px = np.nan

    pending_signal = None
    pending_entry = None
    pending_fallback = None
    pending_exit = None

    long_block_until = 0
    short_block_until = 0
    long_need_center = False
    short_need_center = False

    box_low_rt = float(cfg.box_low)
    box_high_rt = float(cfg.box_high)
    if macro_dynamic:
        box_low_rt = np.nan
        box_high_rt = np.nan
        for bi in range(n):
            if np.isfinite(macro_low_arr[bi]) and np.isfinite(macro_high_arr[bi]) and (macro_high_arr[bi] > macro_low_arr[bi]):
                box_low_rt = float(macro_low_arr[bi])
                box_high_rt = float(macro_high_arr[bi])
                break
    box_active = True
    pending_box_update = None
    break_dn = 0
    break_up = 0
    expand_count = 0
    freeze_until_i = -1
    box_invalid_count = 0
    box_inactive_time = None
    first_box_low_rt = box_low_rt
    first_box_high_rt = box_high_rt
    expand_apply_indices: List[int] = []
    hard_stop_after_expand6_count = 0

    trades: List[Dict[str, object]] = []
    rounds: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []

    maker_orders = 0
    maker_fills = 0
    missed_entry = 0
    improve_sum_bps = 0.0
    improve_cnt = 0
    layer_add_count = 0
    taker_fallback_count = 0
    maker_fee_total = 0.0
    taker_fee_total = 0.0
    degrade_mode_bars = 0
    degrade_level1_bars = 0
    degrade_level2_bars = 0
    layer_disabled_count_due_to_degrade = 0
    trade_box_invalid_count = 0
    close_outside_count = 0
    closed_round_count = 0
    hard_stop_count_first_k = 0
    early_stop_due_to_bad_probe = False
    early_stop_reason = ""
    early_stop_time = None
    signal_count_total = 0
    signal_count_long = 0
    signal_count_short = 0
    short_enabled_true_count = 0
    short_enabled_total_count = 0
    short_reject_events_idx: List[int] = []
    short_lookback_bars = max(int(cfg.short_enable_lookback_days) * 6, 1)
    start_allowed_next = True
    regime_start_allowed_next = True
    start_gate_pass_count = 0
    start_gate_eval_count = 0
    regime_start_gate_pass_count = 0
    regime_start_gate_eval_count = 0
    edge_pending_up: List[Dict[str, float]] = []
    edge_pending_dn: List[Dict[str, float]] = []
    edge_confirm_up: List[int] = []
    edge_confirm_dn: List[int] = []
    gate_lb = max(int(cfg.gate_edge_reject_lookback_bars), 1)
    cb_disable_until_i = -1
    cb_trigger_count = 0
    cb_disable_bars = 0
    cb_outside_up_streak = 0
    cb_outside_dn_streak = 0
    structural_cooldown_left = 0
    structural_break_count = 0
    structural_cooldown_active_bars = 0

    invalidate_count_engine = 0
    invalid_first_ts = None
    invalidate_start_i = 10**9
    invalidate_until_i = -1
    inv_break_dn_engine = 0
    inv_break_up_engine = 0

    perf_stop_count = 0
    perf_start_i = 10**9
    perf_until_i = -1
    perf_recent_rounds: List[Dict[str, object]] = []
    invalidate_active_bars = 0
    perf_active_bars = 0

    def is_invalidate_active(bar_i: int) -> bool:
        return bool(cfg.invalidate_mode == "on" and bar_i >= invalidate_start_i and bar_i < invalidate_until_i)

    def is_perf_active(bar_i: int) -> bool:
        return bool(cfg.perf_stop_mode == "on" and bar_i >= perf_start_i and bar_i < perf_until_i)

    def is_cb_active(bar_i: int) -> bool:
        return bool(cfg.circuit_breaker_mode == "on" and bar_i < cb_disable_until_i)

    def mark_equity(px: float) -> float:
        if side > 0:
            return cash + qty * px
        if side < 0:
            return cash - qty * px
        return cash

    def margin_utilization(px: float) -> float:
        eq_now = mark_equity(px)
        if side == 0 or qty <= 0.0 or (not np.isfinite(px)) or eq_now <= 1e-12 or p.max_leverage <= 1e-12:
            return 0.0
        return float(abs(qty * px) / (eq_now * p.max_leverage))

    def apply_fill_price(raw_px: float, action: str, liquidity: str) -> float:
        if liquidity == "maker":
            return raw_px
        if action == "buy":
            return raw_px * (1.0 + slp)
        return raw_px * (1.0 - slp)

    def execute(action: str, q: float, raw_px: float, i: int, reason: str, tag: str, liquidity: str) -> float:
        nonlocal cash, maker_fee_total, taker_fee_total
        if q <= 0.0:
            return np.nan
        px = apply_fill_price(raw_px, action, liquidity)
        notional = q * px
        fee = notional * comm
        if liquidity == "maker":
            maker_fee_total += fee
        else:
            taker_fee_total += fee
        if action == "buy":
            cash -= notional + fee
        else:
            cash += notional - fee
        trades.append(
            {
                "time": times[i],
                "idx": i,
                "action": action,
                "qty": q,
                "raw_price": raw_px,
                "exec_price": px,
                "liquidity": liquidity,
                "commission": fee,
                "reason": reason,
                "tag": tag,
                "cash_after": cash,
            }
        )
        return px

    def finalize_round(i: int, exit_reason: str, cooldown_stop_bars_used: Optional[int] = None):
        nonlocal side, qty, avg_px, entry_idx, round_start_cash, tp1_done, layers_used
        nonlocal runner_active, runner_trail_px
        nonlocal long_block_until, short_block_until, long_need_center, short_need_center
        nonlocal closed_round_count, hard_stop_count_first_k
        nonlocal early_stop_due_to_bad_probe, early_stop_reason, early_stop_time
        nonlocal pending_signal, pending_entry, pending_fallback
        nonlocal perf_recent_rounds
        if side == 0:
            return
        pnl_round = cash - round_start_cash
        rounds.append(
            {
                "entry_time": times[entry_idx] if entry_idx >= 0 else pd.NaT,
                "exit_time": times[i],
                "side": "long" if side > 0 else "short",
                "exit_reason": exit_reason,
                "pnl": pnl_round,
                "hold_bars": max(i - entry_idx, 0),
                "layers_used": layers_used,
            }
        )
        perf_recent_rounds.append({"pnl": pnl_round, "hard_stop": exit_reason == "HARD_STOP"})
        if len(perf_recent_rounds) > max(int(cfg.perf_window_trades), 1):
            perf_recent_rounds = perf_recent_rounds[-max(int(cfg.perf_window_trades), 1):]
        if exit_reason in {"TP1", "TP2", "TIME_STOP", "RUNNER_STOP"}:
            if side > 0:
                long_block_until = i + p.cooldown_rearm_bars
                long_need_center = False
            else:
                short_block_until = i + p.cooldown_rearm_bars
                short_need_center = False
        elif exit_reason in {"HARD_STOP", "ATR_STOP", "BOX_INVALID", "INVALID", "PERF_STOP", "CIRCUIT_BREAK", "OUT_OF_BAND"}:
            stop_cd = int(cooldown_stop_bars_used) if cooldown_stop_bars_used is not None else int(p.cooldown_stop_bars)
            if side > 0:
                long_block_until = i + stop_cd
                long_need_center = True
            else:
                short_block_until = i + stop_cd
                short_need_center = True
        closed_round_count += 1
        if cfg.enable_early_stop and (closed_round_count <= cfg.early_stop_first_k_trades) and exit_reason == "HARD_STOP":
            hard_stop_count_first_k += 1
            if (not early_stop_due_to_bad_probe) and (hard_stop_count_first_k >= cfg.early_stop_hard_stop_threshold):
                early_stop_due_to_bad_probe = True
                early_stop_reason = "EARLY_STOP_DUE_TO_BAD_PROBE:HARD_STOP_CLUSTER"
                early_stop_time = times[i]
                pending_signal = None
                pending_entry = None
                pending_fallback = None
        side = 0
        qty = 0.0
        avg_px = 0.0
        entry_idx = -1
        round_start_cash = 0.0
        tp1_done = False
        layers_used = 0
        runner_active = False
        runner_trail_px = np.nan

    def close_full(i: int, raw_px: float, reason: str, cooldown_stop_bars_used: Optional[int] = None):
        if side == 0 or qty <= 0.0:
            return
        if side > 0:
            execute("sell", qty, raw_px, i, reason, "EXIT", "taker")
        else:
            execute("buy", qty, raw_px, i, reason, "EXIT", "taker")
        finalize_round(i, reason, cooldown_stop_bars_used)

    def close_partial(i: int, raw_px: float, reason: str, q_close: float, activate_runner: bool = False, runner_init_trail: float = np.nan):
        nonlocal qty, tp1_done, runner_active, runner_trail_px
        nonlocal pending_signal, pending_entry, pending_fallback
        if side == 0 or qty <= 0.0:
            return
        q = min(qty, q_close)
        if q <= 0.0:
            return
        if side > 0:
            execute("sell", q, raw_px, i, reason, "EXIT_PARTIAL", "taker")
        else:
            execute("buy", q, raw_px, i, reason, "EXIT_PARTIAL", "taker")
        qty -= q
        if reason in {"TP1", "TP2_PARTIAL"}:
            tp1_done = True
        if activate_runner and qty > 1e-12:
            runner_active = True
            runner_trail_px = float(runner_init_trail) if np.isfinite(runner_init_trail) else float(raw_px)
            pending_signal = None
            pending_entry = None
            pending_fallback = None
        if qty <= 1e-12:
            finalize_round(i, reason)

    allow_short = cfg.side_mode == "both"
    for i in range(n):
        bar_live = bool(active_time_mask[i])
        if macro_dynamic:
            if np.isfinite(macro_low_arr[i]) and np.isfinite(macro_high_arr[i]) and (macro_high_arr[i] > macro_low_arr[i]):
                box_low_rt = float(macro_low_arr[i])
                box_high_rt = float(macro_high_arr[i])
                if not np.isfinite(first_box_low_rt):
                    first_box_low_rt = box_low_rt
                    first_box_high_rt = box_high_rt
        else:
            if pending_box_update is not None and int(pending_box_update["apply_i"]) == i:
                if "low" in pending_box_update and "high" in pending_box_update:
                    old_low = box_low_rt
                    old_high = box_high_rt
                    box_low_rt = float(pending_box_update["low"])
                    box_high_rt = float(pending_box_update["high"])
                    if (not np.isclose(old_low, box_low_rt)) or (not np.isclose(old_high, box_high_rt)):
                        if bar_live:
                            expand_count += 1
                        expand_apply_indices.append(i)
                        freeze_until_i = max(freeze_until_i, i + cfg.dyn_cfg.freeze_after_expand_bars)
                        # Freeze means no new open/add. Drop all pending entry intents immediately.
                        pending_signal = None
                        pending_entry = None
                        pending_fallback = None
                if "active" in pending_box_update:
                    next_active = bool(pending_box_update["active"])
                    if box_active and (not next_active):
                        if bar_live:
                            box_invalid_count += 1
                        if box_inactive_time is None:
                            box_inactive_time = times[i]
                    box_active = next_active
                    if not box_active:
                        pending_signal = None
                        pending_entry = None
                        pending_fallback = None
                        if cfg.enable_early_stop and (i < cfg.early_stop_first_m_bars) and (not early_stop_due_to_bad_probe):
                            early_stop_due_to_bad_probe = True
                            early_stop_reason = "EARLY_STOP_DUE_TO_BAD_PROBE:BOX_INVALID_EARLY"
                            early_stop_time = times[i]
                pending_box_update = None

        start_allowed_now = bool(start_allowed_next) if cfg.start_gate_mode == "on" else True
        regime_start_allowed_now = bool(regime_start_allowed_next) if cfg.regime_gate_mode == "on" else True
        cb_active_now = is_cb_active(i)
        structural_cooldown_now = bool(structural_cooldown_left > 0)
        if cb_active_now and bar_live:
            cb_disable_bars += 1
        if structural_cooldown_now and bar_live:
            structural_cooldown_active_bars += 1
        invalidate_active_now = is_invalidate_active(i)
        perf_stop_active_now = is_perf_active(i)
        if invalidate_active_now and bar_live:
            invalidate_active_bars += 1
        if perf_stop_active_now and bar_live:
            perf_active_bars += 1

        box_width_rt = box_high_rt - box_low_rt
        risk_box_ready = bool(np.isfinite(box_low_rt) and np.isfinite(box_high_rt) and (box_width_rt > 0.0))
        if not risk_box_ready:
            box_width_rt = np.nan

        if cfg.dyn_cfg.break_buffer_mode == "pct" and np.isfinite(box_width_rt):
            break_buf = cfg.dyn_cfg.break_buffer_pct * box_width_rt
        else:
            break_buf = cfg.dyn_cfg.break_buffer_atr_mult * atr[i]
        if macro_dynamic and np.isfinite(macro_low_arr[i]) and np.isfinite(macro_high_arr[i]) and (macro_high_arr[i] > macro_low_arr[i]):
            structural_low = float(macro_low_arr[i])
            structural_high = float(macro_high_arr[i])
        else:
            structural_low = float(box_low_rt)
            structural_high = float(box_high_rt)
        structural_box_ready = bool(np.isfinite(structural_low) and np.isfinite(structural_high) and (structural_high > structural_low))
        structural_break_now = bool(structural_box_ready and ((c[i] > structural_high) or (c[i] < structural_low)))
        risk_low_entry = box_low_rt - break_buf if risk_box_ready else np.nan
        risk_high_entry = box_high_rt + break_buf if risk_box_ready else np.nan

        if trade_mode == "risk" or (not risk_box_ready):
            trade_low_raw = box_low_rt
            trade_high_raw = box_high_rt
        else:
            trade_low_raw = trade_low_raw_arr[i]
            trade_high_raw = trade_high_raw_arr[i]
        if np.isfinite(trade_low_raw) and np.isfinite(trade_high_raw) and np.isfinite(risk_low_entry) and np.isfinite(risk_high_entry):
            trade_low = max(float(trade_low_raw), float(risk_low_entry))
            trade_high = min(float(trade_high_raw), float(risk_high_entry))
            trade_box_width = trade_high - trade_low
            trade_box_invalid = trade_box_width <= 1e-12
        else:
            trade_low = np.nan
            trade_high = np.nan
            trade_box_width = np.nan
            trade_box_invalid = True
        if trade_box_invalid and bar_live:
            trade_box_invalid_count += 1
            x = np.nan
        else:
            x = (c[i] - trade_low) / trade_box_width

        short_allowed_rule = True
        if allow_short:
            if cfg.short_enable_rule == "simple":
                min_idx = i - short_lookback_bars + 1
                while short_reject_events_idx and short_reject_events_idx[0] < min_idx:
                    short_reject_events_idx.pop(0)
                if (not trade_box_invalid) and np.isfinite(trade_box_width):
                    touch_px = trade_high - float(cfg.short_enable_touch_band) * trade_box_width
                    reject_px = trade_high - float(cfg.short_enable_reject_close_gap) * trade_box_width
                    touched_upper = h[i] >= touch_px
                    rejected_down = (c[i] <= reject_px) and (c[i] < o[i])
                if touched_upper and rejected_down:
                    short_reject_events_idx.append(i)
                short_allowed_rule = len(short_reject_events_idx) >= int(cfg.short_enable_min_rejects)
            if bar_live:
                short_enabled_total_count += 1
            if bar_live and short_allowed_rule:
                short_enabled_true_count += 1

        recent_expand_event = bool(expand_apply_indices and ((i - expand_apply_indices[-1]) <= int(cfg.degrade_risk_lookback_bars)))
        risk_event = bool((c[i] < risk_low_entry) or (c[i] > risk_high_entry) or (break_dn > 0) or (break_up > 0) or recent_expand_event)
        trend_event = False
        if cfg.degrade_on_trend and (not trade_box_invalid) and i >= 1:
            ma_now = ma_slow_arr[i]
            ma_prev = ma_slow_arr[i - 1]
            if np.isfinite(ma_now) and np.isfinite(ma_prev):
                slope_norm = abs(ma_now - ma_prev) / max(trade_box_width, 1e-12)
                atr_expand_ratio = atr[i] / max(trade_box_width, 1e-12)
                trend_event = bool((slope_norm >= cfg.trend_slope_thresh) or (atr_expand_ratio >= cfg.atr_expand_thresh))
        degrade_level = 0
        if cfg.degrade_on_trend and risk_event:
            degrade_level = 2 if trend_event else 1
        degrade_active = degrade_level > 0
        if degrade_active and bar_live:
            degrade_mode_bars += 1
            if degrade_level == 1:
                degrade_level1_bars += 1
            elif degrade_level == 2:
                degrade_level2_bars += 1

        long_stop_rt = box_low_rt - p.hard_stop_pct * box_width_rt
        short_stop_rt = box_high_rt + p.hard_stop_pct * box_width_rt

        if long_need_center and np.isfinite(x) and (0.45 <= x <= 0.55):
            long_need_center = False
        if short_need_center and np.isfinite(x) and (0.45 <= x <= 0.55):
            short_need_center = False

        if pending_exit is not None and pending_exit["execute_idx"] == i:
            assert i == pending_exit["signal_idx"] + 1, "Exit execution must be t+1"
            if pending_exit["full"]:
                close_full(i, o[i], pending_exit["reason"], pending_exit.get("cooldown_stop_bars_used"))
            else:
                close_partial(
                    i,
                    o[i],
                    pending_exit["reason"],
                    pending_exit["qty"],
                    activate_runner=bool(pending_exit.get("activate_runner", False)),
                    runner_init_trail=float(pending_exit.get("runner_init_trail", np.nan)),
                )
            pending_exit = None

        if pending_fallback is not None and pending_fallback["execute_idx"] == i:
            assert i == pending_fallback["signal_idx"] + 1, "Fallback entry execution must be t+1"
            if (
                box_active
                and (not runner_active)
                and i >= freeze_until_i
                and (not early_stop_due_to_bad_probe)
                and (not invalidate_active_now)
                and (not perf_stop_active_now)
                and start_allowed_now
                and regime_start_allowed_now
                and (not cb_active_now)
                and (not structural_cooldown_now)
            ):
                q = pending_fallback["qty"]
                sgn = pending_fallback["side"]
                layer_idx = pending_fallback["layer_idx"]
                if q > 0:
                    # Only allow same-direction open/add; never execute opposite-direction fallback.
                    if side == 0 or side == sgn:
                        if sgn > 0:
                            fill_px = execute("buy", q, o[i], i, f"ENTRY_L{layer_idx+1}", "ENTRY_FALLBACK", "taker")
                        else:
                            fill_px = execute("sell", q, o[i], i, f"ENTRY_L{layer_idx+1}", "ENTRY_FALLBACK", "taker")
                        taker_fallback_count += 1
                        if side == 0:
                            side = sgn
                            qty = q
                            avg_px = fill_px
                            entry_idx = i
                            round_start_cash = pending_fallback["cash_before"]
                            layers_used = 1
                            tp1_done = False
                        else:
                            new_qty = qty + q
                            avg_px = (avg_px * qty + fill_px * q) / max(new_qty, 1e-12)
                            qty = new_qty
                            layers_used = max(layers_used, layer_idx + 1)
                            layer_add_count += 1
            pending_fallback = None

        if pending_signal is not None and pending_signal["post_idx"] == i:
            assert i == pending_signal["signal_idx"] + 1, "Order posting must be t+1"
            if (
                box_active
                and (not runner_active)
                and i >= freeze_until_i
                and (not early_stop_due_to_bad_probe)
                and (not invalidate_active_now)
                and (not perf_stop_active_now)
                and start_allowed_now
                and regime_start_allowed_now
                and (not cb_active_now)
                and (not structural_cooldown_now)
            ):
                if cfg.entry_execution_mode == "taker_entry":
                    q = float(pending_signal["qty"])
                    sgn = int(pending_signal["side"])
                    layer_idx = int(pending_signal["layer_idx"])
                    if q > 0.0 and (side == 0 or side == sgn):
                        if sgn > 0:
                            fill_px = execute("buy", q, o[i], i, f"ENTRY_L{layer_idx+1}", "ENTRY_TAKER", "taker")
                        else:
                            fill_px = execute("sell", q, o[i], i, f"ENTRY_L{layer_idx+1}", "ENTRY_TAKER", "taker")
                        if side == 0:
                            side = sgn
                            qty = q
                            avg_px = fill_px
                            entry_idx = i
                            round_start_cash = pending_signal["cash_before"]
                            layers_used = 1
                            tp1_done = False
                        else:
                            new_qty = qty + q
                            avg_px = (avg_px * qty + fill_px * q) / max(new_qty, 1e-12)
                            qty = new_qty
                            layers_used = max(layers_used, layer_idx + 1)
                            layer_add_count += 1
                else:
                    pending_entry = dict(pending_signal)
                    pending_entry["taker_benchmark_open"] = o[i]
                    signal_close = float(pending_entry["signal_close"])
                    improve_px = float(pending_entry["improve_px"])
                    if pending_entry["side"] > 0:
                        raw_limit = signal_close - improve_px
                        min_limit = o[i] * (1.0 - p.min_improve_bps / 10000.0)
                        limit_px = min(raw_limit, min_limit)
                    else:
                        raw_limit = signal_close + improve_px
                        min_limit = o[i] * (1.0 + p.min_improve_bps / 10000.0)
                        limit_px = max(raw_limit, min_limit)
                    pending_entry["limit_px"] = max(limit_px, 1e-9)
                    maker_orders += 1
            pending_signal = None

        if pending_entry is not None and i >= pending_entry["post_idx"]:
            assert i >= pending_entry["post_idx"], "Maker fill cannot happen before post bar"
            if (
                (not box_active)
                or runner_active
                or (i < freeze_until_i)
                or early_stop_due_to_bad_probe
                or invalidate_active_now
                or perf_stop_active_now
                or (not start_allowed_now)
                or (not regime_start_allowed_now)
                or cb_active_now
                or structural_cooldown_now
            ):
                missed_entry += 1
                pending_entry = None
                continue
            sgn = pending_entry["side"]
            lp = pending_entry["limit_px"]
            q = pending_entry["qty"]
            layer_idx = pending_entry["layer_idx"]
            filled = False
            if side != 0 and side != sgn:
                # Position direction changed after signal; cancel stale opposite maker order.
                missed_entry += 1
                pending_entry = None
                continue
            # Strict cross-through maker fill model on OHLC bars:
            # no equality fills at wick extremes and no probabilistic fills.
            can_be_maker = not np.isclose(lp, pending_entry["taker_benchmark_open"], rtol=0.0, atol=1e-12)
            eps = 1e-6
            touched_now = bool(
                can_be_maker
                and (
                    (sgn > 0 and l[i] < (lp - eps))
                    or (sgn < 0 and h[i] > (lp + eps))
                )
            )
            if touched_now and ("queue_trigger_idx" not in pending_entry):
                pending_entry["queue_trigger_idx"] = i
            delay_ok = False
            if "queue_trigger_idx" in pending_entry:
                trigger_i = int(pending_entry["queue_trigger_idx"])
                delay_ok = i >= (trigger_i + max(int(cfg.maker_queue_delay_bars), 0))
            if touched_now and delay_ok:
                if sgn > 0:
                    fill_px = execute("buy", q, lp, i, f"ENTRY_L{layer_idx+1}", "ENTRY_MAKER", "maker")
                else:
                    fill_px = execute("sell", q, lp, i, f"ENTRY_L{layer_idx+1}", "ENTRY_MAKER", "maker")
                if side == 0:
                    side = sgn
                    qty = q
                    avg_px = fill_px
                    entry_idx = i
                    round_start_cash = pending_entry["cash_before"]
                    layers_used = 1
                    tp1_done = False
                else:
                    new_qty = qty + q
                    avg_px = (avg_px * qty + fill_px * q) / max(new_qty, 1e-12)
                    qty = new_qty
                    layers_used = max(layers_used, layer_idx + 1)
                    layer_add_count += 1
                maker_fills += 1
                if sgn > 0:
                    improve_sum_bps += (pending_entry["taker_benchmark_open"] - lp) / pending_entry["taker_benchmark_open"] * 10000.0
                else:
                    improve_sum_bps += (lp - pending_entry["taker_benchmark_open"]) / pending_entry["taker_benchmark_open"] * 10000.0
                improve_cnt += 1
                filled = True

            if filled:
                pending_entry = None
            elif i >= pending_entry["expire_idx"]:
                missed_entry += 1
                if p.maker_fallback_taker and i + 1 < n:
                    pending_fallback = {
                        "execute_idx": i + 1,
                        "signal_idx": i,
                        "side": pending_entry["side"],
                        "layer_idx": pending_entry["layer_idx"],
                        "qty": pending_entry["qty"],
                        "cash_before": pending_entry["cash_before"],
                    }
                pending_entry = None

        if side != 0 and pending_exit is None:
            bars_held = i - entry_idx if entry_idx >= 0 else 0
            reason = None
            full = True
            close_qty = qty
            activate_runner = False
            runner_init_trail = np.nan
            if runner_active:
                hard_stop_hit = (side > 0 and l[i] <= long_stop_rt) or (side < 0 and h[i] >= short_stop_rt)
                runner_stop_hit = bool(
                    np.isfinite(runner_trail_px)
                    and (
                        (side > 0 and l[i] <= runner_trail_px)
                        or (side < 0 and h[i] >= runner_trail_px)
                    )
                )
                if hard_stop_hit:
                    reason = "HARD_STOP"
                elif runner_stop_hit:
                    reason = "RUNNER_STOP"
                elif np.isfinite(atr[i]):
                    if side > 0:
                        next_trail = c[i] - float(cfg.runner_atr_mult) * atr[i]
                        runner_trail_px = float(next_trail) if not np.isfinite(runner_trail_px) else max(float(runner_trail_px), float(next_trail))
                    else:
                        next_trail = c[i] + float(cfg.runner_atr_mult) * atr[i]
                        runner_trail_px = float(next_trail) if not np.isfinite(runner_trail_px) else min(float(runner_trail_px), float(next_trail))
            else:
                atr_stop_hit = bool(
                    cfg.atr_stop_mode == "on"
                    and np.isfinite(atr[i])
                    and (
                        (side > 0 and l[i] <= (avg_px - float(p.atr_stop_mult) * atr[i]))
                        or (side < 0 and h[i] >= (avg_px + float(p.atr_stop_mult) * atr[i]))
                    )
                )
                hard_stop_hit = (side > 0 and l[i] <= long_stop_rt) or (side < 0 and h[i] >= short_stop_rt)
                tp2_hit = np.isfinite(x) and ((side > 0 and x >= p.mid2) or (side < 0 and x <= 1.0 - p.mid2))
                tp1_hit = np.isfinite(x) and ((side > 0 and x >= p.mid1) or (side < 0 and x <= 1.0 - p.mid1))
                time_stop_bars_eff = max(int(cfg.local_time_stop_bars), 1)
                time_hit = bars_held >= time_stop_bars_eff

                if atr_stop_hit:
                    reason = "ATR_STOP"
                elif hard_stop_hit:
                    reason = "HARD_STOP"
                elif tp2_hit:
                    if cfg.enable_runner and (0.0 < float(cfg.runner_pct) < 1.0):
                        keep_qty = qty * float(cfg.runner_pct)
                        close_qty = qty - keep_qty
                        if keep_qty > 1e-12 and close_qty > 1e-12 and np.isfinite(atr[i]):
                            reason = "TP2_PARTIAL"
                            full = False
                            activate_runner = True
                            if side > 0:
                                runner_init_trail = c[i] - float(cfg.runner_atr_mult) * atr[i]
                            else:
                                runner_init_trail = c[i] + float(cfg.runner_atr_mult) * atr[i]
                        else:
                            reason = "TP2"
                            close_qty = qty
                    else:
                        reason = "TP2"
                elif (not tp1_done) and tp1_hit:
                    reason = "TP1"
                    full = False
                    close_qty = qty * 0.50
                elif time_hit:
                    reason = "TIME_STOP"

            if reason is not None and i + 1 < n:
                if reason == "HARD_STOP":
                    for ex_i in expand_apply_indices:
                        if i >= ex_i and i <= ex_i + 6:
                            hard_stop_after_expand6_count += 1
                            break
                stop_cd_used = int(p.cooldown_stop_bars + (6 if degrade_level == 2 else 0))
                pending_exit = {
                    "execute_idx": i + 1,
                    "signal_idx": i,
                    "full": full,
                    "qty": close_qty,
                    "reason": reason,
                    "activate_runner": activate_runner,
                    "runner_init_trail": runner_init_trail,
                    "cooldown_stop_bars_used": stop_cd_used if reason in {"HARD_STOP", "ATR_STOP", "BOX_INVALID", "CIRCUIT_BREAK"} else int(p.cooldown_stop_bars),
                }

        if np.isfinite(atr[i]):
            down_break = c[i] < (box_low_rt - break_buf)
            up_break = c[i] > (box_high_rt + break_buf)
            break_dn = break_dn + 1 if down_break else 0
            break_up = break_up + 1 if up_break else 0
            if bar_live and (down_break or up_break):
                close_outside_count += 1

            if (not macro_dynamic) and cfg.dyn_cfg.mode != "none" and box_active and (i + 1 < n):
                confirmed = (break_dn >= cfg.dyn_cfg.break_confirm_closes) or (break_up >= cfg.dyn_cfg.break_confirm_closes)
                if cfg.dyn_cfg.mode == "expand_invalidate" and confirmed:
                    pending_box_update = {"apply_i": i + 1, "active": False}
                    if cfg.dyn_cfg.invalidate_force_close and side != 0 and pending_exit is None:
                        pending_exit = {
                            "execute_idx": i + 1,
                            "signal_idx": i,
                            "full": True,
                            "qty": qty,
                            "reason": "BOX_INVALID",
                            "cooldown_stop_bars_used": int(p.cooldown_stop_bars + (6 if degrade_level == 2 else 0)),
                        }
                elif (not confirmed) and expand_count < cfg.dyn_cfg.max_expands and pending_box_update is None:
                    if down_break:
                        new_low = min(box_low_rt, l[i] - cfg.dyn_cfg.expand_atr_mult * atr[i])
                        pending_box_update = {"apply_i": i + 1, "low": float(new_low), "high": float(box_high_rt)}
                    elif up_break:
                        new_high = max(box_high_rt, h[i] + cfg.dyn_cfg.expand_atr_mult * atr[i])
                        pending_box_update = {"apply_i": i + 1, "low": float(box_low_rt), "high": float(new_high)}

        # Start gate evidence and eligibility are computed at bar t close and apply from t+1.
        if np.isfinite(atr[i]):
            touch_margin_now = float(cfg.gate_edge_reject_atr_mult) * float(atr[i])
            if c[i] >= (box_high_rt - touch_margin_now):
                edge_pending_up.append({"idx": float(i), "touch_close": float(c[i]), "atr_touch": float(atr[i])})
            if c[i] <= (box_low_rt + touch_margin_now):
                edge_pending_dn.append({"idx": float(i), "touch_close": float(c[i]), "atr_touch": float(atr[i])})

            up_next: List[Dict[str, float]] = []
            for ev in edge_pending_up:
                ev_i = int(ev["idx"])
                if i <= ev_i:
                    up_next.append(ev)
                    continue
                confirm_px = float(ev["touch_close"]) - float(cfg.gate_edge_reject_atr_mult) * float(ev["atr_touch"])
                if c[i] <= confirm_px:
                    edge_confirm_up.append(i)
                elif (i - ev_i) <= gate_lb:
                    up_next.append(ev)
            edge_pending_up = up_next

            dn_next: List[Dict[str, float]] = []
            for ev in edge_pending_dn:
                ev_i = int(ev["idx"])
                if i <= ev_i:
                    dn_next.append(ev)
                    continue
                confirm_px = float(ev["touch_close"]) + float(cfg.gate_edge_reject_atr_mult) * float(ev["atr_touch"])
                if c[i] >= confirm_px:
                    edge_confirm_dn.append(i)
                elif (i - ev_i) <= gate_lb:
                    dn_next.append(ev)
            edge_pending_dn = dn_next

        edge_min_i = i - gate_lb + 1
        edge_confirm_up = [xj for xj in edge_confirm_up if xj >= edge_min_i]
        edge_confirm_dn = [xj for xj in edge_confirm_dn if xj >= edge_min_i]
        gate_cond1 = bool(np.isfinite(adx_arr[i]) and (adx_arr[i] < float(cfg.gate_adx_thresh)))
        gate_cond2 = bool(
            (np.isfinite(ema_slope_abs_arr[i]) and (ema_slope_abs_arr[i] < float(cfg.gate_ema_slope_thresh)))
            or (np.isfinite(chop_arr[i]) and (chop_arr[i] >= float(cfg.gate_chop_thresh)))
        )
        gate_cond3 = bool(max(len(edge_confirm_up), len(edge_confirm_dn)) >= int(cfg.gate_edge_reject_min_count))
        if cfg.start_gate_mode == "on":
            start_allowed_eval_i = (int(gate_cond1) + int(gate_cond2) + int(gate_cond3)) >= 2
        else:
            start_allowed_eval_i = True
        if bar_live:
            start_gate_eval_count += 1
        if bar_live and start_allowed_eval_i:
            start_gate_pass_count += 1

        regime_cond1 = bool(np.isfinite(adx_arr[i]) and (adx_arr[i] < float(cfg.regime_gate_adx_thresh)))
        regime_cond2 = bool(np.isfinite(bb_width_arr[i]) and (bb_width_arr[i] > float(cfg.regime_gate_bbwidth_min)))
        regime_cond3 = bool(np.isfinite(chop_arr[i]) and (chop_arr[i] > float(cfg.regime_gate_chop_thresh)))
        if cfg.regime_gate_mode == "on":
            regime_start_allowed_eval_i = bool(regime_cond1 and regime_cond2 and regime_cond3)
        else:
            regime_start_allowed_eval_i = True
        if bar_live:
            regime_start_gate_eval_count += 1
        if bar_live and regime_start_allowed_eval_i:
            regime_start_gate_pass_count += 1

        outside_up_now = bool(np.isfinite(bb_upper_arr[i]) and (c[i] > bb_upper_arr[i]))
        outside_dn_now = bool(np.isfinite(bb_lower_arr[i]) and (c[i] < bb_lower_arr[i]))
        cb_outside_up_streak = cb_outside_up_streak + 1 if outside_up_now else 0
        cb_outside_dn_streak = cb_outside_dn_streak + 1 if outside_dn_now else 0
        cb_cond_outside = bool(
            (cb_outside_up_streak >= int(cfg.circuit_break_outside_consecutive))
            or (cb_outside_dn_streak >= int(cfg.circuit_break_outside_consecutive))
        )
        cb_cond1 = bool(np.isfinite(adx_arr[i]) and (adx_arr[i] > float(cfg.circuit_break_adx_thresh)))
        cb_cond2 = bool(
            np.isfinite(bb_width_arr[i]) and np.isfinite(bb_width_q50_arr[i]) and (bb_width_arr[i] > bb_width_q50_arr[i])
        )
        cb_trigger_now = bool(cfg.circuit_breaker_mode == "on" and (cb_cond1 or cb_cond2 or cb_cond_outside))
        if cb_trigger_now and (i + 1 < n):
            if (i + 1) >= cb_disable_until_i:
                if bar_live:
                    cb_trigger_count += 1
            cb_disable_until_i = max(cb_disable_until_i, i + 1 + int(cfg.cooldown_cb_bars))
            pending_signal = None
            pending_entry = None
            pending_fallback = None
            if cfg.cb_force_flatten and side != 0 and pending_exit is None:
                pending_exit = {
                    "execute_idx": i + 1,
                    "signal_idx": i,
                    "full": True,
                    "qty": qty,
                    "reason": "CIRCUIT_BREAK",
                    "cooldown_stop_bars_used": int(p.cooldown_stop_bars),
                }

        if cfg.invalidate_mode == "on" and np.isfinite(atr[i]):
            if cfg.invalidate_buffer_mode == "pct":
                inv_buf = float(cfg.invalidate_buffer_pct) * box_width_rt
            else:
                inv_buf = float(cfg.invalidate_buffer_atr_mult) * atr[i]
            inv_down = bool(c[i] < (box_low_rt - inv_buf))
            inv_up = bool(c[i] > (box_high_rt + inv_buf))
            inv_break_dn_engine = inv_break_dn_engine + 1 if inv_down else 0
            inv_break_up_engine = inv_break_up_engine + 1 if inv_up else 0
            inv_trigger = bool(
                (inv_break_dn_engine == int(cfg.invalidate_m)) or (inv_break_up_engine == int(cfg.invalidate_m))
            )
            if inv_trigger and (i + 1 < n):
                if bar_live:
                    invalidate_count_engine += 1
                if invalid_first_ts is None:
                    invalid_first_ts = times[i]
                invalidate_start_i = min(invalidate_start_i, i + 1)
                invalidate_until_i = max(invalidate_until_i, i + 1 + int(cfg.cooldown_after_invalidate_bars))
                pending_signal = None
                pending_entry = None
                pending_fallback = None
                if cfg.invalidate_action == "force_flatten" and side != 0 and pending_exit is None:
                    pending_exit = {
                        "execute_idx": i + 1,
                        "signal_idx": i,
                        "full": True,
                        "qty": qty,
                        "reason": "INVALID",
                        "cooldown_stop_bars_used": int(p.cooldown_stop_bars),
                    }

        if cfg.perf_stop_mode == "on" and (i + 1 < n) and (not is_perf_active(i + 1)):
            w_perf = max(int(cfg.perf_window_trades), 1)
            if len(perf_recent_rounds) >= w_perf:
                recent = perf_recent_rounds[-w_perf:]
                gp_recent = float(sum(max(float(r["pnl"]), 0.0) for r in recent))
                gl_recent = float(sum(max(-float(r["pnl"]), 0.0) for r in recent))
                pf_recent = gp_recent / gl_recent if gl_recent > 1e-12 else float("inf")
                hs_recent = float(sum(1 for r in recent if bool(r["hard_stop"])) / len(recent))
                perf_trigger = bool(
                    (pf_recent < float(cfg.perf_min_profit_factor))
                    or (hs_recent > float(cfg.perf_max_hard_stop_ratio))
                )
                if perf_trigger:
                    if bar_live:
                        perf_stop_count += 1
                    perf_start_i = min(perf_start_i, i + 1)
                    perf_until_i = max(perf_until_i, i + 1 + int(cfg.cooldown_after_perf_stop_bars))
                    pending_signal = None
                    pending_entry = None
                    pending_fallback = None
                    if cfg.perf_action == "force_flatten" and side != 0 and pending_exit is None:
                        pending_exit = {
                            "execute_idx": i + 1,
                            "signal_idx": i,
                            "full": True,
                            "qty": qty,
                            "reason": "PERF_STOP",
                            "cooldown_stop_bars_used": int(p.cooldown_stop_bars),
                        }

        start_allowed_next = bool(start_allowed_eval_i)
        regime_start_allowed_next = bool(regime_start_allowed_eval_i)
        cb_active_next = is_cb_active(i + 1) if (i + 1 < n) else False
        invalidate_active_next = is_invalidate_active(i + 1) if (i + 1 < n) else False
        perf_stop_active_next = is_perf_active(i + 1) if (i + 1 < n) else False

        if pending_signal is None and pending_entry is None and pending_exit is None and (not runner_active):
            if (
                i + 1 < n
                and bar_live
                and np.isfinite(atr[i])
                and box_active
                and i >= freeze_until_i
                and (not trade_box_invalid)
                and (not early_stop_due_to_bad_probe)
                and start_allowed_eval_i
                and regime_start_allowed_eval_i
                and (not cb_active_next)
                and (not invalidate_active_next)
                and (not perf_stop_active_next)
                and (not structural_cooldown_now)
                and (not structural_break_now)
            ):
                long_ready = (i >= long_block_until) and (not long_need_center)
                short_ready = (i >= short_block_until) and (not short_need_center)

                short_L = 1.0 - p.long_L
                long_levels = [p.long_L - k * p.layer_step for k in range(3)]
                short_levels = [short_L + k * p.layer_step for k in range(3)]
                if degrade_level == 2:
                    max_layers_allowed = 1
                elif degrade_level == 1:
                    max_layers_allowed = 2
                else:
                    max_layers_allowed = 3

                cand_side = 0
                cand_layer = -1
                if side == 0:
                    if long_ready and x <= long_levels[0]:
                        cand_side = 1
                        cand_layer = 0
                    elif allow_short and short_ready and short_allowed_rule and x >= short_levels[0]:
                        cand_side = -1
                        cand_layer = 0
                elif side > 0 and layers_used < 3:
                    if layers_used >= max_layers_allowed:
                        if x <= long_levels[layers_used]:
                            layer_disabled_count_due_to_degrade += 1
                    elif x <= long_levels[layers_used]:
                        cand_side = 1
                        cand_layer = layers_used
                elif allow_short and side < 0 and layers_used < 3:
                    if layers_used >= max_layers_allowed:
                        if x >= short_levels[layers_used]:
                            layer_disabled_count_due_to_degrade += 1
                    elif short_allowed_rule and x >= short_levels[layers_used]:
                        cand_side = -1
                        cand_layer = layers_used

                if cand_side != 0:
                    improve_px = p.improve_mult * atr[i] * 0.01
                    signal_close = c[i]
                    prelim_limit = signal_close - improve_px if cand_side > 0 else signal_close + improve_px
                    prelim_limit = max(prelim_limit, 1e-9)
                    eq_now = mark_equity(c[i])
                    stop_px = long_stop_rt if cand_side > 0 else short_stop_rt
                    per_unit_risk = abs(prelim_limit - stop_px)
                    if per_unit_risk > 1e-12:
                        if degrade_level == 2:
                            base_risk_mult = 0.5
                        elif degrade_level == 1:
                            base_risk_mult = 0.75
                        else:
                            base_risk_mult = 1.0
                        base_risk_eff = p.base_risk_pct * base_risk_mult
                        base_qty = eq_now * (base_risk_eff / 100.0) / per_unit_risk
                        target_qty = base_qty * p.layer_weights[cand_layer]
                        curr_qty = qty if side == cand_side else 0.0
                        max_qty_allowed = max(eq_now * p.max_leverage / prelim_limit, 0.0)
                        add_cap = max(max_qty_allowed - curr_qty, 0.0)
                        q = min(target_qty, add_cap)
                        if q > 1e-10:
                            signal_count_total += 1
                            if cand_side > 0:
                                signal_count_long += 1
                            else:
                                signal_count_short += 1
                            pending_signal = {
                                "signal_idx": i,
                                "post_idx": i + 1,
                                "expire_idx": i + p.maker_ttl_bars,
                                "side": cand_side,
                                "layer_idx": cand_layer,
                                "qty": q,
                                "signal_close": signal_close,
                                "improve_px": improve_px,
                                "cash_before": cash if side == 0 else round_start_cash,
                            }

        if bar_live:
            equity_rows.append(
                {
                    "time": times[i],
                    "equity": mark_equity(c[i]),
                    "close": c[i],
                    "margin_utilization": margin_utilization(c[i]),
                    "start_allowed": int(start_allowed_now),
                    "regime_start_allowed": int(regime_start_allowed_now),
                    "start_allowed_all": int(start_allowed_now and regime_start_allowed_now),
                    "cb_disable_new_entries": int(cb_active_now),
                    "structural_cooldown_active": int(structural_cooldown_now),
                    "side": side,
                    "qty": qty,
                    "avg_px": avg_px,
                    "runner_active": int(runner_active),
                    "runner_trail_px": runner_trail_px,
                    "x": x,
                    "risk_low_rt": box_low_rt,
                    "risk_high_rt": box_high_rt,
                    "risk_box_active": box_active,
                    "risk_expand_count": expand_count,
                    "trade_low": trade_low,
                    "trade_high": trade_high,
                    "trade_box_width": trade_box_width,
                    "trade_box_invalid": bool(trade_box_invalid),
                    "box_low_rt": box_low_rt,
                    "box_high_rt": box_high_rt,
                    "box_active": box_active,
                    "expand_count": expand_count,
                    "break_dn": break_dn,
                    "break_up": break_up,
                    "box_invalid": int(invalidate_active_now),
                    "perf_stop_active": int(perf_stop_active_now),
                    "cooldown_left": int(
                        max(
                            (cb_disable_until_i - i) if cb_active_now else 0,
                            (invalidate_until_i - i) if invalidate_active_now else 0,
                            (perf_until_i - i) if perf_stop_active_now else 0,
                        )
                    ),
                }
            )

        if structural_break_now:
            if bar_live:
                structural_break_count += 1
            structural_cooldown_left = max(int(cfg.structural_cooldown_bars), 0)
        elif structural_cooldown_left > 0:
            structural_cooldown_left -= 1

    if side != 0:
        last = n - 1
        close_full(last, c[last], "FORCE_END")
        if equity_rows:
            equity_rows[-1]["equity"] = mark_equity(c[last])
            equity_rows[-1]["side"] = side
            equity_rows[-1]["qty"] = qty
            equity_rows[-1]["avg_px"] = avg_px

    trades_df = pd.DataFrame(trades)
    rounds_df = pd.DataFrame(rounds)
    eq_df = pd.DataFrame(equity_rows).set_index("time")
    bars_df = active_wdf.copy()

    final_eq = float(eq_df["equity"].iloc[-1]) if not eq_df.empty else initial_equity
    ret_pct = (final_eq / max(initial_equity, 1e-12) - 1.0) * 100.0
    max_dd = calc_max_dd_pct(eq_df["equity"])

    if rounds_df.empty:
        pf = 0.0
        pf_reliable = True
        exp = 0.0
        hs_ratio = 0.0
        hard_stop_count = 0
        atr_stop_count = 0
        time_stop_count = 0
        runner_stop_count = 0
        circuit_break_exit_count = 0
        avg_layers = 0.0
        avg_hold_bars = 0.0
        long_trade_count = 0
        short_trade_count = 0
    else:
        gp = float(rounds_df.loc[rounds_df["pnl"] > 0, "pnl"].sum())
        gl = float(-rounds_df.loc[rounds_df["pnl"] < 0, "pnl"].sum())
        pf_reliable = gl > 1e-12
        pf = gp / gl if pf_reliable else float("inf")
        exp = float(rounds_df["pnl"].mean())
        hs_ratio = float((rounds_df["exit_reason"] == "HARD_STOP").mean())
        hard_stop_count = int((rounds_df["exit_reason"] == "HARD_STOP").sum())
        atr_stop_count = int((rounds_df["exit_reason"] == "ATR_STOP").sum())
        time_stop_count = int((rounds_df["exit_reason"] == "TIME_STOP").sum())
        runner_stop_count = int((rounds_df["exit_reason"] == "RUNNER_STOP").sum())
        circuit_break_exit_count = int((rounds_df["exit_reason"] == "CIRCUIT_BREAK").sum())
        avg_layers = float(rounds_df["layers_used"].mean())
        avg_hold_bars = float(rounds_df["hold_bars"].mean())
        long_trade_count = int((rounds_df["side"] == "long").sum())
        short_trade_count = int((rounds_df["side"] == "short").sum())

    days = max((active_wdf.index[-1] - active_wdf.index[0]).total_seconds() / 86400.0, 1e-9)
    daily_orders = len(trades_df) / days
    fill_rate = float(maker_fills / maker_orders) if maker_orders > 0 else 0.0
    missed_entry_rate = float(missed_entry / maker_orders) if maker_orders > 0 else 0.0
    avg_improve = float(improve_sum_bps / improve_cnt) if improve_cnt > 0 else 0.0
    trade_box_invalid_ratio = float(trade_box_invalid_count / active_bar_count) if active_bar_count > 0 else 0.0
    close_outside_ratio = float(close_outside_count / active_bar_count) if active_bar_count > 0 else 0.0
    objective = ret_pct - 0.6 * max_dd - 0.2 * missed_entry_rate * 100.0

    entry_mask = trades_df["tag"].str.startswith("ENTRY", na=False) if (not trades_df.empty and "tag" in trades_df.columns) else pd.Series([], dtype=bool)
    exit_mask = trades_df["tag"].str.startswith("EXIT", na=False) if (not trades_df.empty and "tag" in trades_df.columns) else pd.Series([], dtype=bool)
    filled_entries = int(entry_mask.sum()) if len(entry_mask) else 0
    filled_exits = int(exit_mask.sum()) if len(exit_mask) else 0
    effective_fill_rate = float(filled_entries / (filled_entries + missed_entry)) if (filled_entries + missed_entry) > 0 else 0.0
    total_fees = float(maker_fee_total + taker_fee_total)
    avg_cost_per_trade = float(total_fees / len(trades_df)) if len(trades_df) > 0 else 0.0
    taker_fallback_ratio = float(taker_fallback_count / filled_entries) if filled_entries > 0 else 0.0
    degrade_mode_bars_ratio = float(degrade_mode_bars / active_bar_count) if active_bar_count > 0 else 0.0
    degrade_level1_ratio = float(degrade_level1_bars / active_bar_count) if active_bar_count > 0 else 0.0
    degrade_level2_ratio = float(degrade_level2_bars / active_bar_count) if active_bar_count > 0 else 0.0
    degrade_too_strong = bool(cfg.degrade_on_trend and ((int(layer_add_count) == 0) or (float(avg_layers) <= 1.1)))
    short_enabled_ratio = (
        float(short_enabled_true_count / short_enabled_total_count)
        if short_enabled_total_count > 0
        else np.nan
    )
    start_gate_pass_rate = float(start_gate_pass_count / start_gate_eval_count) if start_gate_eval_count > 0 else np.nan
    regime_start_gate_pass_rate = (
        float(regime_start_gate_pass_count / regime_start_gate_eval_count) if regime_start_gate_eval_count > 0 else np.nan
    )
    cb_disable_ratio = float(cb_disable_bars / active_bar_count) if active_bar_count > 0 else 0.0
    structural_cooldown_ratio = float(structural_cooldown_active_bars / active_bar_count) if active_bar_count > 0 else 0.0
    max_margin_utilization = float(eq_df["margin_utilization"].max()) if ("margin_utilization" in eq_df.columns and not eq_df.empty) else 0.0

    stats = {
        "return_pct": ret_pct,
        "max_dd_pct": max_dd,
        "profit_factor": float(pf),
        "pf_reliable": bool(pf_reliable),
        "expectancy_after_cost": exp,
        "round_trips": int(len(rounds_df)),
        "hard_stop_count": int(hard_stop_count),
        "atr_stop_count": int(atr_stop_count),
        "time_stop_count": int(time_stop_count),
        "runner_stop_count": int(runner_stop_count),
        "circuit_break_exit_count": int(circuit_break_exit_count),
        "avg_hold_bars": avg_hold_bars,
        "long_trade_count": int(long_trade_count),
        "short_trade_count": int(short_trade_count),
        "daily_orders": float(daily_orders),
        "orders_total": int(len(trades_df)),
        "signal_count_total": int(signal_count_total),
        "signal_count_long": int(signal_count_long),
        "signal_count_short": int(signal_count_short),
        "filled_entries": filled_entries,
        "filled_exits": filled_exits,
        "effective_fill_rate": effective_fill_rate,
        "hard_stop_ratio": hs_ratio,
        "avg_layers": avg_layers,
        "layer_add_count": int(layer_add_count),
        "layer_disabled_count_due_to_degrade": int(layer_disabled_count_due_to_degrade),
        "degrade_mode_bars_ratio": degrade_mode_bars_ratio,
        "degrade_level1_ratio": degrade_level1_ratio,
        "degrade_level2_ratio": degrade_level2_ratio,
        "degrade_too_strong": degrade_too_strong,
        "short_enabled_ratio": short_enabled_ratio,
        "start_gate_pass_rate": start_gate_pass_rate,
        "regime_start_gate_pass_rate": regime_start_gate_pass_rate,
        "cb_trigger_count": int(cb_trigger_count),
        "cb_disable_bars_ratio": cb_disable_ratio,
        "structural_break_count": int(structural_break_count),
        "structural_cooldown_bars_ratio": structural_cooldown_ratio,
        "missed_entry": int(missed_entry),
        "missed_entry_rate": missed_entry_rate,
        "entry_fill_rate": fill_rate,
        "avg_improve_vs_taker": avg_improve,
        "fill_prob_used": 1.0,
        "queue_delay_used": int(cfg.maker_queue_delay_bars),
        "taker_fallback_count": int(taker_fallback_count),
        "taker_fallback_ratio": taker_fallback_ratio,
        "maker_fees": float(maker_fee_total),
        "taker_fees": float(taker_fee_total),
        "avg_cost_per_trade": avg_cost_per_trade,
        "box_invalid_count": int(box_invalid_count + invalidate_count_engine),
        "box_invalid_count_dynamic": int(box_invalid_count),
        "box_invalid_count_engine": int(invalidate_count_engine),
        "invalid_first_ts": str(invalid_first_ts) if invalid_first_ts is not None else "",
        "box_invalid_active_ratio": float(invalidate_active_bars / active_bar_count) if active_bar_count > 0 else 0.0,
        "perf_stop_count": int(perf_stop_count),
        "perf_stop_active_ratio": float(perf_active_bars / active_bar_count) if active_bar_count > 0 else 0.0,
        "risk_expand_count": int(expand_count),
        "expand_count": int(expand_count),
        "trade_box_invalid_count": int(trade_box_invalid_count),
        "trade_box_invalid_ratio": trade_box_invalid_ratio,
        "close_outside_ratio": close_outside_ratio,
        "first_box_low": float(eq_df["risk_low_rt"].iloc[0]) if not eq_df.empty else float(first_box_low_rt),
        "first_box_high": float(eq_df["risk_high_rt"].iloc[0]) if not eq_df.empty else float(first_box_high_rt),
        "final_box_low": float(eq_df["risk_low_rt"].iloc[-1]) if not eq_df.empty else float(box_low_rt),
        "final_box_high": float(eq_df["risk_high_rt"].iloc[-1]) if not eq_df.empty else float(box_high_rt),
        "box_inactive_time": str(box_inactive_time) if box_inactive_time is not None else "",
        "hard_stop_after_expand6_count": int(hard_stop_after_expand6_count),
        "hard_stop_after_expand6_hit": bool(hard_stop_after_expand6_count > 0),
        "selected_risk_high": float(cfg.box_high) if cfg.box_source != "dynamic" else np.nan,
        "early_stop_due_to_bad_probe": bool(early_stop_due_to_bad_probe),
        "early_stop_reason": str(early_stop_reason),
        "early_stop_time": str(early_stop_time) if early_stop_time is not None else "",
        "objective": objective,
        "initial_equity": initial_equity,
        "final_equity": final_eq,
        "max_margin_utilization": max_margin_utilization,
    }
    return {"stats": stats, "trades": trades_df, "rounds": rounds_df, "equity": eq_df, "bars": bars_df}


def build_fixed_probe_params(cfg: RunConfig) -> Params:
    return Params(
        long_L=0.38,
        mid1=0.60,
        base_risk_pct=0.35,
        atr_stop_mult=1.5,
        macro_lookback_bars=max(int(cfg.macro_lookback_bars), 2),
        layer_weights_value=cfg.override_layer_weights if cfg.override_layer_weights is not None else (1.0, 1.4, 2.0),
        max_leverage_value=float(cfg.override_max_leverage) if cfg.override_max_leverage is not None else 2.0,
    )


def _probe_cache_key(df: pd.DataFrame, cfg: RunConfig) -> Tuple[object, ...]:
    first_ts = df.index[0] if len(df.index) > 0 else None
    last_ts = df.index[-1] if len(df.index) > 0 else None
    dc = cfg.dyn_cfg
    return (
        str(first_ts),
        str(last_ts),
        len(df),
        str(cfg.start_utc),
        str(cfg.end_utc),
        float(cfg.box_low),
        float(cfg.box_high),
        tuple(sorted(float(x) for x in cfg.risk_high_candidates)),
        int(cfg.probe_days),
        str(cfg.probe_metric),
        str(cfg.probe_execution_mode),
        str(cfg.fallback_no_signal_mode),
        float(cfg.probe_tie_eps),
        int(cfg.probe_min_round_trips),
        str(cfg.side_mode),
        float(cfg.commission_pct),
        float(cfg.slippage_pct),
        float(cfg.maker_fill_prob),
        int(cfg.maker_queue_delay_bars),
        int(cfg.seed),
        bool(cfg.degrade_on_trend),
        float(cfg.trend_slope_thresh),
        float(cfg.atr_expand_thresh),
        int(cfg.degrade_risk_lookback_bars),
        int(cfg.structural_cooldown_bars),
        str(cfg.short_enable_rule),
        int(cfg.short_enable_lookback_days),
        int(cfg.short_enable_min_rejects),
        float(cfg.short_enable_touch_band),
        float(cfg.short_enable_reject_close_gap),
        str(cfg.start_gate_mode),
        float(cfg.gate_adx_thresh),
        float(cfg.gate_ema_slope_thresh),
        float(cfg.gate_chop_thresh),
        int(cfg.gate_edge_reject_lookback_bars),
        int(cfg.gate_edge_reject_min_count),
        float(cfg.gate_edge_reject_atr_mult),
        str(cfg.invalidate_mode),
        int(cfg.invalidate_m),
        str(cfg.invalidate_buffer_mode),
        float(cfg.invalidate_buffer_atr_mult),
        float(cfg.invalidate_buffer_pct),
        str(cfg.invalidate_action),
        int(cfg.cooldown_after_invalidate_bars),
        str(cfg.perf_stop_mode),
        int(cfg.perf_window_trades),
        float(cfg.perf_min_profit_factor),
        float(cfg.perf_max_hard_stop_ratio),
        str(cfg.perf_action),
        int(cfg.cooldown_after_perf_stop_bars),
        str(cfg.regime_gate_mode),
        float(cfg.regime_gate_adx_thresh),
        float(cfg.regime_gate_bbwidth_min),
        float(cfg.regime_gate_chop_thresh),
        dc.mode,
        int(dc.break_confirm_closes),
        dc.break_buffer_mode,
        float(dc.break_buffer_pct),
        float(dc.break_buffer_atr_mult),
        float(dc.expand_atr_mult),
        bool(dc.invalidate_force_close),
        int(dc.max_expands),
        int(dc.freeze_after_expand_bars),
        str(cfg.trade_box_mode_default),
        int(cfg.trade_box_lookback_default),
        float(cfg.trade_box_q_low_default),
        float(cfg.trade_box_q_high_default),
        int(cfg.trade_box_ema_len_default),
        float(cfg.trade_box_atr_mult_default),
    )


def run_probe_selection(df: pd.DataFrame, cfg: RunConfig) -> Dict[str, object]:
    if cfg.box_source == "dynamic":
        return {
            "selected_high": np.nan,
            "probe_results": pd.DataFrame(),
            "probe_days": 0,
            "probe_metric": str(cfg.probe_metric),
            "probe_end_exclusive": cfg.start_utc,
            "metric_gap_to_2nd": np.nan,
            "tie_reason": "PROBE_SKIPPED_DYNAMIC_BOX",
            "fallback_v2_used": False,
            "passed_filters_count": 0,
            "probe_param_set": "DYNAMIC_BOX_NO_PROBE",
            "probe_execution_mode": str(cfg.probe_execution_mode),
        }
    candidates = sorted({float(x) for x in cfg.risk_high_candidates if float(x) > cfg.box_low})
    probe_days = max(int(cfg.probe_days), 0)
    probe_metric = str(cfg.probe_metric)
    tie_eps = max(float(cfg.probe_tie_eps), 0.0)
    probe_params = build_fixed_probe_params(cfg)
    if cfg.probe_execution_mode not in {"maker_fallback_taker", "taker_entry"}:
        raise ValueError(f"unsupported probe_execution_mode: {cfg.probe_execution_mode}")
    probe_param_set = str(probe_params.__dict__)

    if (not candidates) or (probe_days <= 0):
        return {
            "selected_high": float(cfg.box_high),
            "probe_results": pd.DataFrame(),
            "probe_days": probe_days,
            "probe_metric": probe_metric,
            "probe_end_exclusive": cfg.start_utc,
            "metric_gap_to_2nd": np.nan,
            "tie_reason": "PROBE_DISABLED",
            "fallback_v2_used": False,
            "passed_filters_count": 0,
            "probe_param_set": probe_param_set,
            "probe_execution_mode": str(cfg.probe_execution_mode),
        }

    cache_key = _probe_cache_key(df, cfg)
    if cache_key in _PROBE_SELECTION_CACHE:
        cached = _PROBE_SELECTION_CACHE[cache_key]
        return {
            "selected_high": float(cached["selected_high"]),
            "probe_results": cached["probe_results"].copy(),
            "probe_days": int(cached["probe_days"]),
            "probe_metric": str(cached["probe_metric"]),
            "probe_end_exclusive": cached["probe_end_exclusive"],
            "metric_gap_to_2nd": float(cached["metric_gap_to_2nd"]) if pd.notna(cached["metric_gap_to_2nd"]) else np.nan,
            "tie_reason": str(cached["tie_reason"]),
            "fallback_v2_used": bool(cached["fallback_v2_used"]),
            "passed_filters_count": int(cached["passed_filters_count"]),
            "probe_param_set": str(cached["probe_param_set"]),
            "probe_execution_mode": str(cached.get("probe_execution_mode", cfg.probe_execution_mode)),
        }

    probe_start = cfg.start_utc
    probe_end_exclusive = probe_start + pd.Timedelta(days=probe_days)
    if probe_end_exclusive >= cfg.end_utc:
        raise ValueError("probe window consumes full backtest window; reduce --probe-days")
    probe_end_inclusive = probe_end_exclusive - pd.Timedelta(nanoseconds=1)

    minimize_metrics = {"max_dd_pct", "hard_stop_ratio", "missed_entry", "missed_entry_rate", "trade_box_invalid_ratio", "box_invalid_count"}
    ascending_metric = probe_metric in minimize_metrics
    probe_rows: List[Dict[str, object]] = []

    probe_cfg_base = clone_cfg(
        cfg,
        start_utc=probe_start,
        end_utc=probe_end_inclusive,
        risk_high_candidates=tuple(),
        probe_days=0,
        enable_early_stop=False,
        entry_execution_mode="taker_entry" if cfg.probe_execution_mode == "taker_entry" else "maker",
    )

    for high in candidates:
        pres = run_backtest(df, probe_params, clone_cfg(probe_cfg_base, box_high=float(high)))
        st = pres["stats"]
        if probe_metric not in st:
            raise KeyError(f"probe metric not found in stats: {probe_metric}")
        probe_round_trips = int(st["round_trips"])
        probe_orders_total = int(st.get("orders_total", 0))
        probe_filled_entries = int(st.get("filled_entries", 0))
        passed_filters = bool(
            (probe_round_trips >= int(cfg.probe_min_round_trips))
            or (probe_orders_total >= PROBE_MIN_ORDERS_TOTAL)
            or (probe_filled_entries >= 1)
        )
        probe_rows.append(
            {
                "candidate_risk_high": float(high),
                "probe_metric_name": probe_metric,
                "probe_metric_value": float(st[probe_metric]),
                "tie_distance_to_input_high": abs(float(high) - float(cfg.box_high)),
                "probe_round_trips": probe_round_trips,
                "probe_orders_total": probe_orders_total,
                "filled_entries": probe_filled_entries,
                "filled_exits": int(st.get("filled_exits", 0)),
                "probe_entry_fill_rate": float(st["entry_fill_rate"]),
                "probe_hard_stop_ratio": float(st["hard_stop_ratio"]),
                "risk_expand_count": int(st.get("risk_expand_count", 0)),
                "box_invalid_count": int(st.get("box_invalid_count", 0)),
                "close_outside_ratio": float(st.get("close_outside_ratio", 0.0)),
                "passed_filters": passed_filters,
                "selected_flag": False,
                "tie_reason": "",
                "metric_gap_to_2nd": np.nan,
                "fallback_v2_used": False,
                "return_pct": float(st["return_pct"]),
                "max_dd_pct": float(st["max_dd_pct"]),
                "profit_factor": float(st["profit_factor"]),
                "round_trips": probe_round_trips,
                "objective": float(st["objective"]),
            }
        )

    probe_df = pd.DataFrame(probe_rows)
    no_signal_all = bool(
        (not probe_df.empty)
        and ((probe_df["probe_orders_total"] == 0) & (probe_df["probe_round_trips"] == 0) & (probe_df["filled_entries"] == 0)).all()
    )
    passed_df = probe_df.loc[probe_df["passed_filters"]].copy()
    fallback_v2_used = False

    if no_signal_all:
        fallback_v2_used = False
        metric_gap_to_2nd = np.nan
        if cfg.fallback_no_signal_mode == "highest_high":
            selected_high = float(max(candidates))
            tie_reason = "FALLBACK_NO_SIGNAL_HIGHEST_HIGH"
        else:
            selected_high = float(cfg.box_high)
            tie_reason = "FALLBACK_NO_SIGNAL_USE_INPUT_HIGH"
        probe_df["selected_flag"] = np.isclose(probe_df["candidate_risk_high"], selected_high, rtol=0.0, atol=1e-12)
        probe_df["metric_gap_to_2nd"] = metric_gap_to_2nd
        probe_df["fallback_v2_used"] = fallback_v2_used
        if bool(probe_df["selected_flag"].any()):
            probe_df.loc[probe_df["selected_flag"], "tie_reason"] = tie_reason
        else:
            probe_df.loc[probe_df.index[0], "tie_reason"] = tie_reason
        probe_df = probe_df.sort_values(["selected_flag", "candidate_risk_high"], ascending=[False, False]).reset_index(drop=True)
        selection = {
            "selected_high": selected_high,
            "probe_results": probe_df,
            "probe_days": probe_days,
            "probe_metric": probe_metric,
            "probe_end_exclusive": probe_end_exclusive,
            "metric_gap_to_2nd": metric_gap_to_2nd,
            "tie_reason": tie_reason,
            "fallback_v2_used": fallback_v2_used,
            "passed_filters_count": int(probe_df["passed_filters"].sum()),
            "probe_param_set": probe_param_set,
            "probe_execution_mode": str(cfg.probe_execution_mode),
        }
        _PROBE_SELECTION_CACHE[cache_key] = {
            "selected_high": selected_high,
            "probe_results": probe_df.copy(),
            "probe_days": probe_days,
            "probe_metric": probe_metric,
            "probe_end_exclusive": probe_end_exclusive,
            "metric_gap_to_2nd": metric_gap_to_2nd,
            "tie_reason": tie_reason,
            "fallback_v2_used": fallback_v2_used,
            "passed_filters_count": int(probe_df["passed_filters"].sum()),
            "probe_param_set": probe_param_set,
            "probe_execution_mode": str(cfg.probe_execution_mode),
        }
        return selection
    elif passed_df.empty:
        fallback_v2_used = True
        base_pool = probe_df.copy()
        for col in ["box_invalid_count", "risk_expand_count", "close_outside_ratio"]:
            if base_pool.empty:
                break
            best_v = base_pool[col].min()
            base_pool = base_pool.loc[base_pool[col] == best_v].copy()
        tie_reason_base = "FALLBACK_V2_MIN_INVALID_EXPAND_OUTSIDE"
    else:
        base_pool = passed_df
        tie_reason_base = "PRIMARY_FILTER_ROUNDTRIP_OR_ORDERS"

    if ascending_metric:
        metric_best = float(base_pool["probe_metric_value"].min())
        metric_rank_df = base_pool.sort_values(["probe_metric_value", "candidate_risk_high"], ascending=[True, False]).reset_index(drop=True)
        tie_pool = base_pool.loc[base_pool["probe_metric_value"] <= (metric_best + tie_eps)].copy()
    else:
        metric_best = float(base_pool["probe_metric_value"].max())
        metric_rank_df = base_pool.sort_values(["probe_metric_value", "candidate_risk_high"], ascending=[False, False]).reset_index(drop=True)
        tie_pool = base_pool.loc[base_pool["probe_metric_value"] >= (metric_best - tie_eps)].copy()

    if len(metric_rank_df) >= 2:
        first_v = float(metric_rank_df.iloc[0]["probe_metric_value"])
        second_v = float(metric_rank_df.iloc[1]["probe_metric_value"])
        metric_gap_to_2nd = (second_v - first_v) if ascending_metric else (first_v - second_v)
    else:
        metric_gap_to_2nd = np.nan

    tie_sorted = tie_pool.sort_values(
        ["candidate_risk_high", "probe_hard_stop_ratio", "max_dd_pct", "tie_distance_to_input_high"],
        ascending=[False, True, True, True],
    )
    selected_high = float(tie_sorted.iloc[0]["candidate_risk_high"])
    if len(tie_pool) > 1:
        tie_reason = f"{tie_reason_base}+EPS_TIE_HIGHER_HIGH_LOWER_HARDSTOP_LOWER_MAXDD_CLOSER_TO_INPUT"
    else:
        tie_reason = f"{tie_reason_base}+METRIC_BEST"

    probe_df["selected_flag"] = probe_df["candidate_risk_high"] == selected_high
    probe_df["metric_gap_to_2nd"] = metric_gap_to_2nd
    probe_df["fallback_v2_used"] = fallback_v2_used
    probe_df.loc[probe_df["selected_flag"], "tie_reason"] = tie_reason

    sort_cols = ["selected_flag", "passed_filters"]
    sort_asc = [False, False]
    if fallback_v2_used:
        sort_cols.extend(["box_invalid_count", "risk_expand_count", "close_outside_ratio", "probe_metric_value", "candidate_risk_high"])
        sort_asc.extend([True, True, True, ascending_metric, False])
    else:
        sort_cols.extend(["probe_metric_value", "candidate_risk_high"])
        sort_asc.extend([ascending_metric, False])
    probe_df = probe_df.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    selection = {
        "selected_high": selected_high,
        "probe_results": probe_df,
        "probe_days": probe_days,
        "probe_metric": probe_metric,
        "probe_end_exclusive": probe_end_exclusive,
        "metric_gap_to_2nd": metric_gap_to_2nd,
        "tie_reason": tie_reason,
        "fallback_v2_used": fallback_v2_used,
        "passed_filters_count": int(probe_df["passed_filters"].sum()),
        "probe_param_set": probe_param_set,
        "probe_execution_mode": str(cfg.probe_execution_mode),
    }
    _PROBE_SELECTION_CACHE[cache_key] = {
        "selected_high": selected_high,
        "probe_results": probe_df.copy(),
        "probe_days": probe_days,
        "probe_metric": probe_metric,
        "probe_end_exclusive": probe_end_exclusive,
        "metric_gap_to_2nd": metric_gap_to_2nd,
        "tie_reason": tie_reason,
        "fallback_v2_used": fallback_v2_used,
        "passed_filters_count": int(probe_df["passed_filters"].sum()),
        "probe_param_set": probe_param_set,
        "probe_execution_mode": str(cfg.probe_execution_mode),
    }
    return selection


def run_backtest_with_probe(df: pd.DataFrame, p: Params, cfg: RunConfig) -> Dict[str, object]:
    if cfg.box_source == "dynamic":
        main_cfg = clone_cfg(cfg, risk_high_candidates=tuple(), probe_days=0)
        res = run_backtest(df, p, main_cfg)
        res["selected_risk_high"] = np.nan
        res["probe_results"] = pd.DataFrame()
        res["stats"]["selected_risk_high"] = np.nan
        res["stats"]["probe_days"] = 0
        res["stats"]["probe_metric"] = str(cfg.probe_metric)
        res["stats"]["probe_end_exclusive"] = str(cfg.start_utc)
        res["stats"]["probe_metric_gap_to_2nd"] = np.nan
        res["stats"]["probe_tie_reason"] = "PROBE_SKIPPED_DYNAMIC_BOX"
        res["stats"]["probe_fallback_v2_used"] = False
        res["stats"]["probe_passed_filters_count"] = 0
        res["stats"]["probe_param_set"] = "DYNAMIC_BOX_NO_PROBE"
        res["stats"]["probe_execution_mode"] = str(cfg.probe_execution_mode)
        return res

    selection = run_probe_selection(df, cfg)
    selected_high = float(selection["selected_high"])
    probe_df = selection["probe_results"]
    probe_end_exclusive = selection["probe_end_exclusive"]
    main_start = probe_end_exclusive if (not probe_df.empty) else cfg.start_utc

    main_cfg = clone_cfg(
        cfg,
        start_utc=main_start,
        box_high=selected_high,
        risk_high_candidates=tuple(),
        probe_days=0,
    )
    res = run_backtest(df, p, main_cfg)
    res["selected_risk_high"] = selected_high
    res["probe_results"] = probe_df
    res["stats"]["selected_risk_high"] = selected_high
    res["stats"]["probe_days"] = int(selection["probe_days"])
    res["stats"]["probe_metric"] = str(selection["probe_metric"])
    res["stats"]["probe_end_exclusive"] = str(probe_end_exclusive)
    res["stats"]["probe_metric_gap_to_2nd"] = float(selection["metric_gap_to_2nd"]) if pd.notna(selection["metric_gap_to_2nd"]) else np.nan
    res["stats"]["probe_tie_reason"] = str(selection["tie_reason"])
    res["stats"]["probe_fallback_v2_used"] = bool(selection["fallback_v2_used"])
    res["stats"]["probe_passed_filters_count"] = int(selection["passed_filters_count"])
    res["stats"]["probe_param_set"] = str(selection["probe_param_set"])
    res["stats"]["probe_execution_mode"] = str(selection.get("probe_execution_mode", cfg.probe_execution_mode))
    return res


def write_visual_html(path: Path, bars: pd.DataFrame, trades: pd.DataFrame, eq: pd.DataFrame):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=("BTC 4H Manual Box Mean Reversion", "Equity"),
    )
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    if "risk_low_rt" in eq.columns and "risk_high_rt" in eq.columns:
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["risk_low_rt"], mode="lines", name="risk_low_rt", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["risk_high_rt"], mode="lines", name="risk_high_rt", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )
    elif "box_low_rt" in eq.columns and "box_high_rt" in eq.columns:
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["box_low_rt"], mode="lines", name="box_low_rt", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["box_high_rt"], mode="lines", name="box_high_rt", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=bars.index, y=[BOX_LOW] * len(bars), mode="lines", name="box_low", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=bars.index, y=[BOX_HIGH] * len(bars), mode="lines", name="box_high", line=dict(dash="dot", width=1)),
            row=1,
            col=1,
        )

    if not trades.empty:
        sl = trades.loc[trades["reason"] == "HARD_STOP"]
        b = trades.loc[(trades["action"] == "buy") & (trades["reason"] != "HARD_STOP")]
        s = trades.loc[(trades["action"] == "sell") & (trades["reason"] != "HARD_STOP")]
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
                ),
                row=1,
                col=1,
            )

    fig.add_trace(go.Scatter(x=eq.index, y=eq["equity"], mode="lines", name="Equity", line=dict(width=2, color="#111")), row=2, col=1)
    fig.update_layout(template="plotly_white", height=900, xaxis_rangeslider_visible=False)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def run_scan(
    df: pd.DataFrame,
    cfg: RunConfig,
    max_combos: int,
    scan_mode: str,
) -> Tuple[pd.DataFrame, Params, Dict[str, object], int]:
    combos, total_grid = build_param_grid(max_combos=max_combos, scan_mode=scan_mode, cfg=cfg)
    print(f"scan_mode={scan_mode}, grid_total={total_grid}, scanning={len(combos)}")
    use_fast_numba = _can_use_numba_fast_path(cfg)
    if use_fast_numba:
        print("numba fast path: ON (scan stats)")

    rows = []
    for i, p in enumerate(combos, start=1):
        if use_fast_numba:
            st = run_backtest_fast_stats(df, p, cfg)
        else:
            result = run_backtest_with_probe(df, p, cfg)
            st = dict(result["stats"])
        st.update(
            {
                "long_L": p.long_L,
                "short_L": 1.0 - p.long_L,
                "mid1": p.mid1,
                "mid2": p.mid2,
                "base_risk_pct": p.base_risk_pct,
                "atr_stop_mult": p.atr_stop_mult,
                "macro_lookback_bars": p.macro_lookback_bars,
                "layer_weights": str(p.layer_weights),
                "layer_step": p.layer_step,
                "max_leverage": p.max_leverage,
                "hard_stop_pct": p.hard_stop_pct,
                "cooldown_rearm_bars": p.cooldown_rearm_bars,
                "cooldown_stop_bars": p.cooldown_stop_bars,
                "time_stop_bars": p.time_stop_bars,
                "maker_ttl_bars": p.maker_ttl_bars,
                "maker_fallback_taker": p.maker_fallback_taker,
                "improve_mult": p.improve_mult,
                "min_improve_bps": p.min_improve_bps,
                "trade_box_mode": cfg.trade_box_mode_default,
                "trade_box_lookback": cfg.trade_box_lookback_default,
                "trade_box_q_low": cfg.trade_box_q_low_default,
                "trade_box_q_high": cfg.trade_box_q_high_default,
                "trade_box_ema_len": cfg.trade_box_ema_len_default,
                "trade_box_atr_mult": cfg.trade_box_atr_mult_default,
                "commission_pct": cfg.commission_pct,
                "slippage_pct": cfg.slippage_pct,
                "side_mode": cfg.side_mode,
                "entry_execution_mode": cfg.entry_execution_mode,
                "maker_fill_prob": cfg.maker_fill_prob,
                "maker_queue_delay_bars": cfg.maker_queue_delay_bars,
                "seed": cfg.seed,
                "regime_gate_mode": cfg.regime_gate_mode,
                "regime_gate_adx_thresh": cfg.regime_gate_adx_thresh,
                "regime_gate_bbwidth_min": cfg.regime_gate_bbwidth_min,
                "regime_gate_chop_thresh": cfg.regime_gate_chop_thresh,
                "circuit_breaker_mode": cfg.circuit_breaker_mode,
                "atr_stop_mode": cfg.atr_stop_mode,
                "atr_stop_mult": p.atr_stop_mult,
                "structural_cooldown_bars": cfg.structural_cooldown_bars,
                "enable_runner": cfg.enable_runner,
                "runner_pct": cfg.runner_pct,
                "runner_atr_mult": cfg.runner_atr_mult,
                "cooldown_cb_bars": cfg.cooldown_cb_bars,
                "degrade_on_trend": cfg.degrade_on_trend,
                "trend_slope_thresh": cfg.trend_slope_thresh,
                "atr_expand_thresh": cfg.atr_expand_thresh,
                "risk_box_low_input": cfg.box_low,
                "risk_box_high_input": cfg.box_high,
                "box_source": cfg.box_source,
                "macro_box_mode": cfg.macro_box_mode,
                "macro_lookback_bars": p.macro_lookback_bars,
                "macro_bb_len": cfg.macro_bb_len,
                "macro_bb_std": cfg.macro_bb_std,
            }
        )
        st["param_complexity"] = float(param_complexity_score(p))
        st["objective_reg"] = float(st["objective"]) - float(cfg.complexity_penalty_lambda) * float(st["param_complexity"])
        st["feasible"] = bool(
            (st["round_trips"] >= cfg.min_round_trips)
            and (st["profit_factor"] >= 1.15)
            and (st["hard_stop_ratio"] <= 0.45)
            and (st["entry_fill_rate"] >= 0.85)
        )
        st["rank_eligible"] = bool(st["round_trips"] >= cfg.min_round_trips)
        st["ranking_score"] = float(st["objective_reg"]) - (0.0 if bool(st.get("pf_reliable", True)) else float(cfg.pf_unreliable_penalty))
        rows.append(st)
        if i % 400 == 0 or i == len(combos):
            print(f"scan progress: {i}/{len(combos)}")

    all_df = pd.DataFrame(rows)
    all_df = all_df.sort_values(["rank_eligible", "feasible", "ranking_score", "return_pct"], ascending=[False, False, False, False]).reset_index(drop=True)
    if all_df.empty:
        raise RuntimeError("No scan output")

    top_rank = all_df.loc[all_df["rank_eligible"]]
    top_feasible = top_rank.loc[top_rank["feasible"]] if not top_rank.empty else pd.DataFrame()
    if not top_feasible.empty:
        best_row = top_feasible.iloc[0]
    elif not top_rank.empty:
        best_row = top_rank.iloc[0]
    else:
        best_row = all_df.iloc[0]
    best_params = Params(
        long_L=float(best_row["long_L"]),
        mid1=float(best_row["mid1"]),
        base_risk_pct=float(best_row["base_risk_pct"]),
        atr_stop_mult=float(best_row["atr_stop_mult"]),
        macro_lookback_bars=int(best_row["macro_lookback_bars"]),
        layer_weights_value=parse_layer_weights(str(best_row["layer_weights"])),
        max_leverage_value=float(best_row["max_leverage"]),
    )
    best_result = run_backtest_with_probe(df, best_params, cfg)
    return all_df, best_params, best_result, total_grid


def pick_best_row(all_df: pd.DataFrame) -> pd.Series:
    top_rank = all_df.loc[all_df["rank_eligible"]] if "rank_eligible" in all_df.columns else all_df
    top_feasible = top_rank.loc[top_rank["feasible"]] if ("feasible" in top_rank.columns and not top_rank.empty) else pd.DataFrame()
    if not top_feasible.empty:
        return top_feasible.iloc[0]
    if not top_rank.empty:
        return top_rank.iloc[0]
    return all_df.iloc[0]


def resolve_robust_params(best: Params, args: argparse.Namespace) -> Params:
    if args.robust_long_L is None:
        return best
    return Params(
        long_L=float(args.robust_long_L),
        mid1=float(args.robust_mid1),
        base_risk_pct=float(args.robust_base_risk_pct),
        atr_stop_mult=float(args.robust_hard_stop_pct),
        macro_lookback_bars=max(int(args.robust_trade_box_lookback), 2),
        layer_weights_value=parse_layer_weights(str(args.robust_layer_weights)),
        max_leverage_value=float(args.robust_max_leverage),
    )


def run_robustness_test(df: pd.DataFrame, cfg: RunConfig, p: Params, out_path: Path) -> pd.DataFrame:
    risk_cases: List[Tuple[float, float, str]] = []
    for hh in [90000.0, 92000.0, 94000.0, 96000.0]:
        risk_cases.append((80000.0, hh, f"risk_low80000_high{int(hh)}"))
    for ll in [78000.0, 79000.0, 80000.0, 81000.0]:
        risk_cases.append((ll, 94000.0, f"risk_low{int(ll)}_high94000"))

    rows = []
    for low_b, high_b, name in risk_cases:
        if high_b <= low_b:
            continue
        case_cfg = RunConfig(
            start_utc=cfg.start_utc,
            end_utc=cfg.end_utc,
            commission_pct=cfg.commission_pct,
            slippage_pct=cfg.slippage_pct,
            box_low=low_b,
            box_high=high_b,
            dyn_cfg=cfg.dyn_cfg,
            side_mode=cfg.side_mode,
            clip_dynamic_to_manual=cfg.clip_dynamic_to_manual,
            trade_box_mode_default=cfg.trade_box_mode_default,
            trade_box_lookback_default=cfg.trade_box_lookback_default,
            trade_box_q_low_default=cfg.trade_box_q_low_default,
            trade_box_q_high_default=cfg.trade_box_q_high_default,
            trade_box_ema_len_default=cfg.trade_box_ema_len_default,
            trade_box_atr_mult_default=cfg.trade_box_atr_mult_default,
            risk_high_candidates=tuple(),
            probe_days=0,
            probe_metric=cfg.probe_metric,
            probe_execution_mode=cfg.probe_execution_mode,
            fallback_no_signal_mode=cfg.fallback_no_signal_mode,
            probe_tie_eps=cfg.probe_tie_eps,
            probe_min_round_trips=cfg.probe_min_round_trips,
            probe_min_entry_fill_rate=cfg.probe_min_entry_fill_rate,
            probe_max_hard_stop_ratio=cfg.probe_max_hard_stop_ratio,
            early_stop_first_k_trades=cfg.early_stop_first_k_trades,
            early_stop_hard_stop_threshold=cfg.early_stop_hard_stop_threshold,
            early_stop_first_m_bars=cfg.early_stop_first_m_bars,
            enable_early_stop=cfg.enable_early_stop,
            entry_execution_mode=cfg.entry_execution_mode,
            maker_fill_prob=cfg.maker_fill_prob,
            maker_queue_delay_bars=cfg.maker_queue_delay_bars,
            seed=cfg.seed,
            min_round_trips=cfg.min_round_trips,
            pf_unreliable_penalty=cfg.pf_unreliable_penalty,
            degrade_on_trend=cfg.degrade_on_trend,
            trend_slope_thresh=cfg.trend_slope_thresh,
            atr_expand_thresh=cfg.atr_expand_thresh,
            degrade_risk_lookback_bars=cfg.degrade_risk_lookback_bars,
            enable_runner=cfg.enable_runner,
            runner_pct=cfg.runner_pct,
            runner_atr_mult=cfg.runner_atr_mult,
        )
        res = run_backtest(df, p, case_cfg)
        st = res["stats"]
        rows.append(
            {
                "robust_group": "risk_box_perturb",
                "case": name,
                "risk_box_low": low_b,
                "risk_box_high": high_b,
                "trade_box_mode": p.trade_box_mode,
                "trade_box_lookback": p.trade_box_lookback,
                "return_pct": st["return_pct"],
                "max_dd_pct": st["max_dd_pct"],
                "profit_factor": st["profit_factor"],
                "round_trips": st["round_trips"],
                "hard_stop_ratio": st["hard_stop_ratio"],
                "box_invalid_count": st["box_invalid_count"],
                "risk_expand_count": st["risk_expand_count"],
                "trade_box_invalid_ratio": st["trade_box_invalid_ratio"],
                "objective": st["objective"],
            }
        )

    for mode in ["rolling_quantile", "rolling_hilo", "ema_atr"]:
        p_case = Params(
            long_L=p.long_L,
            mid1=p.mid1,
            base_risk_pct=p.base_risk_pct,
            atr_stop_mult=p.atr_stop_mult,
            macro_lookback_bars=p.macro_lookback_bars,
        )
        cfg_case = clone_cfg(cfg, trade_box_mode_default=mode)
        res = run_backtest(df, p_case, cfg_case)
        st = res["stats"]
        rows.append(
            {
                "robust_group": "trade_box_shape",
                "case": f"trade_{mode}",
                "risk_box_low": cfg.box_low,
                "risk_box_high": cfg.box_high,
                "trade_box_mode": mode,
                "trade_box_lookback": cfg_case.trade_box_lookback_default,
                "return_pct": st["return_pct"],
                "max_dd_pct": st["max_dd_pct"],
                "profit_factor": st["profit_factor"],
                "round_trips": st["round_trips"],
                "hard_stop_ratio": st["hard_stop_ratio"],
                "box_invalid_count": st["box_invalid_count"],
                "risk_expand_count": st["risk_expand_count"],
                "trade_box_invalid_ratio": st["trade_box_invalid_ratio"],
                "objective": st["objective"],
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    return out_df


def run_baseline_suite_mode(
    df: pd.DataFrame,
    cfg: RunConfig,
    args: argparse.Namespace,
    csv_path: Path,
    out_dir: Path,
) -> Dict[str, Path]:
    baseline_cfg = clone_cfg(
        cfg,
        start_utc=parse_ts_utc("2025-11-21 00:00:00+00:00"),
        end_utc=parse_ts_utc("2026-01-14 00:00:00+00:00"),
        box_low=80000.0,
        box_high=94000.0,
        risk_high_candidates=(92000.0, 93000.0, 94000.0, 95000.0, 96000.0),
        probe_days=3,
        side_mode="both",
        degrade_on_trend=False,
    )
    baseline_params = Params(
        long_L=0.36,
        mid1=0.60,
        base_risk_pct=0.45,
        atr_stop_mult=1.5,
        macro_lookback_bars=84,
    )
    res = run_backtest_with_probe(df, baseline_params, baseline_cfg)
    prefix = f"manual_box_{args.round_tag}_baseline"
    out_trades = out_dir / f"{prefix}_best_trades.csv"
    out_equity = out_dir / f"{prefix}_best_equity.csv"
    out_html = out_dir / f"{prefix}_best_visual.html"
    out_report = out_dir / f"{prefix}_report.md"
    out_probe = out_dir / f"{prefix}_probe_results.csv"
    out_validation = out_dir / f"{prefix}_probe_stability_summary.csv"
    out_walk = out_dir / f"{prefix}_walk_forward_summary.csv"
    out_walk_agg = out_dir / f"{prefix}_walk_forward_aggregate.md"

    res["trades"].to_csv(out_trades, index=False)
    res["equity"].reset_index().to_csv(out_equity, index=False)
    write_visual_html(out_html, res["bars"], res["trades"], res["equity"])
    if isinstance(res.get("probe_results"), pd.DataFrame) and (not res["probe_results"].empty):
        res["probe_results"].to_csv(out_probe, index=False)
    val_df = run_validation_suite(df, baseline_cfg, baseline_params, out_validation)
    walk_df = run_walk_forward(
        df=df,
        cfg=baseline_cfg,
        p=baseline_params,
        segment_days=max(int(args.segment_days), 1),
        step_days=max(int(args.step_days), 1),
        out_path=out_walk,
        max_combos=max(int(args.max_combos), 1),
        scan_mode=str(args.scan_mode),
    )
    walk_artifacts = dict(walk_df.attrs.get("wfa_artifacts", {}))
    walk_agg = write_walk_forward_aggregate(out_walk_agg, walk_df, walk_artifacts)
    gate_reasons: List[str] = []
    if walk_agg["early_stop_rate"] > float(args.release_gate_max_early_stop_rate):
        gate_reasons.append("early_stop_rate")
    if walk_agg["median_return"] <= float(args.release_gate_min_median_return):
        gate_reasons.append("median_return")
    if (not np.isfinite(walk_agg["median_pf"])) or (walk_agg["median_pf"] < float(args.release_gate_min_median_pf)):
        gate_reasons.append("median_pf")
    gate_status = "NOT_ROBUST" if gate_reasons else "ROBUST"
    fallback_reason_dist: Dict[str, int] = {}
    if isinstance(res.get("probe_results"), pd.DataFrame) and (not res["probe_results"].empty):
        rr = res["probe_results"]["tie_reason"].fillna("")
        rr = rr.loc[rr != ""]
        fallback_reason_dist = {str(k): int(v) for k, v in rr.value_counts().to_dict().items()}
    with out_report.open("w", encoding="utf-8") as f:
        f.write(f"# {prefix} Report\n\n")
        f.write("## Setup\n")
        f.write(f"- version: `{__version__}`\n")
        f.write(f"- git_hash: `{__git_hash__}`\n")
        f.write(f"- run_tag: `{args.round_tag}`\n")
        f.write(f"- csv: `{csv_path}`\n")
        f.write(f"- data_window_utc: `{baseline_cfg.start_utc}` -> `{baseline_cfg.end_utc}`\n")
        f.write(f"- box_source: `{baseline_cfg.box_source}`\n")
        f.write(f"- macro_box_mode: `{baseline_cfg.macro_box_mode}`\n")
        f.write(f"- macro_lookback_bars: `{baseline_cfg.macro_lookback_bars}`\n")
        f.write(f"- box_low: `{baseline_cfg.box_low}`\n")
        f.write(f"- box_high: `{baseline_cfg.box_high}`\n")
        f.write(f"- candidate_highs: `{baseline_cfg.risk_high_candidates}`\n")
        f.write(f"- probe_days: `{baseline_cfg.probe_days}`\n")
        f.write(f"- fallback_reason_distribution: `{fallback_reason_dist}`\n")
        f.write("\n## Params Summary\n")
        f.write(f"- summary: `{baseline_params.__dict__}`\n")
        f.write("\n## Metrics\n")
        for k in ["return_pct", "max_dd_pct", "profit_factor", "round_trips", "hard_stop_ratio", "objective", "selected_risk_high", "probe_tie_reason"]:
            v = res["stats"].get(k)
            if isinstance(v, float):
                f.write(f"- {k}: `{v:.6f}`\n")
            else:
                f.write(f"- {k}: `{v}`\n")
        f.write("\n## Release Gate\n")
        f.write(f"- status: `{gate_status}`\n")
        f.write(f"- reasons: `{gate_reasons}`\n")
        f.write(f"- median_return: `{walk_agg['median_return']:.6f}`\n")
        f.write(f"- early_stop_rate: `{walk_agg['early_stop_rate']:.6f}`\n")
        f.write(f"- median_pf: `{walk_agg['median_pf']:.6f}`\n")

    files = [Path("manual_box_roundX.py"), out_trades, out_equity, out_html, out_report, out_probe, out_validation, out_walk, out_walk_agg]
    for key in ["combined_equity_path", "combined_trades_path", "combined_bars_path", "combined_visual_path"]:
        if walk_artifacts.get(key):
            files.append(Path(str(walk_artifacts[key])))
    manifest = build_archive_manifest(
        args=args,
        run_tag=f"{args.round_tag}_baseline",
        cfg=baseline_cfg,
        best_params=baseline_params,
        best_stats=res["stats"],
        release_gate_status=gate_status,
        release_gate_reasons=gate_reasons,
        files=files,
    )
    out_zip = create_archive_zip(out_dir, f"{args.round_tag}_baseline", manifest, files)
    return {
        "trades": out_trades,
        "equity": out_equity,
        "html": out_html,
        "report": out_report,
        "probe": out_probe,
        "validation": out_validation,
        "walk": out_walk,
        "walk_agg": out_walk_agg,
        "zip": out_zip,
    }


def run_validation_suite(df: pd.DataFrame, cfg: RunConfig, p: Params, out_path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total_days = max((cfg.end_utc - cfg.start_utc).total_seconds() / 86400.0, 0.0)
    for probe_days in [3, 5, 7]:
        probe_days_use = min(int(probe_days), max(int(total_days) - 1, 0))
        vcfg = clone_cfg(
            cfg,
            probe_days=int(probe_days_use),
            enable_early_stop=bool(len(cfg.risk_high_candidates) > 0 and probe_days_use > 0),
        )
        res = run_backtest_with_probe(df, p, vcfg)
        st = res["stats"]
        rows.append(
            {
                "probe_days": probe_days_use,
                "selected_high": float(st.get("selected_risk_high", cfg.box_high)),
                "gap": float(st.get("probe_metric_gap_to_2nd", np.nan)),
                "main_return": float(st["return_pct"]),
                "main_max_dd": float(st["max_dd_pct"]),
                "main_objective": float(st["objective"]),
                "early_stop_flag": bool(st.get("early_stop_due_to_bad_probe", False)),
                "probe_fallback_v2_used": bool(st.get("probe_fallback_v2_used", False)),
                "probe_tie_reason": str(st.get("probe_tie_reason", "")),
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    return out_df


def run_walk_forward(
    df: pd.DataFrame,
    cfg: RunConfig,
    p: Params,
    segment_days: int,
    step_days: int,
    out_path: Path,
    max_combos: int,
    scan_mode: str,
) -> pd.DataFrame:
    seg_days = max(int(segment_days), 1)
    step = max(int(step_days), 1)
    rows: List[Dict[str, object]] = []
    combined_eq_parts: List[pd.DataFrame] = []
    combined_trade_parts: List[pd.DataFrame] = []
    combined_bar_parts: List[pd.DataFrame] = []
    prev_segment_best_params: Optional[Params] = None
    chain_equity = float(cfg.initial_equity)
    history_bars = max(required_history_bars(cfg, p), max(MACRO_LOOKBACK_GRID), max(MACRO_LOOKBACK_LOCAL))
    fold_anchor = cfg.start_utc + pd.Timedelta(days=seg_days)
    seg_idx = 0
    wfa_paths = derive_wfa_output_paths(out_path)

    while fold_anchor < cfg.end_utc:
        is_start = fold_anchor - pd.Timedelta(days=seg_days)
        is_end_exclusive = fold_anchor
        oos_start = fold_anchor
        oos_end_exclusive = min(fold_anchor + pd.Timedelta(days=step), cfg.end_utc + pd.Timedelta(nanoseconds=1))
        if oos_end_exclusive <= oos_start:
            break

        is_end_inclusive = is_end_exclusive - pd.Timedelta(nanoseconds=1)
        oos_end_inclusive = oos_end_exclusive - pd.Timedelta(nanoseconds=1)
        if df.loc[(df.index >= oos_start) & (df.index <= oos_end_inclusive)].empty:
            break

        adaptive_probe_days = min(5, max(3, int(seg_days // 5)))
        oos_total_days = max((oos_end_exclusive - oos_start).total_seconds() / 86400.0, 0.0)
        seg_probe_days = min(int(adaptive_probe_days), max(int(oos_total_days) - 1, 0))

        is_cfg = clone_cfg(
            cfg,
            start_utc=is_start,
            end_utc=is_end_inclusive,
            history_start_utc=resolve_history_start_utc(df, is_start, history_bars),
            probe_days=seg_probe_days,
            enable_early_stop=False,
            initial_equity=10000.0,
        )

        scan_error = ""
        is_scan_df = pd.DataFrame()
        local_best_params = p
        local_best_stats: Dict[str, object] = {}
        total_grid = 0
        is_max_return = np.nan
        is_best_pf = np.nan
        try:
            is_scan_df, local_best_params, local_best_result, total_grid = run_scan(
                df=df,
                cfg=is_cfg,
                max_combos=max_combos,
                scan_mode=scan_mode,
            )
            local_best_stats = dict(local_best_result["stats"])
            if not is_scan_df.empty:
                is_max_return = float(is_scan_df["return_pct"].max())
                is_best_pf = float(pick_best_row(is_scan_df).get("profit_factor", np.nan))
        except Exception as exc:  # noqa: BLE001
            scan_error = str(exc)

        if scan_error:
            if prev_segment_best_params is not None:
                used_params = prev_segment_best_params
                params_source = "FALLBACK_PREV_SEGMENT_SCAN_ERROR"
            else:
                used_params = local_best_params
                params_source = "FIRST_SEGMENT_SCAN_ERROR_USE_DEFAULT"
        else:
            used_params = local_best_params
            params_source = "IS_SCAN_LOCAL_BEST"

        if not scan_error:
            prev_segment_best_params = local_best_params

        oos_cfg = clone_cfg(
            cfg,
            start_utc=oos_start,
            end_utc=oos_end_inclusive,
            history_start_utc=resolve_history_start_utc(df, oos_start, history_bars),
            probe_days=seg_probe_days,
            enable_early_stop=bool(len(cfg.risk_high_candidates) > 0 and seg_probe_days > 0),
            initial_equity=chain_equity,
        )
        oos_res = run_backtest_with_probe(df, used_params, oos_cfg)
        st = oos_res["stats"]
        tie_reason = str(st.get("probe_tie_reason", ""))
        fallback_selected = tie_reason.startswith("FALLBACK")
        selected_source = "fallback" if fallback_selected else ("probe" if tie_reason not in {"", "PROBE_DISABLED"} else "probe_disabled")

        eq_part = oos_res["equity"].reset_index().copy()
        if not eq_part.empty:
            eq_part["segment_idx"] = seg_idx
            eq_part["is_start_utc"] = str(is_start)
            eq_part["is_end_utc"] = str(is_end_inclusive)
            eq_part["oos_start_utc"] = str(oos_start)
            eq_part["oos_end_utc"] = str(oos_end_inclusive)
            eq_part["params_source"] = params_source
            combined_eq_parts.append(eq_part)

        trades_part = oos_res["trades"].copy()
        if not trades_part.empty:
            trades_part["segment_idx"] = seg_idx
            trades_part["is_start_utc"] = str(is_start)
            trades_part["is_end_utc"] = str(is_end_inclusive)
            trades_part["oos_start_utc"] = str(oos_start)
            trades_part["oos_end_utc"] = str(oos_end_inclusive)
            trades_part["params_source"] = params_source
            trades_part["cash_after_chained"] = trades_part["cash_after"]
            combined_trade_parts.append(trades_part)

        bars_part = oos_res["bars"].reset_index().copy()
        if not bars_part.empty:
            bars_part["segment_idx"] = seg_idx
            combined_bar_parts.append(bars_part)

        chain_equity = float(st.get("final_equity", chain_equity))
        rows.append(
            {
                "segment_idx": seg_idx,
                "is_start_utc": str(is_start),
                "is_end_utc": str(is_end_inclusive),
                "oos_start_utc": str(oos_start),
                "oos_end_utc": str(oos_end_inclusive),
                "segment_days": seg_days,
                "step_days": step,
                "probe_days_used": seg_probe_days,
                "params_source": params_source,
                "scan_error": scan_error,
                "is_scan_total_grid": int(total_grid),
                "is_scan_rows": int(len(is_scan_df)),
                "is_max_return_pct": float(is_max_return) if np.isfinite(is_max_return) else np.nan,
                "is_best_profit_factor": float(is_best_pf) if np.isfinite(is_best_pf) else np.nan,
                "is_best_return_pct": float(local_best_stats.get("return_pct", np.nan)),
                "is_best_objective": float(local_best_stats.get("objective", np.nan)),
                "used_long_L": float(used_params.long_L),
                "used_mid1": float(used_params.mid1),
                "used_mid2": float(used_params.mid2),
                "used_base_risk_pct": float(used_params.base_risk_pct),
                "used_atr_stop_mult": float(used_params.atr_stop_mult),
                "used_macro_lookback_bars": int(used_params.macro_lookback_bars),
                "selected_high": float(st.get("selected_risk_high", oos_cfg.box_high)),
                "gap": float(st.get("probe_metric_gap_to_2nd", np.nan)),
                "return_pct": float(st["return_pct"]),
                "max_dd_pct": float(st["max_dd_pct"]),
                "profit_factor": float(st["profit_factor"]),
                "pf_reliable": bool(st.get("pf_reliable", True)),
                "orders_total": int(st.get("orders_total", 0)),
                "round_trips": int(st["round_trips"]),
                "hard_stop_ratio": float(st["hard_stop_ratio"]),
                "objective": float(st["objective"]),
                "early_stop_flag": bool(st.get("early_stop_due_to_bad_probe", False)),
                "probe_fallback_v2_used": bool(st.get("probe_fallback_v2_used", False)),
                "probe_tie_reason": tie_reason,
                "selected_source": selected_source,
                "oos_initial_equity": float(oos_cfg.initial_equity),
                "oos_final_equity": float(st.get("final_equity", np.nan)),
            }
        )
        print(
            f"wfa fold {seg_idx}: IS {is_start} -> {is_end_inclusive}, OOS {oos_start} -> {oos_end_inclusive}, "
            f"source={params_source}, oos_return={float(st['return_pct']):.4f}%"
        )
        seg_idx += 1
        fold_anchor = fold_anchor + pd.Timedelta(days=step)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)

    combined_equity_path = None
    combined_trades_path = None
    combined_bars_path = None
    combined_visual_path = None
    combined_final_equity = float(cfg.initial_equity)
    combined_return_pct = float("nan")

    if combined_eq_parts:
        combined_eq_df = pd.concat(combined_eq_parts, ignore_index=True)
        combined_eq_df = combined_eq_df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
        combined_equity_path = wfa_paths["equity"]
        combined_eq_df.to_csv(combined_equity_path, index=False)
        combined_final_equity = float(combined_eq_df["equity"].iloc[-1])
        combined_return_pct = (combined_final_equity / max(float(cfg.initial_equity), 1e-12) - 1.0) * 100.0
    else:
        combined_eq_df = pd.DataFrame()

    if combined_trade_parts:
        combined_trades_df = pd.concat(combined_trade_parts, ignore_index=True)
        combined_trades_df = combined_trades_df.sort_values("time").reset_index(drop=True)
        combined_trades_path = wfa_paths["trades"]
        combined_trades_df.to_csv(combined_trades_path, index=False)
    else:
        combined_trades_df = pd.DataFrame()

    if combined_bar_parts:
        combined_bars_df = pd.concat(combined_bar_parts, ignore_index=True)
        combined_bars_df = combined_bars_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        combined_bars_df = combined_bars_df.rename(columns={"timestamp": "time"})
        combined_bars_path = wfa_paths["bars"]
        combined_bars_df.to_csv(combined_bars_path, index=False)
    else:
        combined_bars_df = pd.DataFrame()

    if (not combined_eq_df.empty) and (not combined_bars_df.empty):
        combined_visual_path = wfa_paths["visual"]
        write_visual_html(
            combined_visual_path,
            combined_bars_df.set_index("time"),
            combined_trades_df if not combined_trades_df.empty else pd.DataFrame(columns=["time", "action", "reason", "exec_price"]),
            combined_eq_df.set_index("time"),
        )

    out_df.attrs["wfa_artifacts"] = {
        "combined_equity_path": str(combined_equity_path) if combined_equity_path is not None else "",
        "combined_trades_path": str(combined_trades_path) if combined_trades_path is not None else "",
        "combined_bars_path": str(combined_bars_path) if combined_bars_path is not None else "",
        "combined_visual_path": str(combined_visual_path) if combined_visual_path is not None else "",
        "combined_final_equity": float(combined_final_equity),
        "combined_return_pct": float(combined_return_pct) if pd.notna(combined_return_pct) else np.nan,
        "combined_segments": int(len(out_df)),
    }
    return out_df


def write_walk_forward_aggregate(path: Path, walk_df: pd.DataFrame, artifacts: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    artifacts = artifacts or dict(walk_df.attrs.get("wfa_artifacts", {}))
    if walk_df.empty:
        summary = {
            "mean_return": 0.0,
            "median_return": 0.0,
            "mean_max_dd": 0.0,
            "median_max_dd": 0.0,
            "mean_pf": 0.0,
            "median_pf": 0.0,
            "pf_reliable_rate": 0.0,
            "win_rate": 0.0,
            "early_stop_rate": 0.0,
        }
        with path.open("w", encoding="utf-8") as f:
            f.write("# Walk Forward Aggregate\n\n")
            f.write("No walk-forward rows.\n")
        return summary

    mean_return = float(walk_df["return_pct"].mean())
    median_return = float(walk_df["return_pct"].median())
    mean_dd = float(walk_df["max_dd_pct"].mean())
    median_dd = float(walk_df["max_dd_pct"].median())
    pf_reliable_mask = walk_df["pf_reliable"].astype(bool) if "pf_reliable" in walk_df.columns else pd.Series([True] * len(walk_df))
    pf_reliable_rate = float(pf_reliable_mask.mean()) if len(pf_reliable_mask) > 0 else 0.0
    pf_ser = walk_df.loc[pf_reliable_mask, "profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    mean_pf = float(pf_ser.mean()) if not pf_ser.empty else float("nan")
    median_pf = float(pf_ser.median()) if not pf_ser.empty else float("nan")
    win_rate = float((walk_df["return_pct"] > 0.0).mean())
    early_stop_rate = float(walk_df["early_stop_flag"].astype(bool).mean())
    pf_q25 = float(pf_ser.quantile(0.25)) if not pf_ser.empty else float("nan")
    pf_q75 = float(pf_ser.quantile(0.75)) if not pf_ser.empty else float("nan")
    if "selected_source" in walk_df.columns:
        sel_src = walk_df["selected_source"].astype(str)
        selected_probe = walk_df.loc[sel_src.eq("probe"), "selected_high"]
        selected_fallback = walk_df.loc[sel_src.eq("fallback"), "selected_high"]
    else:
        selected_probe = pd.Series([], dtype=float)
        selected_fallback = pd.Series([], dtype=float)
    selected_probe_dist = selected_probe.value_counts(normalize=True).sort_index()
    selected_fallback_dist = selected_fallback.value_counts(normalize=True).sort_index()
    if "selected_source" in walk_df.columns and "probe_tie_reason" in walk_df.columns:
        fallback_reason_counts = walk_df.loc[walk_df["selected_source"].astype(str).eq("fallback"), "probe_tie_reason"].fillna("").value_counts().to_dict()
    else:
        fallback_reason_counts = {}

    with path.open("w", encoding="utf-8") as f:
        f.write("# Walk Forward Aggregate\n\n")
        f.write("## Summary\n")
        f.write(f"- mean_return_pct: `{mean_return:.6f}`\n")
        f.write(f"- median_return_pct: `{median_return:.6f}`\n")
        f.write(f"- mean_max_dd_pct: `{mean_dd:.6f}`\n")
        f.write(f"- median_max_dd_pct: `{median_dd:.6f}`\n")
        f.write(f"- mean_profit_factor: `{mean_pf:.6f}`\n")
        f.write(f"- median_profit_factor: `{median_pf:.6f}`\n")
        f.write(f"- pf_reliable_rate: `{pf_reliable_rate:.6f}`\n")
        f.write(f"- win_rate: `{win_rate:.6f}`\n")
        f.write(f"- early_stop_rate: `{early_stop_rate:.6f}`\n")
        f.write(f"- pf_q25: `{pf_q25:.6f}`\n")
        f.write(f"- pf_q75: `{pf_q75:.6f}`\n")
        if artifacts:
            f.write("\n## Stitched OOS\n")
            if artifacts.get("combined_return_pct") is not None and pd.notna(artifacts.get("combined_return_pct")):
                f.write(f"- combined_return_pct: `{float(artifacts['combined_return_pct']):.6f}`\n")
            if artifacts.get("combined_final_equity") is not None:
                f.write(f"- combined_final_equity: `{float(artifacts['combined_final_equity']):.6f}`\n")
            for key in ["combined_equity_path", "combined_trades_path", "combined_bars_path", "combined_visual_path"]:
                if artifacts.get(key):
                    f.write(f"- {key}: `{artifacts[key]}`\n")
        f.write("\n## Selected High Distribution\n")
        f.write("- selected_by_probe:\n")
        if not selected_probe_dist.empty:
            f.write(selected_probe_dist.rename("weight").to_frame().to_markdown())
            f.write("\n")
        else:
            f.write("none\n")
        f.write("- selected_by_fallback:\n")
        if not selected_fallback_dist.empty:
            f.write(selected_fallback_dist.rename("weight").to_frame().to_markdown())
            f.write("\n")
        else:
            f.write("none\n")
        f.write(f"- fallback_reason_counts: `{fallback_reason_counts}`\n")

    return {
        "mean_return": mean_return,
        "median_return": median_return,
        "mean_max_dd": mean_dd,
        "median_max_dd": median_dd,
        "mean_pf": mean_pf,
        "median_pf": median_pf,
        "pf_reliable_rate": pf_reliable_rate,
        "win_rate": win_rate,
        "early_stop_rate": early_stop_rate,
        "pf_q25": pf_q25,
        "pf_q75": pf_q75,
        "combined_return_pct": float(artifacts["combined_return_pct"]) if artifacts and pd.notna(artifacts.get("combined_return_pct")) else float("nan"),
        "combined_final_equity": float(artifacts["combined_final_equity"]) if artifacts and artifacts.get("combined_final_equity") is not None else float("nan"),
    }


def _max_probe_days_for_window(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> int:
    if end_utc <= start_utc:
        return 0
    max_days = int((end_utc - start_utc) / pd.Timedelta(days=1))
    while max_days > 0 and (start_utc + pd.Timedelta(days=max_days) >= end_utc):
        max_days -= 1
    return max(max_days, 0)


def _adaptive_walk_forward_schedule(box_days: float) -> Tuple[Optional[int], Optional[int], str]:
    if box_days >= 28.0:
        return 14, 4, ""
    if box_days >= 14.0:
        return 7, 3, ""
    return None, None, "BOX_TOO_SHORT_FOR_WF"


def _main_window_qualification(stats: Dict[str, object], min_round_trips_box: int, main_bars: int) -> Tuple[int, int, bool, bool, str]:
    orders_total = int(stats.get("orders_total", 0))
    round_trips = int(stats.get("round_trips", 0))
    filled_entries = int(stats.get("filled_entries", 0))
    main_no_trade = bool((orders_total == 0) and (round_trips == 0) and (filled_entries == 0))
    main_rank_eligible = bool(round_trips >= int(min_round_trips_box))
    if main_bars < 3:
        reason = "BOX_TOO_SHORT"
    elif main_no_trade:
        reason = "NO_TRADES"
    elif not main_rank_eligible:
        reason = "INSUFFICIENT_TRIPS"
    else:
        reason = ""
    return orders_total, round_trips, main_rank_eligible, main_no_trade, reason


def run_batch_box_eval(df: pd.DataFrame, cfg: RunConfig, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    boxes_path = Path(args.boxes_csv)
    if not boxes_path.exists():
        raise FileNotFoundError(f"boxes csv not found: {boxes_path}")

    boxes_raw = pd.read_csv(boxes_path)
    if boxes_raw.empty:
        raise ValueError(f"boxes csv is empty: {boxes_path}")

    c_start = infer_column(boxes_raw, ["startts", "start", "starttime"])
    c_end = infer_column(boxes_raw, ["endts", "end", "endtime"])
    c_low = None
    c_high = None
    try:
        c_low = infer_column(boxes_raw, ["boxlow", "low"])
        c_high = infer_column(boxes_raw, ["boxhigh", "high"])
    except Exception:  # noqa: BLE001
        if cfg.box_source != "dynamic":
            raise
    try:
        c_id = infer_column(boxes_raw, ["boxid", "id", "name"])
    except Exception:  # noqa: BLE001
        c_id = None
    try:
        c_cands = infer_column(boxes_raw, ["riskhighcandidates", "candidates", "candidatehighs"])
    except Exception:  # noqa: BLE001
        c_cands = None

    rows: List[Dict[str, object]] = []
    summary_path = out_dir / str(args.batch_summary_name)
    print(f"batch-box-eval: boxes={len(boxes_raw)}, summary={summary_path}")

    def _main_diag_fields(st: Dict[str, object], prefix: str) -> Dict[str, object]:
        return {
            f"main_filled_entries_{prefix}": int(st.get("filled_entries", 0)),
            f"main_missed_entry_{prefix}": int(st.get("missed_entry", 0)),
            f"main_entry_fill_rate_{prefix}": float(st.get("entry_fill_rate", np.nan)),
            f"main_trade_box_invalid_ratio_{prefix}": float(st.get("trade_box_invalid_ratio", np.nan)),
            f"main_box_invalid_count_{prefix}": int(st.get("box_invalid_count", 0)),
            f"main_risk_expand_count_{prefix}": int(st.get("risk_expand_count", 0)),
            f"main_orders_total_{prefix}": int(st.get("orders_total", 0)),
            f"main_round_trips_{prefix}": int(st.get("round_trips", 0)),
            f"main_hard_stop_count_{prefix}": int(st.get("hard_stop_count", 0)),
            f"main_atr_stop_count_{prefix}": int(st.get("atr_stop_count", 0)),
            f"main_time_stop_count_{prefix}": int(st.get("time_stop_count", 0)),
            f"main_runner_stop_count_{prefix}": int(st.get("runner_stop_count", 0)),
            f"main_avg_hold_bars_{prefix}": float(st.get("avg_hold_bars", np.nan)),
            f"main_avg_layers_{prefix}": float(st.get("avg_layers", np.nan)),
            f"main_layer_add_count_{prefix}": int(st.get("layer_add_count", 0)),
            f"main_short_trade_count_{prefix}": int(st.get("short_trade_count", 0)),
            f"main_long_trade_count_{prefix}": int(st.get("long_trade_count", 0)),
            f"main_start_gate_pass_rate_{prefix}": float(st.get("start_gate_pass_rate", np.nan)),
            f"main_regime_start_gate_pass_rate_{prefix}": float(st.get("regime_start_gate_pass_rate", np.nan)),
            f"main_perf_stop_count_{prefix}": int(st.get("perf_stop_count", 0)),
            f"main_cb_trigger_count_{prefix}": int(st.get("cb_trigger_count", 0)),
            f"main_cb_disable_bars_ratio_{prefix}": float(st.get("cb_disable_bars_ratio", 0.0)),
            f"main_signal_count_total_{prefix}": int(st.get("signal_count_total", 0)),
            f"main_signal_count_long_{prefix}": int(st.get("signal_count_long", 0)),
            f"main_signal_count_short_{prefix}": int(st.get("signal_count_short", 0)),
            f"short_enabled_ratio_{prefix}": float(st.get("short_enabled_ratio", np.nan)),
            f"main_max_margin_utilization_{prefix}": float(st.get("max_margin_utilization", np.nan)),
        }

    for ridx, row in boxes_raw.iterrows():
        box_id = str(row[c_id]).strip() if c_id is not None else f"box_{ridx + 1:03d}"
        if not box_id:
            box_id = f"box_{ridx + 1:03d}"
        safe_box_id = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in box_id)
        try:
            start_utc = parse_ts_utc(str(row[c_start]))
            end_utc = parse_ts_utc(str(row[c_end]))
            if c_low is not None and c_high is not None:
                box_low = float(row[c_low])
                box_high = float(row[c_high])
            else:
                # Dynamic box mode can run with time-only rows; manual box fields become placeholders.
                box_low = float(cfg.box_low)
                box_high = float(cfg.box_high)
            if end_utc <= start_utc:
                raise ValueError("end_ts must be after start_ts")
            if (cfg.box_source != "dynamic") and (box_high <= box_low):
                raise ValueError("box_high must be > box_low")

            risk_high_candidates = parse_float_list(str(row[c_cands])) if c_cands is not None else tuple()
            box_days = float((end_utc - start_utc).total_seconds() / 86400.0)
            box_bars = int(df.loc[(df.index >= start_utc) & (df.index <= end_utc)].shape[0])
            min_round_trips_box = max(1, int(np.floor(box_bars / 40.0)))

            probe_days_cap = _max_probe_days_for_window(start_utc, end_utc)
            probe_days_use = min(int(cfg.probe_days), int(probe_days_cap))
            enable_early_stop = bool(len(risk_high_candidates) > 0 and probe_days_use > 0)

            case_cfg_base = clone_cfg(
                cfg,
                start_utc=start_utc,
                end_utc=end_utc,
                box_low=box_low,
                box_high=box_high,
                risk_high_candidates=risk_high_candidates,
                probe_days=probe_days_use,
                enable_early_stop=enable_early_stop,
                min_round_trips=min_round_trips_box,
            )

            print(f"batch progress: {ridx + 1}/{len(boxes_raw)} -> {box_id}")

            cfg_both = clone_cfg(case_cfg_base, side_mode="both")
            _, both_best_params, both_best_result, _ = run_scan(df, cfg_both, args.max_combos, args.scan_mode)
            st_both = dict(both_best_result["stats"])
            both_main_bars = int(len(both_best_result["bars"])) if "bars" in both_best_result else 0
            (
                main_orders_total_both,
                main_round_trips_both,
                main_rank_eligible_both,
                main_no_trade_both,
                main_reason_both,
            ) = _main_window_qualification(st_both, min_round_trips_box, both_main_bars)
            both_return = float(st_both["return_pct"]) if main_reason_both == "" else np.nan
            both_objective = float(st_both["objective"]) if main_reason_both == "" else np.nan
            selected_high = float(st_both.get("selected_risk_high", box_high))
            probe_tie_reason_main = str(st_both.get("probe_tie_reason", ""))
            selected_source_main = "fallback" if probe_tie_reason_main.startswith("FALLBACK") else ("probe" if probe_tie_reason_main else "probe_disabled")
            both_diag = _main_diag_fields(st_both, "both")

            both_return_no_short_rule = np.nan
            both_objective_no_short_rule = np.nan
            both_return_change_vs_no_short_rule = np.nan
            both_objective_change_vs_no_short_rule = np.nan
            if str(cfg_both.short_enable_rule) != "none":
                cfg_both_no_short = clone_cfg(cfg_both, short_enable_rule="none")
                _, _, both_no_short_res, _ = run_scan(df, cfg_both_no_short, args.max_combos, args.scan_mode)
                st_both_no_short = dict(both_no_short_res["stats"])
                both_no_short_bars = int(len(both_no_short_res["bars"])) if "bars" in both_no_short_res else 0
                _, _, _, _, reason_no_short = _main_window_qualification(st_both_no_short, min_round_trips_box, both_no_short_bars)
                if reason_no_short == "":
                    both_return_no_short_rule = float(st_both_no_short["return_pct"])
                    both_objective_no_short_rule = float(st_both_no_short["objective"])
                if pd.notna(both_return) and pd.notna(both_return_no_short_rule):
                    both_return_change_vs_no_short_rule = float(both_return - both_return_no_short_rule)
                if pd.notna(both_objective) and pd.notna(both_objective_no_short_rule):
                    both_objective_change_vs_no_short_rule = float(both_objective - both_objective_no_short_rule)

            cfg_long = clone_cfg(case_cfg_base, side_mode="long_only")
            _, _, long_best_result, _ = run_scan(df, cfg_long, args.max_combos, args.scan_mode)
            st_long = dict(long_best_result["stats"])
            long_main_bars = int(len(long_best_result["bars"])) if "bars" in long_best_result else 0
            (
                main_orders_total_long,
                main_round_trips_long,
                main_rank_eligible_long,
                main_no_trade_long,
                main_reason_long,
            ) = _main_window_qualification(st_long, min_round_trips_box, long_main_bars)
            long_only_return = float(st_long["return_pct"]) if main_reason_long == "" else np.nan
            long_only_objective = float(st_long["objective"]) if main_reason_long == "" else np.nan
            long_diag = _main_diag_fields(st_long, "long")
            both_minus_long_only = (
                float(both_return - long_only_return)
                if (pd.notna(both_return) and pd.notna(long_only_return))
                else np.nan
            )

            diag_run_used = False
            diag_min_round_trips_box = np.nan
            diag_main_reason = ""
            diag_signal_count_total = np.nan
            diag_filled_entries = np.nan
            diag_entry_fill_rate = np.nan
            diag_trade_box_invalid_ratio = np.nan
            diag_orders_total = np.nan
            diag_round_trips = np.nan
            if main_reason_both in {"NO_TRADES", "INSUFFICIENT_TRIPS", "BOX_TOO_SHORT"}:
                diag_run_used = True
                diag_min_round_trips_box = 1
                cfg_diag = clone_cfg(cfg_both, min_round_trips=1)
                _, _, diag_result, _ = run_scan(df, cfg_diag, args.max_combos, args.scan_mode)
                st_diag = dict(diag_result["stats"])
                diag_bars = int(len(diag_result["bars"])) if "bars" in diag_result else 0
                _, _, _, _, diag_main_reason = _main_window_qualification(st_diag, 1, diag_bars)
                diag_signal_count_total = float(st_diag.get("signal_count_total", np.nan))
                diag_filled_entries = float(st_diag.get("filled_entries", np.nan))
                diag_entry_fill_rate = float(st_diag.get("entry_fill_rate", np.nan))
                diag_trade_box_invalid_ratio = float(st_diag.get("trade_box_invalid_ratio", np.nan))
                diag_orders_total = float(st_diag.get("orders_total", np.nan))
                diag_round_trips = float(st_diag.get("round_trips", np.nan))

            box_prefix = f"manual_box_{args.round_tag}_{safe_box_id}"
            both_trades_path = out_dir / f"{box_prefix}_best_trades.csv"
            both_equity_path = out_dir / f"{box_prefix}_best_equity.csv"
            both_html_path = out_dir / f"{box_prefix}_best_visual.html"
            both_report_path = out_dir / f"{box_prefix}_report.md"
            long_trades_path = out_dir / f"{box_prefix}_longonly_best_trades.csv"
            long_equity_path = out_dir / f"{box_prefix}_longonly_best_equity.csv"
            long_html_path = out_dir / f"{box_prefix}_longonly_best_visual.html"
            long_report_path = out_dir / f"{box_prefix}_longonly_report.md"
            side_compare_path = out_dir / f"{box_prefix}_side_compare_summary.csv"

            both_best_result["trades"].to_csv(both_trades_path, index=False)
            both_best_result["equity"].reset_index().to_csv(both_equity_path, index=False)
            write_visual_html(both_html_path, both_best_result["bars"], both_best_result["trades"], both_best_result["equity"])
            with both_report_path.open("w", encoding="utf-8") as f:
                f.write(f"# {box_prefix} (both)\n\n")
                f.write(f"- box_id: `{box_id}`\n")
                f.write(f"- start_ts: `{start_utc}`\n")
                f.write(f"- end_ts: `{end_utc}`\n")
                f.write(f"- box_low/high: `{box_low}` / `{box_high}`\n")
                f.write(f"- return_pct: `{st_both.get('return_pct', np.nan)}`\n")
                f.write(f"- max_dd_pct: `{st_both.get('max_dd_pct', np.nan)}`\n")
                f.write(f"- profit_factor: `{st_both.get('profit_factor', np.nan)}`\n")
                f.write(f"- objective: `{st_both.get('objective', np.nan)}`\n")
                f.write(f"- round_trips: `{st_both.get('round_trips', np.nan)}`\n")
                f.write(f"- hard_stop_count: `{st_both.get('hard_stop_count', np.nan)}`\n")
                f.write(f"- hard_stop_ratio: `{st_both.get('hard_stop_ratio', np.nan)}`\n")
                f.write(f"- atr_stop_count: `{st_both.get('atr_stop_count', np.nan)}`\n")
                f.write(f"- time_stop_count: `{st_both.get('time_stop_count', np.nan)}`\n")
                f.write(f"- runner_stop_count: `{st_both.get('runner_stop_count', np.nan)}`\n")
                f.write(f"- cb_trigger_count: `{st_both.get('cb_trigger_count', np.nan)}`\n")
                f.write(f"- cb_disable_bars_ratio: `{st_both.get('cb_disable_bars_ratio', np.nan)}`\n")
                f.write(f"- exit_reasons_in_trades: `{sorted(set(both_best_result['trades']['reason'].dropna().astype(str).tolist())) if not both_best_result['trades'].empty else []}`\n")

            long_best_result["trades"].to_csv(long_trades_path, index=False)
            long_best_result["equity"].reset_index().to_csv(long_equity_path, index=False)
            write_visual_html(long_html_path, long_best_result["bars"], long_best_result["trades"], long_best_result["equity"])
            with long_report_path.open("w", encoding="utf-8") as f:
                f.write(f"# {box_prefix} (long_only)\n\n")
                f.write(f"- box_id: `{box_id}`\n")
                f.write(f"- return_pct: `{st_long.get('return_pct', np.nan)}`\n")
                f.write(f"- max_dd_pct: `{st_long.get('max_dd_pct', np.nan)}`\n")
                f.write(f"- profit_factor: `{st_long.get('profit_factor', np.nan)}`\n")
                f.write(f"- objective: `{st_long.get('objective', np.nan)}`\n")
                f.write(f"- round_trips: `{st_long.get('round_trips', np.nan)}`\n")
                f.write(f"- hard_stop_count: `{st_long.get('hard_stop_count', np.nan)}`\n")
                f.write(f"- hard_stop_ratio: `{st_long.get('hard_stop_ratio', np.nan)}`\n")
                f.write(f"- atr_stop_count: `{st_long.get('atr_stop_count', np.nan)}`\n")
                f.write(f"- time_stop_count: `{st_long.get('time_stop_count', np.nan)}`\n")
                f.write(f"- runner_stop_count: `{st_long.get('runner_stop_count', np.nan)}`\n")
                f.write(f"- cb_trigger_count: `{st_long.get('cb_trigger_count', np.nan)}`\n")
                f.write(f"- cb_disable_bars_ratio: `{st_long.get('cb_disable_bars_ratio', np.nan)}`\n")

            pd.DataFrame(
                [
                    {
                        "variant": "both",
                        "return_pct": st_both.get("return_pct", np.nan),
                        "max_dd_pct": st_both.get("max_dd_pct", np.nan),
                        "profit_factor": st_both.get("profit_factor", np.nan),
                        "objective": st_both.get("objective", np.nan),
                        "round_trips": st_both.get("round_trips", np.nan),
                        "hard_stop_count": st_both.get("hard_stop_count", np.nan),
                        "cb_trigger_count": st_both.get("cb_trigger_count", np.nan),
                        "cb_disable_bars_ratio": st_both.get("cb_disable_bars_ratio", np.nan),
                    },
                    {
                        "variant": "long_only",
                        "return_pct": st_long.get("return_pct", np.nan),
                        "max_dd_pct": st_long.get("max_dd_pct", np.nan),
                        "profit_factor": st_long.get("profit_factor", np.nan),
                        "objective": st_long.get("objective", np.nan),
                        "round_trips": st_long.get("round_trips", np.nan),
                        "hard_stop_count": st_long.get("hard_stop_count", np.nan),
                        "cb_trigger_count": st_long.get("cb_trigger_count", np.nan),
                        "cb_disable_bars_ratio": st_long.get("cb_disable_bars_ratio", np.nan),
                    },
                ]
            ).to_csv(side_compare_path, index=False)

            wf_segment_days, wf_step_days, wf_reason = _adaptive_walk_forward_schedule(box_days)
            walk_forward_csv = ""
            wf_num_segments = np.nan
            wf_num_trading_segments = np.nan
            wf_no_trade_segment_rate = np.nan
            wf_pf_reliable_rate = np.nan
            wf_selected_source_probe_ratio = np.nan
            wf_selected_source_fallback_ratio = np.nan
            walk_forward_median_objective = np.nan
            walk_forward_early_stop_rate = np.nan
            if wf_segment_days is not None and wf_step_days is not None:
                walk_path = out_dir / f"manual_box_{args.round_tag}_{safe_box_id}_walk_forward_summary.csv"
                walk_df = run_walk_forward(
                    df=df,
                    cfg=cfg_both,
                    p=both_best_params,
                    segment_days=int(wf_segment_days),
                    step_days=int(wf_step_days),
                    out_path=walk_path,
                    max_combos=max(int(args.max_combos), 1),
                    scan_mode=str(args.scan_mode),
                )
                walk_forward_csv = str(walk_path)
                wf_num_segments = int(len(walk_df))
                if int(wf_num_segments) > 0:
                    if "orders_total" in walk_df.columns:
                        trading_mask = (walk_df["orders_total"].astype(int) > 0) | (walk_df["round_trips"].astype(int) > 0)
                    else:
                        trading_mask = walk_df["round_trips"].astype(int) > 0
                    wf_num_trading_segments = int(trading_mask.sum())
                    wf_no_trade_segment_rate = float((int(wf_num_segments) - int(wf_num_trading_segments)) / int(wf_num_segments))
                    wf_pf_reliable_rate = float(walk_df["pf_reliable"].astype(bool).mean()) if "pf_reliable" in walk_df.columns else np.nan
                    if "selected_source" in walk_df.columns:
                        wf_selected_source_probe_ratio = float((walk_df["selected_source"].astype(str) == "probe").mean())
                        wf_selected_source_fallback_ratio = float((walk_df["selected_source"].astype(str) == "fallback").mean())
                    if int(wf_num_trading_segments) > 0:
                        walk_forward_median_objective = float(walk_df.loc[trading_mask, "objective"].median())
                        walk_forward_early_stop_rate = float(walk_df["early_stop_flag"].astype(bool).mean())
                        wf_reason = ""
                    else:
                        walk_forward_median_objective = np.nan
                        wf_reason = "NO_TRADING_SEGMENTS"
                else:
                    wf_reason = "NO_WF_SEGMENTS"
                    wf_num_trading_segments = 0

            rows.append(
                {
                    "box_id": box_id,
                    "status": "ok",
                    "start_ts": str(start_utc),
                    "end_ts": str(end_utc),
                    "box_low": box_low,
                    "box_high": box_high,
                    "risk_high_candidates": ",".join(str(x) for x in risk_high_candidates),
                    "box_days": box_days,
                    "box_bars": box_bars,
                    "min_round_trips_box": min_round_trips_box,
                    "main_orders_total_both": main_orders_total_both,
                    "main_round_trips_both": main_round_trips_both,
                    "main_rank_eligible_both": bool(main_rank_eligible_both),
                    "main_no_trade_both": bool(main_no_trade_both),
                    "main_reason": main_reason_both,
                    "main_reason_both": main_reason_both,
                    "main_orders_total_long": main_orders_total_long,
                    "main_round_trips_long": main_round_trips_long,
                    "main_rank_eligible_long": bool(main_rank_eligible_long),
                    "main_no_trade_long": bool(main_no_trade_long),
                    "main_reason_long": main_reason_long,
                    **both_diag,
                    **long_diag,
                    "both_return": both_return,
                    "both_profit_factor": float(st_both.get("profit_factor", np.nan)),
                    "long_only_return": long_only_return,
                    "long_only_profit_factor": float(st_long.get("profit_factor", np.nan)),
                    "both_minus_long_only": both_minus_long_only,
                    "both_objective": both_objective,
                    "long_only_objective": long_only_objective,
                    "both_return_no_short_rule": both_return_no_short_rule,
                    "both_objective_no_short_rule": both_objective_no_short_rule,
                    "both_return_change_vs_no_short_rule": both_return_change_vs_no_short_rule,
                    "both_objective_change_vs_no_short_rule": both_objective_change_vs_no_short_rule,
                    "short_enable_rule": str(cfg.short_enable_rule),
                    "short_enable_lookback_days": int(cfg.short_enable_lookback_days),
                    "short_enable_min_rejects": int(cfg.short_enable_min_rejects),
                    "walk_forward_median_objective": walk_forward_median_objective,
                    "walk_forward_early_stop_rate": walk_forward_early_stop_rate,
                    "wf_median_objective": walk_forward_median_objective,
                    "wf_early_stop_rate": walk_forward_early_stop_rate,
                    "wf_num_segments": wf_num_segments,
                    "wf_num_trading_segments": wf_num_trading_segments,
                    "wf_no_trade_segment_rate": wf_no_trade_segment_rate,
                    "wf_pf_reliable_rate": wf_pf_reliable_rate,
                    "wf_reason": wf_reason,
                    "diag_run_used": bool(diag_run_used),
                    "diag_min_round_trips_box": diag_min_round_trips_box,
                    "diag_main_reason": diag_main_reason,
                    "diag_signal_count_total": diag_signal_count_total,
                    "diag_filled_entries": diag_filled_entries,
                    "diag_entry_fill_rate": diag_entry_fill_rate,
                    "diag_trade_box_invalid_ratio": diag_trade_box_invalid_ratio,
                    "diag_orders_total": diag_orders_total,
                    "diag_round_trips": diag_round_trips,
                    "selected_high": selected_high,
                    "selected_source_main": selected_source_main,
                    "probe_tie_reason_main": probe_tie_reason_main,
                    "wf_selected_source_probe_ratio": wf_selected_source_probe_ratio,
                    "wf_selected_source_fallback_ratio": wf_selected_source_fallback_ratio,
                    "both_report": str(both_report_path),
                    "both_html": str(both_html_path),
                    "long_only_report": str(long_report_path),
                    "long_only_html": str(long_html_path),
                    "side_compare_csv": str(side_compare_path),
                    "walk_forward_csv": walk_forward_csv,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "box_id": box_id,
                    "status": "error",
                    "start_ts": str(row.get(c_start, "")),
                    "end_ts": str(row.get(c_end, "")),
                    "box_low": row.get(c_low, np.nan),
                    "box_high": row.get(c_high, np.nan),
                    "risk_high_candidates": str(row.get(c_cands, "")) if c_cands is not None else "",
                    "box_days": np.nan,
                    "box_bars": np.nan,
                    "min_round_trips_box": np.nan,
                    "main_orders_total_both": np.nan,
                    "main_round_trips_both": np.nan,
                    "main_rank_eligible_both": np.nan,
                    "main_no_trade_both": np.nan,
                    "main_reason": "",
                    "main_reason_both": "",
                    "main_orders_total_long": np.nan,
                    "main_round_trips_long": np.nan,
                    "main_rank_eligible_long": np.nan,
                    "main_no_trade_long": np.nan,
                    "main_reason_long": "",
                    "main_filled_entries_both": np.nan,
                    "main_missed_entry_both": np.nan,
                    "main_entry_fill_rate_both": np.nan,
                    "main_trade_box_invalid_ratio_both": np.nan,
                    "main_box_invalid_count_both": np.nan,
                    "main_risk_expand_count_both": np.nan,
                    "main_avg_hold_bars_both": np.nan,
                    "main_avg_layers_both": np.nan,
                    "main_layer_add_count_both": np.nan,
                    "main_short_trade_count_both": np.nan,
                    "main_long_trade_count_both": np.nan,
                    "main_signal_count_total_both": np.nan,
                    "main_signal_count_long_both": np.nan,
                    "main_signal_count_short_both": np.nan,
                    "short_enabled_ratio_both": np.nan,
                    "main_filled_entries_long": np.nan,
                    "main_missed_entry_long": np.nan,
                    "main_entry_fill_rate_long": np.nan,
                    "main_trade_box_invalid_ratio_long": np.nan,
                    "main_box_invalid_count_long": np.nan,
                    "main_risk_expand_count_long": np.nan,
                    "main_avg_hold_bars_long": np.nan,
                    "main_avg_layers_long": np.nan,
                    "main_layer_add_count_long": np.nan,
                    "main_short_trade_count_long": np.nan,
                    "main_long_trade_count_long": np.nan,
                    "main_signal_count_total_long": np.nan,
                    "main_signal_count_long_long": np.nan,
                    "main_signal_count_short_long": np.nan,
                    "short_enabled_ratio_long": np.nan,
                    "both_return": np.nan,
                    "long_only_return": np.nan,
                    "both_minus_long_only": np.nan,
                    "both_objective": np.nan,
                    "long_only_objective": np.nan,
                    "both_return_no_short_rule": np.nan,
                    "both_objective_no_short_rule": np.nan,
                    "both_return_change_vs_no_short_rule": np.nan,
                    "both_objective_change_vs_no_short_rule": np.nan,
                    "short_enable_rule": str(cfg.short_enable_rule),
                    "short_enable_lookback_days": int(cfg.short_enable_lookback_days),
                    "short_enable_min_rejects": int(cfg.short_enable_min_rejects),
                    "walk_forward_median_objective": np.nan,
                    "walk_forward_early_stop_rate": np.nan,
                    "wf_median_objective": np.nan,
                    "wf_early_stop_rate": np.nan,
                    "wf_num_segments": np.nan,
                    "wf_num_trading_segments": np.nan,
                    "wf_no_trade_segment_rate": np.nan,
                    "wf_pf_reliable_rate": np.nan,
                    "wf_reason": "",
                    "diag_run_used": np.nan,
                    "diag_min_round_trips_box": np.nan,
                    "diag_main_reason": "",
                    "diag_signal_count_total": np.nan,
                    "diag_filled_entries": np.nan,
                    "diag_entry_fill_rate": np.nan,
                    "diag_trade_box_invalid_ratio": np.nan,
                    "diag_orders_total": np.nan,
                    "diag_round_trips": np.nan,
                    "selected_high": np.nan,
                    "selected_source_main": "",
                    "probe_tie_reason_main": "",
                    "wf_selected_source_probe_ratio": np.nan,
                    "wf_selected_source_fallback_ratio": np.nan,
                    "both_report": "",
                    "both_html": "",
                    "long_only_report": "",
                    "long_only_html": "",
                    "side_compare_csv": "",
                    "walk_forward_csv": "",
                    "error": str(exc),
                }
            )

        out_df = pd.DataFrame(rows)
        out_df.to_csv(summary_path, index=False)

    return pd.DataFrame(rows)


def main():
    global BOX_LOW, BOX_HIGH, BOX_WIDTH, __git_hash__

    parser = argparse.ArgumentParser(description="Manual box mean-reversion with strict no-lookahead")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV)
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--box-source", type=str, choices=["dynamic", "manual"], default="dynamic")
    parser.add_argument("--clip-dynamic-to-manual", type=parse_bool, default=False)
    parser.add_argument("--macro-box-mode", type=str, choices=["donchian", "bb"], default="donchian")
    parser.add_argument("--macro-lookback-bars", type=int, default=84)
    parser.add_argument("--macro-bb-len", type=int, default=84)
    parser.add_argument("--macro-bb-std", type=float, default=2.0)
    parser.add_argument("--box-low", type=float, default=BOX_LOW)
    parser.add_argument("--box-high", type=float, default=BOX_HIGH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--round-tag", type=str, default="roundX")
    parser.add_argument("--archive-run", action="store_true")
    parser.add_argument("--baseline-suite", action="store_true")
    parser.add_argument("--batch-box-eval", action="store_true")
    parser.add_argument("--boxes-csv", type=str, default="boxes.csv")
    parser.add_argument("--batch-summary-name", type=str, default="batch_boxes_summary.csv")
    parser.add_argument("--max-combos", type=int, default=12000)
    parser.add_argument("--scan-mode", type=str, choices=["local", "random"], default="local")
    parser.add_argument("--commission-pct", type=float, default=0.04)
    parser.add_argument("--slippage-pct", type=float, default=0.0)
    parser.add_argument("--maker-fill-prob", type=float, default=1.0)
    parser.add_argument("--maker-queue-delay-bars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--open-html", action="store_true")
    parser.add_argument("--side-mode", type=str, choices=["both", "long_only"], default="both")
    parser.add_argument("--compare-long-only", action="store_true")
    parser.add_argument(
        "--trade-box-mode",
        type=str,
        choices=["risk", "rolling_hilo", "rolling_quantile", "ema_atr"],
        default="rolling_quantile",
    )
    parser.add_argument("--trade-box-lookback", type=int, default=60)
    parser.add_argument("--trade-box-q-low", type=float, default=0.20)
    parser.add_argument("--trade-box-q-high", type=float, default=0.80)
    parser.add_argument("--trade-box-ema-len", type=int, default=34)
    parser.add_argument("--trade-box-atr-mult", type=float, default=1.0)
    parser.add_argument("--risk-high-candidates", type=str, default="")
    parser.add_argument("--probe-days", type=int, default=5)
    parser.add_argument("--probe-metric", type=str, default="objective")
    parser.add_argument("--probe-execution-mode", type=str, choices=["taker_entry", "maker_fallback_taker"], default="taker_entry")
    parser.add_argument("--fallback-no-signal-mode", type=str, choices=["use_input_high", "highest_high"], default="use_input_high")
    parser.add_argument("--probe-tie-eps", type=float, default=0.3)
    parser.add_argument("--probe-min-round-trips", type=int, default=1)
    parser.add_argument("--probe-min-entry-fill-rate", type=float, default=0.85)
    parser.add_argument("--probe-max-hard-stop-ratio", type=float, default=0.6)
    parser.add_argument("--early-stop-first-k-trades", type=int, default=3)
    parser.add_argument("--early-stop-hard-stop-threshold", type=int, default=2)
    parser.add_argument("--early-stop-first-m-bars", type=int, default=12)
    parser.add_argument("--min-round-trips", type=int, default=6)
    parser.add_argument("--pf-unreliable-penalty", type=float, default=0.5)
    parser.add_argument("--complexity-penalty-lambda", type=float, default=0.35)
    parser.add_argument("--degrade-on-trend", type=parse_bool, default=False)
    parser.add_argument("--ab-degrade-compare", action="store_true")
    parser.add_argument("--trend-slope-thresh", type=float, default=0.015)
    parser.add_argument("--atr-expand-thresh", type=float, default=0.12)
    parser.add_argument("--degrade-risk-lookback-bars", type=int, default=12)
    parser.add_argument("--short-enable-rule", type=str, choices=["none", "simple"], default="none")
    parser.add_argument("--short-enable-lookback-days", type=int, default=10)
    parser.add_argument("--short-enable-min-rejects", type=int, default=2)
    parser.add_argument("--short-enable-touch-band", type=float, default=0.10)
    parser.add_argument("--short-enable-reject-close-gap", type=float, default=0.02)
    parser.add_argument("--start-gate", type=str, choices=["off", "on"], default="on")
    parser.add_argument("--gate-adx-thresh", type=float, default=25.0)
    parser.add_argument("--gate-ema-slope-thresh", type=float, default=0.0015)
    parser.add_argument("--gate-chop-thresh", type=float, default=55.0)
    parser.add_argument("--gate-edge-reject-lookback-bars", type=int, default=60)
    parser.add_argument("--gate-edge-reject-min-count", type=int, default=2)
    parser.add_argument("--gate-edge-reject-atr-mult", type=float, default=0.5)
    parser.add_argument("--invalidate", type=str, choices=["off", "on"], default="on")
    parser.add_argument("--invalidate-m", type=int, default=2)
    parser.add_argument("--invalidate-buffer-mode", type=str, choices=["pct", "atr"], default="atr")
    parser.add_argument("--invalidate-buffer-atr-mult", type=float, default=1.0)
    parser.add_argument("--invalidate-buffer-pct", type=float, default=0.15)
    parser.add_argument("--invalidate-action", type=str, choices=["disable_only", "force_flatten"], default="disable_only")
    parser.add_argument("--cooldown-after-invalidate-bars", type=int, default=24)
    parser.add_argument("--perf-stop", type=str, choices=["off", "on"], default="on")
    parser.add_argument("--perf-window-trades", type=int, default=12)
    parser.add_argument("--perf-min-profit-factor", type=float, default=1.0)
    parser.add_argument("--perf-max-hard-stop-ratio", type=float, default=0.30)
    parser.add_argument("--perf-action", type=str, choices=["disable_only", "force_flatten"], default="disable_only")
    parser.add_argument("--cooldown-after-perf-stop-bars", type=int, default=24)
    parser.add_argument("--regime-gate", type=str, choices=["off", "on"], default="off")
    parser.add_argument("--regime-gate-adx-thresh", type=float, default=22.0)
    parser.add_argument("--regime-gate-bbwidth-min", type=float, default=0.05)
    parser.add_argument("--regime-gate-chop-thresh", type=float, default=50.0)
    parser.add_argument("--regime-gate-bbwidth-q-thresh", type=float, default=0.30)
    parser.add_argument("--regime-gate-slope-thresh", type=float, default=0.0015)
    parser.add_argument("--circuit-breaker", type=str, choices=["off", "on"], default="off")
    parser.add_argument("--circuit-break-adx-thresh", type=float, default=25.0)
    parser.add_argument("--circuit-break-bbwidth-q-thresh", type=float, default=0.50)
    parser.add_argument("--circuit-break-outside-consecutive", type=int, default=2)
    parser.add_argument("--cooldown-cb-bars", type=int, default=8)
    parser.add_argument("--cb-force-flatten", type=parse_bool, default=False)
    parser.add_argument("--atr-stop", type=str, choices=["off", "on"], default="off")
    parser.add_argument("--atr-stop-mult", type=float, default=1.5)
    parser.add_argument("--structural-cooldown-bars", type=int, default=6)
    parser.add_argument("--local-time-stop-bars", type=int, default=30)
    parser.add_argument("--enable-runner", type=parse_bool, default=False)
    parser.add_argument("--runner-pct", type=float, default=0.20)
    parser.add_argument("--runner-atr-mult", type=float, default=2.0)
    parser.add_argument("--override-base-risk-pct", type=float, default=None)
    parser.add_argument("--override-layer-weights", type=str, default="")
    parser.add_argument("--override-max-leverage", type=float, default=None)
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--run-validation-suite", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--segment-days", type=int, default=14)
    parser.add_argument("--step-days", type=int, default=4)
    parser.add_argument("--release-gate-max-early-stop-rate", type=float, default=0.30)
    parser.add_argument("--release-gate-min-median-return", type=float, default=0.0)
    parser.add_argument("--release-gate-min-median-pf", type=float, default=1.1)

    parser.add_argument("--dynamic-box-mode", type=str, choices=["none", "expand", "expand_invalidate"], default="none")
    parser.add_argument("--break-confirm-closes", type=int, default=2)
    parser.add_argument("--break-buffer-mode", type=str, choices=["pct", "atr"], default="atr")
    parser.add_argument("--break-buffer-pct", type=float, default=0.15)
    parser.add_argument("--break-buffer-atr-mult", type=float, default=1.0)
    parser.add_argument("--expand-atr-mult", type=float, default=1.0)
    parser.add_argument("--invalidate-force-close", type=parse_bool, default=False)
    parser.add_argument("--max-expands", type=int, default=6)
    parser.add_argument("--freeze-after-expand-bars", type=int, default=6)

    parser.add_argument("--robustness-test", action="store_true")
    parser.add_argument("--robust-long-L", type=float, default=None)
    parser.add_argument("--robust-mid1", type=float, default=0.60)
    parser.add_argument("--robust-mid2", type=float, default=0.64)
    parser.add_argument("--robust-base-risk-pct", type=float, default=0.45)
    parser.add_argument("--robust-layer-weights", type=str, default="1.0,1.5,2.2")
    parser.add_argument("--robust-layer-step", type=float, default=0.07)
    parser.add_argument("--robust-max-leverage", type=float, default=2.0)
    parser.add_argument("--robust-hard-stop-pct", type=float, default=0.05)
    parser.add_argument("--robust-cooldown-rearm-bars", type=int, default=2)
    parser.add_argument("--robust-cooldown-stop-bars", type=int, default=24)
    parser.add_argument("--robust-time-stop-bars", type=int, default=36)
    parser.add_argument("--robust-maker-ttl-bars", type=int, default=2)
    parser.add_argument("--robust-maker-fallback-taker", type=parse_bool, default=True)
    parser.add_argument("--robust-improve-mult", type=float, default=0.15)
    parser.add_argument("--robust-min-improve-bps", type=float, default=0.5)
    parser.add_argument(
        "--robust-trade-box-mode",
        type=str,
        choices=["risk", "rolling_hilo", "rolling_quantile", "ema_atr"],
        default="rolling_quantile",
    )
    parser.add_argument("--robust-trade-box-lookback", type=int, default=60)
    parser.add_argument("--robust-trade-box-q-low", type=float, default=0.20)
    parser.add_argument("--robust-trade-box-q-high", type=float, default=0.80)
    parser.add_argument("--robust-trade-box-ema-len", type=int, default=34)
    parser.add_argument("--robust-trade-box-atr-mult", type=float, default=1.0)

    args = parser.parse_args()
    __git_hash__ = resolve_git_hash()

    BOX_LOW = float(args.box_low)
    BOX_HIGH = float(args.box_high)
    BOX_WIDTH = BOX_HIGH - BOX_LOW
    if args.box_source == "manual" and BOX_WIDTH <= 0:
        raise ValueError("box_high must be greater than box_low")

    start_utc = parse_ts_utc(args.start)
    end_utc = parse_ts_utc(args.end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    if float(args.trade_box_q_high) <= float(args.trade_box_q_low):
        raise ValueError("trade-box-q-high must be > trade-box-q-low")
    if float(args.robust_trade_box_q_high) <= float(args.robust_trade_box_q_low):
        raise ValueError("robust-trade-box-q-high must be > robust-trade-box-q-low")
    if not (0.0 <= float(args.maker_fill_prob) <= 1.0):
        raise ValueError("maker-fill-prob must be in [0,1]")
    if int(args.macro_lookback_bars) < 2:
        raise ValueError("macro-lookback-bars must be >=2")
    if int(args.macro_bb_len) < 2:
        raise ValueError("macro-bb-len must be >=2")
    if float(args.macro_bb_std) <= 0.0:
        raise ValueError("macro-bb-std must be >0")
    if not (0.0 < float(args.regime_gate_bbwidth_q_thresh) < 1.0):
        raise ValueError("regime-gate-bbwidth-q-thresh must be in (0,1)")
    if not (0.0 < float(args.circuit_break_bbwidth_q_thresh) < 1.0):
        raise ValueError("circuit-break-bbwidth-q-thresh must be in (0,1)")
    if float(args.regime_gate_bbwidth_min) < 0.0:
        raise ValueError("regime-gate-bbwidth-min must be >= 0")
    if float(args.regime_gate_chop_thresh) < 0.0:
        raise ValueError("regime-gate-chop-thresh must be >= 0")
    if not (0.0 <= float(args.runner_pct) < 1.0):
        raise ValueError("runner-pct must be in [0,1)")
    if float(args.runner_atr_mult) <= 0.0:
        raise ValueError("runner-atr-mult must be > 0")
    override_layer_weights = None
    override_layer_weights_txt = str(args.override_layer_weights).strip()
    if override_layer_weights_txt:
        override_layer_weights = parse_layer_weights(override_layer_weights_txt)
        if any(x <= 0.0 for x in override_layer_weights):
            raise ValueError("override-layer-weights must be positive floats")
    if args.override_base_risk_pct is not None and float(args.override_base_risk_pct) <= 0.0:
        raise ValueError("override-base-risk-pct must be > 0")
    if args.override_max_leverage is not None and float(args.override_max_leverage) <= 0.0:
        raise ValueError("override-max-leverage must be > 0")

    dyn_cfg = DynamicBoxConfig(
        mode=args.dynamic_box_mode,
        break_confirm_closes=max(int(args.break_confirm_closes), 1),
        break_buffer_mode=args.break_buffer_mode,
        break_buffer_pct=float(args.break_buffer_pct),
        break_buffer_atr_mult=float(args.break_buffer_atr_mult),
        expand_atr_mult=float(args.expand_atr_mult),
        invalidate_force_close=parse_bool(args.invalidate_force_close),
        max_expands=max(int(args.max_expands), 0),
        freeze_after_expand_bars=max(int(args.freeze_after_expand_bars), 0),
    )
    risk_high_candidates = parse_float_list(args.risk_high_candidates)
    probe_days = max(int(args.probe_days), 0)
    enable_early_stop = bool(len(risk_high_candidates) > 0 and probe_days > 0)

    cfg = RunConfig(
        start_utc=start_utc,
        end_utc=end_utc,
        commission_pct=float(args.commission_pct),
        slippage_pct=float(args.slippage_pct),
        box_low=BOX_LOW,
        box_high=BOX_HIGH,
        dyn_cfg=dyn_cfg,
        side_mode=args.side_mode,
        trade_box_mode_default=str(args.trade_box_mode),
        trade_box_lookback_default=max(int(args.trade_box_lookback), 2),
        trade_box_q_low_default=float(args.trade_box_q_low),
        trade_box_q_high_default=float(args.trade_box_q_high),
        trade_box_ema_len_default=max(int(args.trade_box_ema_len), 2),
        trade_box_atr_mult_default=float(args.trade_box_atr_mult),
        risk_high_candidates=risk_high_candidates,
        probe_days=probe_days,
        probe_metric=str(args.probe_metric),
        probe_execution_mode=str(args.probe_execution_mode),
        fallback_no_signal_mode=str(args.fallback_no_signal_mode),
        probe_tie_eps=max(float(args.probe_tie_eps), 0.0),
        probe_min_round_trips=max(int(args.probe_min_round_trips), 0),
        probe_min_entry_fill_rate=float(args.probe_min_entry_fill_rate),
        probe_max_hard_stop_ratio=float(args.probe_max_hard_stop_ratio),
        early_stop_first_k_trades=max(int(args.early_stop_first_k_trades), 0),
        early_stop_hard_stop_threshold=max(int(args.early_stop_hard_stop_threshold), 0),
        early_stop_first_m_bars=max(int(args.early_stop_first_m_bars), 0),
        enable_early_stop=enable_early_stop,
        entry_execution_mode="maker",
        maker_fill_prob=float(args.maker_fill_prob),
        maker_queue_delay_bars=max(int(args.maker_queue_delay_bars), 0),
        seed=int(args.seed),
        min_round_trips=max(int(args.min_round_trips), 0),
        pf_unreliable_penalty=max(float(args.pf_unreliable_penalty), 0.0),
        complexity_penalty_lambda=max(float(args.complexity_penalty_lambda), 0.0),
        degrade_on_trend=parse_bool(args.degrade_on_trend),
        trend_slope_thresh=max(float(args.trend_slope_thresh), 0.0),
        atr_expand_thresh=max(float(args.atr_expand_thresh), 0.0),
        degrade_risk_lookback_bars=max(int(args.degrade_risk_lookback_bars), 1),
        short_enable_rule=str(args.short_enable_rule),
        short_enable_lookback_days=max(int(args.short_enable_lookback_days), 1),
        short_enable_min_rejects=max(int(args.short_enable_min_rejects), 1),
        short_enable_touch_band=max(float(args.short_enable_touch_band), 0.0),
        short_enable_reject_close_gap=max(float(args.short_enable_reject_close_gap), 0.0),
        start_gate_mode=str(args.start_gate),
        gate_adx_thresh=float(args.gate_adx_thresh),
        gate_ema_slope_thresh=max(float(args.gate_ema_slope_thresh), 0.0),
        gate_chop_thresh=float(args.gate_chop_thresh),
        gate_edge_reject_lookback_bars=max(int(args.gate_edge_reject_lookback_bars), 1),
        gate_edge_reject_min_count=max(int(args.gate_edge_reject_min_count), 1),
        gate_edge_reject_atr_mult=max(float(args.gate_edge_reject_atr_mult), 0.0),
        invalidate_mode=str(args.invalidate),
        invalidate_m=max(int(args.invalidate_m), 1),
        invalidate_buffer_mode=str(args.invalidate_buffer_mode),
        invalidate_buffer_atr_mult=max(float(args.invalidate_buffer_atr_mult), 0.0),
        invalidate_buffer_pct=max(float(args.invalidate_buffer_pct), 0.0),
        invalidate_action=str(args.invalidate_action),
        cooldown_after_invalidate_bars=max(int(args.cooldown_after_invalidate_bars), 0),
        perf_stop_mode=str(args.perf_stop),
        perf_window_trades=max(int(args.perf_window_trades), 1),
        perf_min_profit_factor=float(args.perf_min_profit_factor),
        perf_max_hard_stop_ratio=max(float(args.perf_max_hard_stop_ratio), 0.0),
        perf_action=str(args.perf_action),
        cooldown_after_perf_stop_bars=max(int(args.cooldown_after_perf_stop_bars), 0),
        box_source=str(args.box_source),
        clip_dynamic_to_manual=parse_bool(args.clip_dynamic_to_manual),
        macro_box_mode=str(args.macro_box_mode),
        macro_lookback_bars=max(int(args.macro_lookback_bars), 2),
        macro_bb_len=max(int(args.macro_bb_len), 2),
        macro_bb_std=float(args.macro_bb_std),
        regime_gate_mode=str(args.regime_gate),
        regime_gate_adx_thresh=float(args.regime_gate_adx_thresh),
        regime_gate_bbwidth_min=float(args.regime_gate_bbwidth_min),
        regime_gate_chop_thresh=float(args.regime_gate_chop_thresh),
        regime_gate_bbwidth_q_thresh=float(args.regime_gate_bbwidth_q_thresh),
        regime_gate_slope_thresh=max(float(args.regime_gate_slope_thresh), 0.0),
        circuit_breaker_mode=str(args.circuit_breaker),
        circuit_break_adx_thresh=float(args.circuit_break_adx_thresh),
        circuit_break_bbwidth_q_thresh=float(args.circuit_break_bbwidth_q_thresh),
        circuit_break_outside_consecutive=max(int(args.circuit_break_outside_consecutive), 2),
        cooldown_cb_bars=max(int(args.cooldown_cb_bars), 0),
        cb_force_flatten=parse_bool(args.cb_force_flatten),
        atr_stop_mode=str(args.atr_stop),
        atr_stop_mult=max(float(args.atr_stop_mult), 0.0),
        structural_cooldown_bars=max(int(args.structural_cooldown_bars), 0),
        local_time_stop_bars=max(int(args.local_time_stop_bars), 1),
        enable_runner=parse_bool(args.enable_runner),
        runner_pct=float(args.runner_pct),
        runner_atr_mult=float(args.runner_atr_mult),
        override_base_risk_pct=float(args.override_base_risk_pct) if args.override_base_risk_pct is not None else None,
        override_layer_weights=override_layer_weights,
        override_max_leverage=float(args.override_max_leverage) if args.override_max_leverage is not None else None,
    )

    csv_path = resolve_csv(args.csv)
    out_dir, out_warn = resolve_output_dir(args.output_dir)

    df = load_data(csv_path)
    df["atr14"] = calc_atr14(df)
    df["adx14"] = calc_adx14(df)
    df["ema50"] = df["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    df["ema50_slope_abs"] = ((df["ema50"] - df["ema50"].shift(1)) / df["ema50"].shift(1).replace(0.0, np.nan)).abs()
    df["ema50_slope20_abs"] = ((df["ema50"] - df["ema50"].shift(20)) / df["ema50"].shift(20).replace(0.0, np.nan)).abs()
    bb_mid, bb_upper, bb_lower, bb_width = calc_bollinger(df, window=20, num_std=2.0)
    df["bb_mid20"] = bb_mid
    df["bb_upper20"] = bb_upper
    df["bb_lower20"] = bb_lower
    df["bb_width20"] = bb_width
    df["bb_width_q30_120"] = df["bb_width20"].rolling(window=120, min_periods=120).quantile(0.30)
    df["bb_width_q50_120"] = df["bb_width20"].rolling(window=120, min_periods=120).quantile(0.50)
    df["chop72"] = calc_choppiness(df, window=72)
    df = df.dropna(subset=["atr14"]).copy()

    if args.batch_box_eval:
        batch_df = run_batch_box_eval(df=df, cfg=cfg, args=args, out_dir=out_dir)
        summary_path = out_dir / str(args.batch_summary_name)
        print("done (batch-box-eval)")
        print(f"- {summary_path}")
        print(f"- rows={len(batch_df)}")
        if args.open_html:
            print("open-html ignored in batch mode (no single html target)")
        return

    if args.baseline_suite:
        out = run_baseline_suite_mode(df=df, cfg=cfg, args=args, csv_path=csv_path, out_dir=out_dir)
        print("done (baseline-suite)")
        for k in ["trades", "equity", "html", "report", "probe", "validation", "walk", "walk_agg", "zip"]:
            if k in out:
                print(f"- {out[k]}")
        if args.open_html and "html" in out:
            try:
                subprocess.run(["open", str(out["html"])], check=False)
                print(f"opened: {out['html']}")
            except Exception as exc:  # noqa: BLE001
                print(f"open failed: {exc}")
        return

    all_df, best_params, best_result, total_grid = run_scan(df, cfg, args.max_combos, args.scan_mode)

    degrade_ab_df = None
    degrade_ab_path = None
    if args.ab_degrade_compare:
        ab_rows: List[Dict[str, object]] = []
        for mode in [False, True]:
            if bool(mode) == bool(cfg.degrade_on_trend):
                ab_all_df, ab_best_params, ab_best_result = all_df, best_params, best_result
            else:
                cfg_mode = clone_cfg(cfg, degrade_on_trend=bool(mode))
                ab_all_df, ab_best_params, ab_best_result, _ = run_scan(df, cfg_mode, args.max_combos, args.scan_mode)
            ab_st = ab_best_result["stats"]
            ab_rows.append(
                {
                    "degrade_on_trend": bool(mode),
                    "return_pct": float(ab_st["return_pct"]),
                    "max_dd_pct": float(ab_st["max_dd_pct"]),
                    "profit_factor": float(ab_st["profit_factor"]),
                    "round_trips": int(ab_st["round_trips"]),
                    "hard_stop_ratio": float(ab_st["hard_stop_ratio"]),
                    "entry_fill_rate": float(ab_st["entry_fill_rate"]),
                    "avg_layers": float(ab_st["avg_layers"]),
                    "layer_add_count": int(ab_st["layer_add_count"]),
                    "degrade_mode_bars_ratio": float(ab_st["degrade_mode_bars_ratio"]),
                    "objective": float(ab_st["objective"]),
                    "selected_risk_high": float(ab_st.get("selected_risk_high", cfg.box_high)),
                    "long_L": float(ab_best_params.long_L),
                    "mid1": float(ab_best_params.mid1),
                    "mid2": float(ab_best_params.mid2),
                    "layer_step": float(ab_best_params.layer_step),
                    "hard_stop_pct": float(ab_best_params.hard_stop_pct),
                    "base_risk_pct": float(ab_best_params.base_risk_pct),
                    "maker_ttl_bars": int(ab_best_params.maker_ttl_bars),
                    "min_improve_bps": float(ab_best_params.min_improve_bps),
                }
            )
        degrade_ab_df = pd.DataFrame(ab_rows).sort_values("degrade_on_trend").reset_index(drop=True)

    prefix = f"manual_box_{args.round_tag}"
    out_scan = out_dir / f"{prefix}_scan_all.csv"
    out_trades = out_dir / f"{prefix}_best_trades.csv"
    out_equity = out_dir / f"{prefix}_best_equity.csv"
    out_html = out_dir / f"{prefix}_best_visual.html"
    out_report = out_dir / f"{prefix}_report.md"
    out_probe = out_dir / f"{prefix}_probe_results.csv"
    if degrade_ab_df is not None:
        degrade_ab_path = out_dir / f"{prefix}_degrade_ab_summary.csv"
        degrade_ab_df.to_csv(degrade_ab_path, index=False)

    all_df.to_csv(out_scan, index=False)
    best_result["trades"].to_csv(out_trades, index=False)
    best_result["equity"].reset_index().to_csv(out_equity, index=False)
    write_visual_html(out_html, best_result["bars"], best_result["trades"], best_result["equity"])
    if isinstance(best_result.get("probe_results"), pd.DataFrame) and (not best_result["probe_results"].empty):
        best_result["probe_results"].to_csv(out_probe, index=False)

    longonly_outputs: Dict[str, Path] = {}
    longonly_best_row = None
    if args.compare_long_only and args.side_mode == "both":
        cfg_long = RunConfig(
            start_utc=cfg.start_utc,
            end_utc=cfg.end_utc,
            commission_pct=cfg.commission_pct,
            slippage_pct=cfg.slippage_pct,
            box_low=cfg.box_low,
            box_high=cfg.box_high,
            dyn_cfg=cfg.dyn_cfg,
            side_mode="long_only",
            clip_dynamic_to_manual=cfg.clip_dynamic_to_manual,
            trade_box_mode_default=cfg.trade_box_mode_default,
            trade_box_lookback_default=cfg.trade_box_lookback_default,
            trade_box_q_low_default=cfg.trade_box_q_low_default,
            trade_box_q_high_default=cfg.trade_box_q_high_default,
            trade_box_ema_len_default=cfg.trade_box_ema_len_default,
            trade_box_atr_mult_default=cfg.trade_box_atr_mult_default,
            risk_high_candidates=cfg.risk_high_candidates,
            probe_days=cfg.probe_days,
            probe_metric=cfg.probe_metric,
            probe_execution_mode=cfg.probe_execution_mode,
            fallback_no_signal_mode=cfg.fallback_no_signal_mode,
            probe_tie_eps=cfg.probe_tie_eps,
            probe_min_round_trips=cfg.probe_min_round_trips,
            probe_min_entry_fill_rate=cfg.probe_min_entry_fill_rate,
            probe_max_hard_stop_ratio=cfg.probe_max_hard_stop_ratio,
            early_stop_first_k_trades=cfg.early_stop_first_k_trades,
            early_stop_hard_stop_threshold=cfg.early_stop_hard_stop_threshold,
            early_stop_first_m_bars=cfg.early_stop_first_m_bars,
            enable_early_stop=cfg.enable_early_stop,
            entry_execution_mode=cfg.entry_execution_mode,
            maker_fill_prob=cfg.maker_fill_prob,
            maker_queue_delay_bars=cfg.maker_queue_delay_bars,
            seed=cfg.seed,
            min_round_trips=cfg.min_round_trips,
            pf_unreliable_penalty=cfg.pf_unreliable_penalty,
            degrade_on_trend=cfg.degrade_on_trend,
            trend_slope_thresh=cfg.trend_slope_thresh,
            atr_expand_thresh=cfg.atr_expand_thresh,
            degrade_risk_lookback_bars=cfg.degrade_risk_lookback_bars,
            enable_runner=cfg.enable_runner,
            runner_pct=cfg.runner_pct,
            runner_atr_mult=cfg.runner_atr_mult,
        )
        long_df, long_best_params, long_best_result, long_total_grid = run_scan(df, cfg_long, args.max_combos, args.scan_mode)
        longonly_best_row = pick_best_row(long_df)
        long_prefix = f"{prefix}_longonly"
        long_scan = out_dir / f"{long_prefix}_scan_all.csv"
        long_trades = out_dir / f"{long_prefix}_best_trades.csv"
        long_equity = out_dir / f"{long_prefix}_best_equity.csv"
        long_html = out_dir / f"{long_prefix}_best_visual.html"
        long_report = out_dir / f"{long_prefix}_report.md"
        long_probe = out_dir / f"{long_prefix}_probe_results.csv"
        long_df.to_csv(long_scan, index=False)
        long_best_result["trades"].to_csv(long_trades, index=False)
        long_best_result["equity"].reset_index().to_csv(long_equity, index=False)
        write_visual_html(long_html, long_best_result["bars"], long_best_result["trades"], long_best_result["equity"])
        if isinstance(long_best_result.get("probe_results"), pd.DataFrame) and (not long_best_result["probe_results"].empty):
            long_best_result["probe_results"].to_csv(long_probe, index=False)
        long_fallback_reason_dist: Dict[str, int] = {}
        if isinstance(long_best_result.get("probe_results"), pd.DataFrame) and (not long_best_result["probe_results"].empty):
            rr2 = long_best_result["probe_results"]["tie_reason"].fillna("")
            rr2 = rr2.loc[rr2 != ""]
            long_fallback_reason_dist = {str(k): int(v) for k, v in rr2.value_counts().to_dict().items()}
        long_params_summary = {
            "long_L": long_best_params.long_L,
            "mid1": long_best_params.mid1,
            "mid2": long_best_params.mid2,
            "layer_step": long_best_params.layer_step,
            "base_risk_pct": long_best_params.base_risk_pct,
            "hard_stop_pct": long_best_params.hard_stop_pct,
            "maker_ttl_bars": long_best_params.maker_ttl_bars,
            "min_improve_bps": long_best_params.min_improve_bps,
            "trade_box_mode": long_best_params.trade_box_mode,
        }
        long_top20 = long_df.head(20).copy()
        with long_report.open("w", encoding="utf-8") as f2:
            f2.write(f"# {long_prefix} Report\n\n")
            f2.write("## Setup\n")
            f2.write(f"- version: `{__version__}`\n")
            f2.write(f"- git_hash: `{__git_hash__}`\n")
            f2.write(f"- run_tag: `{args.round_tag}`\n")
            f2.write(f"- csv: `{csv_path}`\n")
            f2.write(f"- trade_window_utc: `{start_utc}` -> `{end_utc}`\n")
            f2.write(f"- box_low: `{BOX_LOW}`\n")
            f2.write(f"- box_high: `{BOX_HIGH}`\n")
            f2.write(f"- side_mode: `long_only`\n")
            f2.write(f"- scanned_combos: `{len(long_df)}` (from grid_total `{long_total_grid}`)\n")
            f2.write(f"- scan_mode: `{args.scan_mode}`\n")
            f2.write(f"- risk_high_candidates: `{cfg_long.risk_high_candidates}`\n")
            f2.write(f"- probe_days: `{cfg_long.probe_days}`\n")
            f2.write(f"- fallback_reason_distribution: `{long_fallback_reason_dist}`\n")
            f2.write(f"- params_summary: `{long_params_summary}`\n")
            f2.write(f"- probe_metric: `{cfg_long.probe_metric}`\n")
            f2.write(f"- probe_execution_mode: `{cfg_long.probe_execution_mode}`\n")
            f2.write(f"- fallback_no_signal_mode: `{cfg_long.fallback_no_signal_mode}`\n")
            f2.write(f"- probe_tie_eps: `{cfg_long.probe_tie_eps}`\n")
            f2.write(f"- probe_filter_round_trips_min: `{cfg_long.probe_min_round_trips}`\n")
            f2.write(f"- probe_filter_orders_total_min: `{PROBE_MIN_ORDERS_TOTAL}`\n")
            f2.write("- probe_filter_logic: `round_trips>=min OR orders_total>=min OR filled_entries>=1; no-signal-all -> fallback_no_signal_mode; else if all filters fail -> fallback_v2`\n")
            f2.write("- probe_fallback_v2_rank: `box_invalid_count asc, risk_expand_count asc, close_outside_ratio asc`\n")
            f2.write(f"- early_stop_first_k_trades: `{cfg_long.early_stop_first_k_trades}`\n")
            f2.write(f"- early_stop_hard_stop_threshold: `{cfg_long.early_stop_hard_stop_threshold}`\n")
            f2.write(f"- early_stop_first_m_bars: `{cfg_long.early_stop_first_m_bars}`\n")
            f2.write(f"- maker_fill_prob: `{cfg_long.maker_fill_prob}`\n")
            f2.write(f"- maker_queue_delay_bars: `{cfg_long.maker_queue_delay_bars}`\n")
            f2.write(f"- seed: `{cfg_long.seed}`\n")
            f2.write(f"- min_round_trips: `{cfg_long.min_round_trips}`\n")
            f2.write(f"- degrade_on_trend: `{cfg_long.degrade_on_trend}`\n")
            f2.write(f"- trend_slope_thresh: `{cfg_long.trend_slope_thresh}`\n")
            f2.write(f"- atr_expand_thresh: `{cfg_long.atr_expand_thresh}`\n")
            f2.write(f"- degrade_risk_lookback_bars: `{cfg_long.degrade_risk_lookback_bars}`\n")
            f2.write("\n## Dual Box Settings\n")
            f2.write("- risk_box: `dynamic/manual box for stop/invalidation/gating`\n")
            f2.write("- trading_box: `mode/lookback/shape for normalized x`\n")
            if isinstance(long_best_result.get("probe_results"), pd.DataFrame) and (not long_best_result["probe_results"].empty):
                f2.write("\n## Probe Selection\n")
                f2.write(f"- selected_risk_high: `{long_best_result['stats'].get('selected_risk_high', cfg_long.box_high)}`\n")
                f2.write(f"- probe_results_file: `{long_probe}`\n")
                f2.write(f"- probe_param_set_fixed: `{long_best_result['stats'].get('probe_param_set', '')}`\n")
                top3 = long_best_result["probe_results"].head(3).copy()
                f2.write(f"- metric_gap_to_2nd: `{top3['metric_gap_to_2nd'].iloc[0]}`\n")
                f2.write(top3.to_markdown(index=False))
                f2.write("\n")
            f2.write("\n## Best Params\n")
            for k, v in long_best_params.__dict__.items():
                f2.write(f"- {k}: `{v}`\n")
            f2.write("\n## Best Metrics\n")
            for k in [
                "return_pct",
                "max_dd_pct",
                "profit_factor",
                "expectancy_after_cost",
                "round_trips",
                "daily_orders",
                "orders_total",
                "pf_reliable",
                "filled_entries",
                "filled_exits",
                "effective_fill_rate",
                "hard_stop_ratio",
                "avg_layers",
                "layer_add_count",
                "layer_disabled_count_due_to_degrade",
                "degrade_mode_bars_ratio",
                "degrade_level1_ratio",
                "degrade_level2_ratio",
                "degrade_too_strong",
                "missed_entry",
                "missed_entry_rate",
                "entry_fill_rate",
                "avg_improve_vs_taker",
                "fill_prob_used",
                "queue_delay_used",
                "taker_fallback_count",
                "taker_fallback_ratio",
                "maker_fees",
                "taker_fees",
                "avg_cost_per_trade",
                "box_invalid_count",
                "risk_expand_count",
                "expand_count",
                "trade_box_invalid_count",
                "trade_box_invalid_ratio",
                "selected_risk_high",
                "probe_metric_gap_to_2nd",
                "probe_tie_reason",
                "probe_fallback_v2_used",
                "probe_passed_filters_count",
                "early_stop_due_to_bad_probe",
                "early_stop_reason",
                "early_stop_time",
                "objective",
            ]:
                v = long_best_result["stats"][k]
                if isinstance(v, float):
                    f2.write(f"- {k}: `{v:.6f}`\n")
                else:
                    f2.write(f"- {k}: `{v}`\n")
            if bool(long_best_result["stats"].get("early_stop_due_to_bad_probe", False)):
                f2.write("- EARLY_STOP_DUE_TO_BAD_PROBE: `True`\n")
            if bool(long_best_result["stats"].get("degrade_too_strong", False)):
                f2.write("- DEGRADE_TOO_STRONG: `True`\n")
            f2.write("\n## Top 20\n")
            f2.write(long_top20.to_markdown(index=False))
            f2.write("\n")
        longonly_outputs = {
            "scan": long_scan,
            "trades": long_trades,
            "equity": long_equity,
            "html": long_html,
            "report": long_report,
            "probe": long_probe,
        }

        compare_rows = [
            {
                "variant": "both",
                "side_mode": "both",
                "return_pct": float(best_result["stats"]["return_pct"]),
                "max_dd_pct": float(best_result["stats"]["max_dd_pct"]),
                "profit_factor": float(best_result["stats"]["profit_factor"]),
                "round_trips": int(best_result["stats"]["round_trips"]),
                "hard_stop_ratio": float(best_result["stats"]["hard_stop_ratio"]),
                "entry_fill_rate": float(best_result["stats"]["entry_fill_rate"]),
                "objective": float(best_result["stats"]["objective"]),
                "long_L": float(best_params.long_L),
                "mid1": float(best_params.mid1),
                "mid2": float(best_params.mid2),
                "layer_step": float(best_params.layer_step),
                "hard_stop_pct": float(best_params.hard_stop_pct),
                "trade_box_mode": str(best_params.trade_box_mode),
                "trade_box_lookback": int(best_params.trade_box_lookback),
                "selected_risk_high": float(best_result["stats"].get("selected_risk_high", cfg.box_high)),
            },
            {
                "variant": "long_only",
                "side_mode": "long_only",
                "return_pct": float(long_best_result["stats"]["return_pct"]),
                "max_dd_pct": float(long_best_result["stats"]["max_dd_pct"]),
                "profit_factor": float(long_best_result["stats"]["profit_factor"]),
                "round_trips": int(long_best_result["stats"]["round_trips"]),
                "hard_stop_ratio": float(long_best_result["stats"]["hard_stop_ratio"]),
                "entry_fill_rate": float(long_best_result["stats"]["entry_fill_rate"]),
                "objective": float(long_best_result["stats"]["objective"]),
                "long_L": float(long_best_params.long_L),
                "mid1": float(long_best_params.mid1),
                "mid2": float(long_best_params.mid2),
                "layer_step": float(long_best_params.layer_step),
                "hard_stop_pct": float(long_best_params.hard_stop_pct),
                "trade_box_mode": str(long_best_params.trade_box_mode),
                "trade_box_lookback": int(long_best_params.trade_box_lookback),
                "selected_risk_high": float(long_best_result["stats"].get("selected_risk_high", cfg_long.box_high)),
            },
        ]
        compare_df = pd.DataFrame(compare_rows)
        compare_path = out_dir / f"{prefix}_side_compare_summary.csv"
        compare_df.to_csv(compare_path, index=False)
        longonly_outputs["compare"] = compare_path

    robust_path = out_dir / "manual_box_robustness_summary.csv"
    robust_df = None
    if args.robustness_test:
        robust_params = resolve_robust_params(best_params, args)
        robust_df = run_robustness_test(df, cfg, robust_params, robust_path)

    validation_path = out_dir / f"{prefix}_probe_stability_summary.csv"
    validation_path_plain = out_dir / "probe_stability_summary.csv"
    validation_df = None
    run_validation = (not args.no_validation) or bool(args.run_validation_suite)
    run_walk = (not args.no_validation) or bool(args.walk_forward)
    if run_validation:
        validation_df = run_validation_suite(df, cfg, best_params, validation_path)
        validation_df.to_csv(validation_path_plain, index=False)

    walk_path = out_dir / f"{prefix}_walk_forward_summary.csv"
    walk_path_plain = out_dir / "walk_forward_summary.csv"
    walk_agg_path = out_dir / f"{prefix}_walk_forward_aggregate.md"
    walk_agg_path_plain = out_dir / "walk_forward_aggregate.md"
    walk_df = None
    walk_agg = None
    walk_artifacts: Dict[str, object] = {}
    if run_walk:
        walk_df = run_walk_forward(
            df=df,
            cfg=cfg,
            p=best_params,
            segment_days=max(int(args.segment_days), 1),
            step_days=max(int(args.step_days), 1),
            out_path=walk_path,
            max_combos=max(int(args.max_combos), 1),
            scan_mode=str(args.scan_mode),
        )
        walk_artifacts = dict(walk_df.attrs.get("wfa_artifacts", {}))
        walk_agg = write_walk_forward_aggregate(walk_agg_path, walk_df, walk_artifacts)
        walk_df.to_csv(walk_path_plain, index=False)
        walk_agg_path_plain.write_text(walk_agg_path.read_text(encoding="utf-8"), encoding="utf-8")

    release_gate_status = "SKIPPED"
    release_gate_not_robust = False
    release_gate_reasons: List[str] = []
    if walk_agg is not None:
        gate_fail_early_stop = walk_agg["early_stop_rate"] > float(args.release_gate_max_early_stop_rate)
        gate_fail_median_ret = walk_agg["median_return"] <= float(args.release_gate_min_median_return)
        gate_fail_median_pf = (not np.isfinite(walk_agg["median_pf"])) or (walk_agg["median_pf"] < float(args.release_gate_min_median_pf))
        if gate_fail_early_stop:
            release_gate_reasons.append("early_stop_rate")
        if gate_fail_median_ret:
            release_gate_reasons.append("median_return")
        if gate_fail_median_pf:
            release_gate_reasons.append("median_pf")
        release_gate_not_robust = bool(gate_fail_early_stop or gate_fail_median_ret or gate_fail_median_pf)
        release_gate_status = "NOT_ROBUST" if release_gate_not_robust else "ROBUST"

    fallback_reason_dist: Dict[str, int] = {}
    if isinstance(best_result.get("probe_results"), pd.DataFrame) and (not best_result["probe_results"].empty):
        rr = best_result["probe_results"]["tie_reason"].fillna("")
        rr = rr.loc[rr != ""]
        fallback_reason_dist = {str(k): int(v) for k, v in rr.value_counts().to_dict().items()}
    params_summary = {
        "long_L": best_params.long_L,
        "mid1": best_params.mid1,
        "mid2": best_params.mid2,
        "layer_step": best_params.layer_step,
        "base_risk_pct": best_params.base_risk_pct,
        "hard_stop_pct": best_params.hard_stop_pct,
        "maker_ttl_bars": best_params.maker_ttl_bars,
        "min_improve_bps": best_params.min_improve_bps,
        "trade_box_mode": best_params.trade_box_mode,
    }

    top20 = all_df.head(20).copy()
    with out_report.open("w", encoding="utf-8") as f:
        f.write(f"# {prefix} Report\n\n")
        f.write("## Setup\n")
        f.write(f"- version: `{__version__}`\n")
        f.write(f"- git_hash: `{__git_hash__}`\n")
        f.write(f"- run_tag: `{args.round_tag}`\n")
        f.write(f"- csv: `{csv_path}`\n")
        f.write(f"- trade_window_utc: `{start_utc}` -> `{end_utc}`\n")
        f.write(f"- box_source: `{cfg.box_source}`\n")
        f.write(f"- clip_dynamic_to_manual: `{cfg.clip_dynamic_to_manual}`\n")
        f.write(f"- macro_box_mode: `{cfg.macro_box_mode}`\n")
        f.write(f"- macro_lookback_bars: `{cfg.macro_lookback_bars}`\n")
        f.write(f"- macro_bb_len/std: `{cfg.macro_bb_len}` / `{cfg.macro_bb_std}`\n")
        f.write(f"- box_low: `{BOX_LOW}`\n")
        f.write(f"- box_high: `{BOX_HIGH}`\n")
        f.write(f"- side_mode: `{args.side_mode}`\n")
        f.write(f"- compare_long_only: `{args.compare_long_only}`\n")
        f.write(f"- ab_degrade_compare: `{args.ab_degrade_compare}`\n")
        f.write(f"- no_validation: `{args.no_validation}`\n")
        f.write(f"- run_validation: `{run_validation}`\n")
        f.write(f"- run_walk_forward: `{run_walk}`\n")
        f.write(f"- risk_high_candidates: `{cfg.risk_high_candidates}`\n")
        f.write(f"- probe_days: `{cfg.probe_days}`\n")
        f.write(f"- fallback_reason_distribution: `{fallback_reason_dist}`\n")
        f.write(f"- params_summary: `{params_summary}`\n")
        f.write(f"- probe_metric: `{cfg.probe_metric}`\n")
        f.write(f"- probe_execution_mode: `{cfg.probe_execution_mode}`\n")
        f.write(f"- fallback_no_signal_mode: `{cfg.fallback_no_signal_mode}`\n")
        f.write(f"- probe_tie_eps: `{cfg.probe_tie_eps}`\n")
        f.write(f"- probe_filter_round_trips_min: `{cfg.probe_min_round_trips}`\n")
        f.write(f"- probe_filter_orders_total_min: `{PROBE_MIN_ORDERS_TOTAL}`\n")
        f.write("- probe_filter_logic: `round_trips>=min OR orders_total>=min OR filled_entries>=1; no-signal-all -> fallback_no_signal_mode; else if all filters fail -> fallback_v2`\n")
        f.write("- probe_fallback_v2_rank: `box_invalid_count asc, risk_expand_count asc, close_outside_ratio asc`\n")
        f.write(f"- early_stop_first_k_trades: `{cfg.early_stop_first_k_trades}`\n")
        f.write(f"- early_stop_hard_stop_threshold: `{cfg.early_stop_hard_stop_threshold}`\n")
        f.write(f"- early_stop_first_m_bars: `{cfg.early_stop_first_m_bars}`\n")
        f.write(f"- maker_fill_prob: `{cfg.maker_fill_prob}`\n")
        f.write(f"- maker_queue_delay_bars: `{cfg.maker_queue_delay_bars}`\n")
        f.write(f"- seed: `{cfg.seed}`\n")
        f.write(f"- min_round_trips: `{cfg.min_round_trips}`\n")
        f.write(f"- degrade_on_trend: `{cfg.degrade_on_trend}`\n")
        f.write(f"- trend_slope_thresh: `{cfg.trend_slope_thresh}`\n")
        f.write(f"- atr_expand_thresh: `{cfg.atr_expand_thresh}`\n")
        f.write(f"- degrade_risk_lookback_bars: `{cfg.degrade_risk_lookback_bars}`\n")
        f.write(f"- regime_gate_mode: `{cfg.regime_gate_mode}`\n")
        f.write(f"- regime_gate_adx_thresh: `{cfg.regime_gate_adx_thresh}`\n")
        f.write(f"- regime_gate_bbwidth_min: `{cfg.regime_gate_bbwidth_min}`\n")
        f.write(f"- regime_gate_chop_thresh: `{cfg.regime_gate_chop_thresh}`\n")
        f.write(f"- structural_cooldown_bars: `{cfg.structural_cooldown_bars}`\n")
        f.write(f"- enable_runner: `{cfg.enable_runner}`\n")
        f.write(f"- runner_pct: `{cfg.runner_pct}`\n")
        f.write(f"- runner_atr_mult: `{cfg.runner_atr_mult}`\n")
        f.write(f"- scanned_combos: `{len(all_df)}` (from grid_total `{total_grid}`)\n")
        f.write(f"- scan_mode: `{args.scan_mode}`\n")
        f.write("\n## Fixed Defaults\n")
        f.write("- local_scan_fixed: `layer_weights=(1.0,1.4,2.0), max_leverage=2.0, cooldown_rearm=1, cooldown_stop=24, time_stop=30, maker_fallback_taker=True, improve_mult=0.10`\n")
        f.write("\n## Dynamic Box Settings\n")
        f.write(f"- dynamic_box_mode: `{dyn_cfg.mode}`\n")
        f.write(f"- break_confirm_closes: `{dyn_cfg.break_confirm_closes}`\n")
        f.write(f"- break_buffer_mode: `{dyn_cfg.break_buffer_mode}`\n")
        f.write(f"- break_buffer_pct: `{dyn_cfg.break_buffer_pct}`\n")
        f.write(f"- break_buffer_atr_mult: `{dyn_cfg.break_buffer_atr_mult}`\n")
        f.write(f"- expand_atr_mult: `{dyn_cfg.expand_atr_mult}`\n")
        f.write(f"- invalidate_force_close: `{dyn_cfg.invalidate_force_close}`\n")
        f.write(f"- max_expands: `{dyn_cfg.max_expands}`\n")
        f.write(f"- freeze_after_expand_bars: `{dyn_cfg.freeze_after_expand_bars}`\n")
        f.write("\n## Dual Box Settings\n")
        f.write("- risk_box: `manual + dynamic expand/invalidate, used for hard stop and invalidation`\n")
        f.write(f"- trade_box_mode_default: `{cfg.trade_box_mode_default}`\n")
        f.write(f"- trade_box_lookback_default: `{cfg.trade_box_lookback_default}`\n")
        f.write(f"- trade_box_q_low_default: `{cfg.trade_box_q_low_default}`\n")
        f.write(f"- trade_box_q_high_default: `{cfg.trade_box_q_high_default}`\n")
        f.write(f"- trade_box_ema_len_default: `{cfg.trade_box_ema_len_default}`\n")
        f.write(f"- trade_box_atr_mult_default: `{cfg.trade_box_atr_mult_default}`\n")
        if isinstance(best_result.get("probe_results"), pd.DataFrame) and (not best_result["probe_results"].empty):
            f.write("\n## Probe Selection\n")
            f.write(f"- selected_risk_high: `{best_result['stats'].get('selected_risk_high', cfg.box_high)}`\n")
            f.write(f"- probe_results_file: `{out_probe}`\n")
            f.write(f"- probe_param_set_fixed: `{best_result['stats'].get('probe_param_set', '')}`\n")
            top3 = best_result["probe_results"].head(3).copy()
            f.write(f"- metric_gap_to_2nd: `{top3['metric_gap_to_2nd'].iloc[0]}`\n")
            f.write(top3.to_markdown(index=False))
            f.write("\n")
        if out_warn:
            f.write(f"- output_warning: `{out_warn}`\n")
        f.write("\n## Best Params\n")
        for k, v in best_params.__dict__.items():
            f.write(f"- {k}: `{v}`\n")
        if longonly_best_row is not None:
            f.write("\n## Long-Only Comparison\n")
            f.write(f"- long_only_scan: `{longonly_outputs.get('scan')}`\n")
            f.write(f"- long_only_report: `{longonly_outputs.get('report')}`\n")
            f.write(f"- long_only_probe: `{longonly_outputs.get('probe')}`\n")
            f.write(f"- side_compare_summary: `{longonly_outputs.get('compare')}`\n")
            f.write("- both_objective: `{:.6f}`\n".format(float(best_result["stats"]["objective"])))
            f.write("- long_only_objective: `{:.6f}`\n".format(float(longonly_best_row["objective"])))
            f.write("- both_return_pct: `{:.6f}`\n".format(float(best_result["stats"]["return_pct"])))
            f.write("- long_only_return_pct: `{:.6f}`\n".format(float(longonly_best_row["return_pct"])))
            f.write("- both_max_dd_pct: `{:.6f}`\n".format(float(best_result["stats"]["max_dd_pct"])))
            f.write("- long_only_max_dd_pct: `{:.6f}`\n".format(float(longonly_best_row["max_dd_pct"])))
        f.write("\n## Best Metrics\n")
        for k in [
            "return_pct",
            "max_dd_pct",
            "profit_factor",
            "expectancy_after_cost",
            "round_trips",
            "daily_orders",
            "orders_total",
            "pf_reliable",
            "filled_entries",
            "filled_exits",
            "effective_fill_rate",
            "hard_stop_ratio",
            "atr_stop_count",
            "time_stop_count",
            "runner_stop_count",
            "avg_layers",
            "layer_add_count",
            "layer_disabled_count_due_to_degrade",
            "degrade_mode_bars_ratio",
            "degrade_level1_ratio",
            "degrade_level2_ratio",
            "degrade_too_strong",
            "missed_entry",
            "missed_entry_rate",
            "entry_fill_rate",
            "avg_improve_vs_taker",
            "fill_prob_used",
            "queue_delay_used",
            "taker_fallback_count",
            "taker_fallback_ratio",
            "maker_fees",
            "taker_fees",
            "avg_cost_per_trade",
            "box_invalid_count",
            "risk_expand_count",
            "expand_count",
            "structural_break_count",
            "structural_cooldown_bars_ratio",
            "trade_box_invalid_count",
            "trade_box_invalid_ratio",
            "start_gate_pass_rate",
            "regime_start_gate_pass_rate",
            "cb_trigger_count",
            "cb_disable_bars_ratio",
            "selected_risk_high",
            "probe_metric_gap_to_2nd",
            "probe_tie_reason",
            "probe_fallback_v2_used",
            "probe_passed_filters_count",
            "early_stop_due_to_bad_probe",
            "early_stop_reason",
            "early_stop_time",
            "first_box_low",
            "first_box_high",
            "final_box_low",
            "final_box_high",
            "box_inactive_time",
            "hard_stop_after_expand6_count",
            "hard_stop_after_expand6_hit",
            "objective",
        ]:
            v = best_result["stats"][k]
            if isinstance(v, float):
                f.write(f"- {k}: `{v:.6f}`\n")
            else:
                f.write(f"- {k}: `{v}`\n")
        if bool(best_result["stats"].get("early_stop_due_to_bad_probe", False)):
            f.write("- EARLY_STOP_DUE_TO_BAD_PROBE: `True`\n")
        if bool(best_result["stats"].get("degrade_too_strong", False)):
            f.write("- DEGRADE_TOO_STRONG: `True`\n")
        f.write("\n## Constraints\n")
        f.write("- objective: `return - 0.6*dd - 0.2*missed_entry_rate*100`\n")
        f.write(f"- feasible iff `round_trips>={cfg.min_round_trips} && profit_factor>=1.15 && hard_stop_ratio<=0.45 && entry_fill_rate>=0.85`\n")
        f.write(f"- feasible_count: `{int(all_df['feasible'].sum())}`\n")
        if robust_df is not None:
            f.write("\n## Robustness\n")
            f.write(f"- robustness_file: `{robust_path}`\n")
            f.write(f"- robustness_rows: `{len(robust_df)}`\n")
            if "robust_group" in robust_df.columns:
                grp = robust_df["robust_group"].value_counts().to_dict()
                f.write(f"- robustness_groups: `{grp}`\n")
            f.write(robust_df.to_markdown(index=False))
            f.write("\n")
        if degrade_ab_df is not None:
            f.write("\n## Degrade A/B\n")
            if degrade_ab_path is not None:
                f.write(f"- degrade_ab_file: `{degrade_ab_path}`\n")
            f.write(degrade_ab_df.to_markdown(index=False))
            f.write("\n")
        if validation_df is not None:
            f.write("\n## Probe Stability Suite\n")
            f.write(f"- probe_stability_file: `{validation_path}`\n")
            f.write(validation_df.to_markdown(index=False))
            f.write("\n")
        if walk_df is not None:
            f.write("\n## Walk Forward\n")
            f.write(f"- walk_forward_file: `{walk_path}`\n")
            f.write(f"- walk_forward_aggregate_file: `{walk_agg_path}`\n")
            for key in ["combined_equity_path", "combined_trades_path", "combined_bars_path", "combined_visual_path"]:
                if walk_artifacts.get(key):
                    f.write(f"- {key}: `{walk_artifacts[key]}`\n")
            f.write(f"- segment_days: `{max(int(args.segment_days), 1)}`\n")
            f.write(f"- step_days: `{max(int(args.step_days), 1)}`\n")
            f.write(walk_df.to_markdown(index=False))
            f.write("\n")
        if walk_agg is not None:
            f.write("\n## Release Gate\n")
            f.write(f"- release_gate_status: `{release_gate_status}`\n")
            f.write(f"- threshold_max_early_stop_rate: `{float(args.release_gate_max_early_stop_rate)}`\n")
            f.write(f"- threshold_min_median_return: `{float(args.release_gate_min_median_return)}`\n")
            f.write(f"- threshold_min_median_pf: `{float(args.release_gate_min_median_pf)}`\n")
            f.write(f"- release_gate_reasons: `{release_gate_reasons}`\n")
            f.write(f"- early_stop_rate: `{walk_agg['early_stop_rate']:.6f}`\n")
            f.write(f"- mean_return: `{walk_agg['mean_return']:.6f}`\n")
            f.write(f"- median_return: `{walk_agg['median_return']:.6f}`\n")
            f.write(f"- mean_pf: `{walk_agg['mean_pf']:.6f}`\n")
            f.write(f"- median_pf: `{walk_agg['median_pf']:.6f}`\n")
            f.write(f"- pf_reliable_rate: `{walk_agg['pf_reliable_rate']:.6f}`\n")
            if pd.notna(walk_agg.get("combined_return_pct", np.nan)):
                f.write(f"- combined_return_pct: `{walk_agg['combined_return_pct']:.6f}`\n")
            if pd.notna(walk_agg.get("combined_final_equity", np.nan)):
                f.write(f"- combined_final_equity: `{walk_agg['combined_final_equity']:.6f}`\n")
            if release_gate_not_robust:
                f.write("- NOT_ROBUST: `True`\n")
        f.write("\n## Top 20\n")
        f.write(top20.to_markdown(index=False))
        f.write("\n")

    archive_zip_path = None
    if args.archive_run:
        archive_files: List[Path] = [Path("manual_box_roundX.py"), out_scan, out_trades, out_equity, out_html, out_report]
        if out_probe.exists():
            archive_files.append(out_probe)
        if robust_df is not None:
            archive_files.append(robust_path)
        if degrade_ab_df is not None and degrade_ab_path is not None:
            archive_files.append(degrade_ab_path)
        if validation_df is not None:
            archive_files.extend([validation_path, validation_path_plain])
        if walk_df is not None:
            archive_files.extend([walk_path, walk_path_plain])
            for key in ["combined_equity_path", "combined_trades_path", "combined_bars_path", "combined_visual_path"]:
                if walk_artifacts.get(key):
                    archive_files.append(Path(str(walk_artifacts[key])))
        if walk_agg is not None:
            archive_files.extend([walk_agg_path, walk_agg_path_plain])
        for k in ["scan", "trades", "equity", "html", "report", "probe", "compare"]:
            if k in longonly_outputs:
                archive_files.append(longonly_outputs[k])
        manifest = build_archive_manifest(
            args=args,
            run_tag=args.round_tag,
            cfg=cfg,
            best_params=best_params,
            best_stats=best_result["stats"],
            release_gate_status=release_gate_status,
            release_gate_reasons=release_gate_reasons,
            files=archive_files,
        )
        archive_zip_path = create_archive_zip(out_dir, args.round_tag, manifest, archive_files)

    print("done")
    print(f"- {out_scan}")
    print(f"- {out_trades}")
    print(f"- {out_equity}")
    print(f"- {out_html}")
    print(f"- {out_report}")
    if out_probe.exists():
        print(f"- {out_probe}")
    for key in ["scan", "trades", "equity", "html", "report", "probe", "compare"]:
        if key in longonly_outputs:
            print(f"- {longonly_outputs[key]}")
    if robust_df is not None:
        print(f"- {robust_path}")
    if degrade_ab_df is not None and degrade_ab_path is not None:
        print(f"- {degrade_ab_path}")
    if validation_df is not None:
        print(f"- {validation_path}")
        print(f"- {validation_path_plain}")
    if walk_df is not None:
        print(f"- {walk_path}")
        print(f"- {walk_path_plain}")
        for key in ["combined_equity_path", "combined_trades_path", "combined_bars_path", "combined_visual_path"]:
            if walk_artifacts.get(key):
                print(f"- {walk_artifacts[key]}")
    if walk_agg is not None:
        print(f"- {walk_agg_path}")
        print(f"- {walk_agg_path_plain}")
    if archive_zip_path is not None:
        print(f"- {archive_zip_path}")
    if args.open_html:
        try:
            subprocess.run(["open", str(out_html)], check=False)
            print(f"opened: {out_html}")
            if "html" in longonly_outputs:
                subprocess.run(["open", str(longonly_outputs["html"])], check=False)
                print(f"opened: {longonly_outputs['html']}")
        except Exception as exc:  # noqa: BLE001
            print(f"open failed: {exc}")


if __name__ == "__main__":
    main()
