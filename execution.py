"""Execution engines for event-first range trading signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class ExecutionConfig:
    width_pct_min: float = 0.001
    width_pct_max: float = 0.06
    method_agreement_min: float = 0.34
    transition_risk_max: float = 0.65
    confidence_min_false_break: float = 0.45
    confidence_min_boundary: float = 0.50
    side_cooldown_bars_false_break: int = 4
    side_cooldown_bars_boundary: int = 6
    allow_states: tuple[str, ...] = ("STABLE", "SQUEEZE")
    false_break_regime_mode: str = "down_up"  # all | down_only | down_up
    regime_lookback_hours: int = 24 * 30
    regime_down_low: float = -0.20
    regime_down_high: float = -0.05
    regime_up_low: float = 0.05
    regime_up_high: float = 0.20
    false_break_up_long_only: bool = False
    allow_warmup_trades: bool = True
    bad_width_bucket_filter_enabled: bool = False
    bad_width_bucket_min_pct: float = 0.010
    bad_width_bucket_max_pct: float = 0.015
    expected_value_filter_enabled: bool = False
    expected_value_mode: str = "to_midline"  # to_midline | to_opposite_edge
    min_reward_to_cost_ratio: float = 1.5
    expected_value_round_trip_cost_rate: float | None = None
    capture_recovery_exec_conf_low_enabled: bool = False
    capture_recovery_exec_conf_low_min_reward_to_cost: float = 3.0
    capture_recovery_exec_conf_low_min_reentry_strength: float = 0.25
    capture_recovery_confirm_near_miss_enabled: bool = False
    capture_recovery_confirm_near_miss_exact_base_confirms: int = 1
    capture_recovery_confirm_near_miss_min_box_confidence: float = 0.98
    capture_recovery_confirm_near_miss_min_h4_method: float = 0.75
    capture_recovery_confirm_near_miss_max_h4_transition: float = 0.08
    capture_recovery_confirm_near_miss_min_reward_to_cost: float = 2.5
    capture_recovery_confirm_near_miss_min_reentry_strength: float = 0.15

    def normalized_regime_mode(self) -> str:
        mode = str(self.false_break_regime_mode).strip().lower()
        if mode in {"all", "down_only", "down_up"}:
            return mode
        return "down_up"

    def normalized_expected_value_mode(self) -> str:
        mode = str(self.expected_value_mode).strip().lower()
        if mode in {"to_midline", "to_opposite_edge"}:
            return mode
        return "to_midline"


def _label_false_break_regime(ret_val: float, cfg: ExecutionConfig) -> str:
    if not np.isfinite(ret_val):
        return "warmup"
    if ret_val <= cfg.regime_down_low:
        return "deep_down"
    if cfg.regime_down_low < ret_val <= cfg.regime_down_high:
        return "down"
    if cfg.regime_down_high < ret_val < cfg.regime_up_low:
        return "flat"
    if cfg.regime_up_low <= ret_val < cfg.regime_up_high:
        return "up"
    return "strong_up"


def _false_break_regime_gate(mode: str, regime: str) -> bool:
    if mode == "all":
        return True
    if regime in {"warmup", "unknown"}:
        return True
    if mode == "down_only":
        return regime == "down"
    if mode == "down_up":
        return regime in {"down", "up"}
    return True


def _expected_target_price(raw_signal: int, row: pd.Series, mode: str) -> float:
    if raw_signal > 0:
        if mode == "to_midline":
            return float(row.get("box_midline", np.nan))
        return float(row.get("box_upper_edge", np.nan))
    if raw_signal < 0:
        if mode == "to_midline":
            return float(row.get("box_midline", np.nan))
        return float(row.get("box_lower_edge", np.nan))
    return np.nan


def _expected_value_metrics(raw_signal: int, row: pd.Series, cfg: ExecutionConfig) -> tuple[float, float, float, float, bool]:
    mode = cfg.normalized_expected_value_mode()
    price = float(row.get("close", row.get("price", np.nan)))
    target_price = _expected_target_price(raw_signal, row, mode)

    if raw_signal == 0 or not np.isfinite(price) or price <= 0.0 or not np.isfinite(target_price):
        reward_rate = np.nan
    else:
        reward_rate = raw_signal * (target_price / price - 1.0)

    cost_rate = float(cfg.expected_value_round_trip_cost_rate or 0.0)
    if np.isfinite(reward_rate) and cost_rate > 0.0:
        reward_to_cost = reward_rate / cost_rate
    elif np.isfinite(reward_rate) and cost_rate <= 0.0 and reward_rate > 0.0:
        reward_to_cost = np.inf
    else:
        reward_to_cost = np.nan

    if not cfg.expected_value_filter_enabled:
        gate = True
    elif not np.isfinite(reward_rate) or reward_rate <= 0.0:
        gate = False
    elif cost_rate <= 0.0:
        gate = True
    else:
        gate = bool(reward_to_cost >= cfg.min_reward_to_cost_ratio)

    return target_price, reward_rate, cost_rate, reward_to_cost, gate


def _false_break_reentry_side(row: pd.Series) -> int:
    reenter_up = bool(row.get("fb_reenter_from_up", False))
    reenter_dn = bool(row.get("fb_reenter_from_down", False))
    if reenter_dn and not reenter_up:
        return 1
    if reenter_up and not reenter_dn:
        return -1
    return 0


def _false_break_reentry_strength(signal: int, row: pd.Series) -> float:
    price = float(row.get("close", row.get("price", np.nan)))
    mid = float(row.get("box_midline", np.nan))
    lower = float(row.get("box_lower_edge", np.nan))
    upper = float(row.get("box_upper_edge", np.nan))

    if not np.isfinite(price) or not np.isfinite(mid):
        return np.nan
    if signal > 0 and np.isfinite(lower) and mid > lower:
        return float(np.clip((price - lower) / (mid - lower), 0.0, np.inf))
    if signal < 0 and np.isfinite(upper) and upper > mid:
        return float(np.clip((upper - price) / (upper - mid), 0.0, np.inf))
    return np.nan


def _false_break_confirm_near_miss_signal(raw_signal: int, row: pd.Series, cfg: ExecutionConfig) -> int:
    if raw_signal != 0 or not cfg.capture_recovery_confirm_near_miss_enabled:
        return 0

    side = _false_break_reentry_side(row)
    base_confirms = float(row.get("event_false_break_base_confirms", np.nan))
    if side == 0 or not np.isfinite(base_confirms):
        return 0
    return side if int(base_confirms) == int(cfg.capture_recovery_confirm_near_miss_exact_base_confirms) else 0


def _exec_conf_low_recovery_ok(
    raw_signal: int,
    reasons: List[str],
    reward_to_cost: float,
    reentry_strength: float,
    cfg: ExecutionConfig,
) -> bool:
    if not cfg.capture_recovery_exec_conf_low_enabled or raw_signal == 0:
        return False
    if reasons != ["event_confidence_low"]:
        return False
    if pd.isna(reward_to_cost) or reward_to_cost < cfg.capture_recovery_exec_conf_low_min_reward_to_cost:
        return False
    if not np.isfinite(reentry_strength) or reentry_strength < cfg.capture_recovery_exec_conf_low_min_reentry_strength:
        return False
    return True


def _confirm_near_miss_recovery_ok(
    recovery_signal: int,
    row: pd.Series,
    reward_to_cost: float,
    reentry_strength: float,
    cfg: ExecutionConfig,
) -> bool:
    if not cfg.capture_recovery_confirm_near_miss_enabled or recovery_signal == 0:
        return False

    base_confirms = float(row.get("event_false_break_base_confirms", np.nan))
    box_confidence = float(row.get("box_confidence", np.nan))
    h4_method = float(row.get("h4_method_agreement", np.nan))
    h4_transition = float(row.get("h4_transition_risk", np.nan))

    if not np.isfinite(base_confirms) or int(base_confirms) != int(cfg.capture_recovery_confirm_near_miss_exact_base_confirms):
        return False
    if not np.isfinite(box_confidence) or box_confidence < cfg.capture_recovery_confirm_near_miss_min_box_confidence:
        return False
    if not np.isfinite(h4_method) or h4_method < cfg.capture_recovery_confirm_near_miss_min_h4_method:
        return False
    if not np.isfinite(h4_transition) or h4_transition > cfg.capture_recovery_confirm_near_miss_max_h4_transition:
        return False
    if pd.isna(reward_to_cost) or reward_to_cost < cfg.capture_recovery_confirm_near_miss_min_reward_to_cost:
        return False
    if not np.isfinite(reentry_strength) or reentry_strength < cfg.capture_recovery_confirm_near_miss_min_reentry_strength:
        return False
    return True


def align_4h_to_1h(
    bars_1h: pd.DataFrame,
    boxes_4h: pd.DataFrame,
    h4_bar_duration: str = "4h",
) -> pd.DataFrame:
    """Attach 4H gating only after 4H bar close (causal merge_asof)."""
    left = bars_1h.copy().sort_values("timestamp").reset_index(drop=True)
    right = boxes_4h.copy().sort_values("timestamp").reset_index(drop=True)

    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True)

    right["h4_effective_time"] = right["timestamp"] + pd.Timedelta(h4_bar_duration)
    keep_cols = [
        "h4_effective_time",
        "range_usable",
        "box_state",
        "box_width_pct",
        "method_agreement",
        "transition_risk",
        "box_confidence",
        "box_valid",
    ]
    present = [c for c in keep_cols if c in right.columns]
    right = right[present].copy()

    rename = {
        "range_usable": "h4_range_usable",
        "box_state": "h4_box_state",
        "box_width_pct": "h4_box_width_pct",
        "method_agreement": "h4_method_agreement",
        "transition_risk": "h4_transition_risk",
        "box_confidence": "h4_box_confidence",
        "box_valid": "h4_box_valid",
    }
    right = right.rename(columns=rename)

    merged = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("h4_effective_time"),
        left_on="timestamp",
        right_on="h4_effective_time",
        direction="backward",
    )

    merged["h4_range_usable"] = merged["h4_range_usable"].where(merged["h4_range_usable"].notna(), False).astype(bool)
    merged["h4_box_valid"] = merged["h4_box_valid"].where(merged["h4_box_valid"].notna(), False).astype(bool)
    merged["h4_box_state"] = merged["h4_box_state"].fillna("INVALID")
    merged["h4_box_width_pct"] = pd.to_numeric(merged["h4_box_width_pct"], errors="coerce")
    merged["h4_method_agreement"] = pd.to_numeric(merged["h4_method_agreement"], errors="coerce").fillna(0.0)
    merged["h4_transition_risk"] = pd.to_numeric(merged["h4_transition_risk"], errors="coerce").fillna(1.0)
    merged["h4_box_confidence"] = pd.to_numeric(merged["h4_box_confidence"], errors="coerce").fillna(0.0)

    return merged.drop(columns=["h4_effective_time"], errors="ignore")


def _run_single_engine(
    df: pd.DataFrame,
    signal_col: str,
    confidence_col: str,
    engine_name: str,
    cooldown_bars: int,
    confidence_min: float,
    cfg: ExecutionConfig,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    n = len(out)

    rows: List[Dict[str, object]] = []
    last_side_entry = {1: -10_000_000, -1: -10_000_000}
    active_side = 0
    active_until = -10_000_000
    regime_mode = cfg.normalized_regime_mode()
    expected_value_mode = cfg.normalized_expected_value_mode()

    for i in range(n):
        if i >= active_until:
            active_side = 0

        raw_signal = int(np.sign(out.at[i, signal_col])) if signal_col in out.columns else 0
        recovery_candidate_signal = 0
        if engine_name == "false_break":
            recovery_candidate_signal = _false_break_confirm_near_miss_signal(raw_signal, out.iloc[i], cfg)
        effective_signal_input = raw_signal if raw_signal != 0 else recovery_candidate_signal
        event_conf = float(out.at[i, confidence_col]) if confidence_col in out.columns else 0.0
        gate_range = bool(out.at[i, "h4_range_usable"]) if "h4_range_usable" in out.columns else False
        gate_state = str(out.at[i, "h4_box_state"]) in cfg.allow_states if "h4_box_state" in out.columns else False
        gate_width = bool(
            pd.notna(out.at[i, "h4_box_width_pct"])
            and cfg.width_pct_min <= float(out.at[i, "h4_box_width_pct"]) <= cfg.width_pct_max
        )
        gate_method = float(out.at[i, "h4_method_agreement"]) >= cfg.method_agreement_min
        gate_transition = float(out.at[i, "h4_transition_risk"]) <= cfg.transition_risk_max
        gate_conf = event_conf >= confidence_min

        fb_regime = str(out.at[i, "fb_regime"]) if "fb_regime" in out.columns else "unknown"
        fb_regime_ret = float(out.at[i, "fb_regime_ret"]) if "fb_regime_ret" in out.columns else np.nan

        if engine_name == "false_break" and "fb_regime" in out.columns:
            gate_fb_regime = _false_break_regime_gate(regime_mode, fb_regime)
            gate_fb_side_bias = not (cfg.false_break_up_long_only and fb_regime == "up" and effective_signal_input < 0)
            gate_allow_warmup = cfg.allow_warmup_trades or fb_regime != "warmup"
        else:
            gate_fb_regime = True
            gate_fb_side_bias = True
            gate_allow_warmup = True

        box_width_pct = float(out.at[i, "box_width_pct"]) if "box_width_pct" in out.columns and pd.notna(out.at[i, "box_width_pct"]) else np.nan
        gate_bad_width_bucket = not (
            engine_name == "false_break"
            and cfg.bad_width_bucket_filter_enabled
            and np.isfinite(box_width_pct)
            and cfg.bad_width_bucket_min_pct <= box_width_pct < cfg.bad_width_bucket_max_pct
        )

        target_price, reward_rate, cost_rate, reward_to_cost, gate_expected_value = _expected_value_metrics(
            effective_signal_input, out.iloc[i], cfg
        )
        reentry_strength = _false_break_reentry_strength(effective_signal_input, out.iloc[i]) if engine_name == "false_break" else np.nan

        reasons: List[str] = []
        if effective_signal_input != 0 and not gate_range:
            reasons.append("h4_range_unusable")
        if effective_signal_input != 0 and not gate_state:
            reasons.append("h4_state_blocked")
        if effective_signal_input != 0 and not gate_width:
            reasons.append("h4_width_blocked")
        if effective_signal_input != 0 and not gate_method:
            reasons.append("h4_method_blocked")
        if effective_signal_input != 0 and not gate_transition:
            reasons.append("h4_transition_blocked")
        if effective_signal_input != 0 and not gate_conf:
            reasons.append("event_confidence_low")
        if effective_signal_input != 0 and engine_name == "false_break" and not gate_fb_regime:
            reasons.append("regime_blocked")
        if effective_signal_input != 0 and engine_name == "false_break" and not gate_fb_side_bias:
            reasons.append("up_long_only")
        if effective_signal_input != 0 and engine_name == "false_break" and not gate_allow_warmup:
            reasons.append("warmup_blocked")
        if effective_signal_input != 0 and engine_name == "false_break" and not gate_bad_width_bucket:
            reasons.append("bad_width_bucket_blocked")
        if effective_signal_input != 0 and cfg.expected_value_filter_enabled and not gate_expected_value:
            reasons.append("expected_value_blocked")

        if effective_signal_input != 0 and i - last_side_entry[effective_signal_input] < cooldown_bars:
            reasons.append("side_cooldown")

        if effective_signal_input != 0 and i < active_until:
            reasons.append("overlap_open_position")

        if effective_signal_input != 0 and active_side != 0 and active_side != effective_signal_input and event_conf < (confidence_min + 0.15):
            reasons.append("opposite_overlap_blocked")

        capture_recovery_reason = ""
        if engine_name == "false_break" and "event_confidence_low" in reasons:
            if _exec_conf_low_recovery_ok(raw_signal, reasons, reward_to_cost, reentry_strength, cfg):
                reasons = [r for r in reasons if r != "event_confidence_low"]
                capture_recovery_reason = "exec_conf_low"
            elif raw_signal == 0 and _confirm_near_miss_recovery_ok(
                recovery_candidate_signal, out.iloc[i], reward_to_cost, reentry_strength, cfg
            ):
                reasons = [r for r in reasons if r != "event_confidence_low"]
                capture_recovery_reason = "confirm_near_miss"

        gate_pass = effective_signal_input != 0 and len(reasons) == 0
        exec_signal = effective_signal_input if gate_pass else 0

        if gate_pass and effective_signal_input != 0:
            active_side = effective_signal_input
            active_until = i + max(1, cooldown_bars)
            last_side_entry[effective_signal_input] = i

        rows.append(
            {
                "timestamp": out.at[i, "timestamp"],
                "engine": engine_name,
                "raw_signal": raw_signal,
                "capture_recovery_candidate_signal": recovery_candidate_signal,
                "effective_signal_input": effective_signal_input,
                "capture_recovery_used": bool(capture_recovery_reason),
                "capture_recovery_reason": capture_recovery_reason,
                "event_confidence": event_conf,
                "fb_regime": fb_regime,
                "fb_regime_ret": fb_regime_ret,
                "gate_fb_regime": gate_fb_regime,
                "gate_fb_side_bias": gate_fb_side_bias,
                "gate_allow_warmup": gate_allow_warmup,
                "gate_bad_width_bucket": gate_bad_width_bucket,
                "gate_h4_range_usable": gate_range,
                "gate_h4_state": gate_state,
                "gate_h4_width": gate_width,
                "gate_h4_method": gate_method,
                "gate_h4_transition": gate_transition,
                "gate_event_confidence": gate_conf,
                "gate_expected_value": gate_expected_value,
                "expected_value_mode": expected_value_mode,
                "expected_target_price": target_price,
                "expected_reward_rate": reward_rate,
                "expected_cost_rate": cost_rate,
                "reward_to_cost_ratio": reward_to_cost,
                "reentry_strength": reentry_strength,
                "box_width_pct": box_width_pct,
                "event_base_confirms": float(out.at[i, "event_false_break_base_confirms"]) if "event_false_break_base_confirms" in out.columns and pd.notna(out.at[i, "event_false_break_base_confirms"]) else np.nan,
                "entry_warmup": bool(fb_regime == "warmup"),
                "gate_pass": gate_pass,
                "blocked_reason": "|".join(reasons) if reasons else "",
                "exec_signal": exec_signal,
                "side": "long" if exec_signal > 0 else "short" if exec_signal < 0 else "none",
                "price": out.at[i, "close"],
                "box_midline": out.at[i, "box_midline"],
                "box_upper_edge": out.at[i, "box_upper_edge"],
                "box_lower_edge": out.at[i, "box_lower_edge"],
                "box_width": out.at[i, "box_width"],
                "box_state": out.at[i, "box_state"],
                "h4_box_state": out.at[i, "h4_box_state"] if "h4_box_state" in out.columns else "INVALID",
            }
        )

    return pd.DataFrame(rows)


def run_execution_engines(df: pd.DataFrame, cfg: ExecutionConfig) -> Dict[str, pd.DataFrame]:
    """Run boundary reversal and false-break reversal engines."""
    base = df.copy()
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)

    lookback = max(int(cfg.regime_lookback_hours), 1)
    base["fb_regime_ret"] = base["close"].pct_change(lookback, fill_method=None)
    base["fb_regime"] = base["fb_regime_ret"].apply(lambda x: _label_false_break_regime(float(x), cfg))

    false_break = _run_single_engine(
        base,
        signal_col="event_false_break_signal",
        confidence_col="event_false_break_confidence",
        engine_name="false_break",
        cooldown_bars=cfg.side_cooldown_bars_false_break,
        confidence_min=cfg.confidence_min_false_break,
        cfg=cfg,
    )

    boundary = _run_single_engine(
        base,
        signal_col="event_box_init_signal",
        confidence_col="event_box_init_confidence",
        engine_name="boundary",
        cooldown_bars=cfg.side_cooldown_bars_boundary,
        confidence_min=cfg.confidence_min_boundary,
        cfg=cfg,
    )

    combined = pd.concat([false_break, boundary], ignore_index=True).sort_values(["timestamp", "engine"])
    combined = combined.reset_index(drop=True)

    return {
        "false_break": false_break,
        "boundary": boundary,
        "combined": combined,
    }


def export_execution_tables(execution_outputs: Dict[str, pd.DataFrame], out_dir: str) -> None:
    for name, df in execution_outputs.items():
        df.to_csv(f"{out_dir}/execution_{name}.csv", index=False)


def _state_color(state: str) -> str:
    return {
        "STABLE": "rgba(102,194,165,0.12)",
        "SQUEEZE": "rgba(252,141,98,0.12)",
        "WIDE": "rgba(141,160,203,0.12)",
    }.get(state, "rgba(200,200,200,0.05)")


def build_master_html(
    bars: pd.DataFrame,
    execution_outputs: Dict[str, pd.DataFrame],
    output_html: str,
) -> None:
    """Interactive master chart with 4H state background + BS points + box lines."""
    df = bars.copy().sort_values("timestamp")

    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC/USDT 1H",
        )
    )

    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["box_upper_edge"], mode="lines", name="Box Upper", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["box_midline"], mode="lines", name="Box Mid", line=dict(width=1.0, dash="dash")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["box_lower_edge"], mode="lines", name="Box Lower", line=dict(width=1.5)))

    for engine_name in ["false_break", "boundary"]:
        edf = execution_outputs[engine_name]
        entries = edf[edf["exec_signal"] != 0]
        if entries.empty:
            continue

        longs = entries[entries["exec_signal"] > 0]
        shorts = entries[entries["exec_signal"] < 0]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["timestamp"],
                    y=longs["price"],
                    mode="markers",
                    name=f"B {engine_name}",
                    marker=dict(symbol="triangle-up", size=8),
                    hovertemplate="%{x}<br>Buy<br>price=%{y:.2f}<extra></extra>",
                )
            )

        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts["timestamp"],
                    y=shorts["price"],
                    mode="markers",
                    name=f"S {engine_name}",
                    marker=dict(symbol="triangle-down", size=8),
                    hovertemplate="%{x}<br>Sell<br>price=%{y:.2f}<extra></extra>",
                )
            )

    if "h4_box_state" in df.columns:
        state = df["h4_box_state"].fillna("INVALID").tolist()
        ts = df["timestamp"].tolist()
        start = 0
        for i in range(1, len(ts) + 1):
            if i == len(ts) or state[i] != state[start]:
                x1 = ts[i - 1] if i - 1 < len(ts) else ts[-1]
                fig.add_vrect(
                    x0=ts[start],
                    x1=x1,
                    fillcolor=_state_color(state[start]),
                    opacity=0.25,
                    line_width=0,
                    layer="below",
                )
                start = i

    fig.update_layout(
        title="Event-First Master Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )

    fig.write_html(output_html, include_plotlyjs="cdn")


def build_event_family_html(
    bars: pd.DataFrame,
    event_df: pd.DataFrame,
    event_signal_col: str,
    event_conf_col: str,
    title: str,
    output_html: str,
) -> None:
    df = bars.copy().sort_values("timestamp")
    evt = event_df.copy().sort_values("timestamp")

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTC/USDT 1H",
        )
    )
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["box_upper_edge"], mode="lines", name="Box Upper"))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["box_midline"], mode="lines", name="Box Mid", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["box_lower_edge"], mode="lines", name="Box Lower"))

    sig = evt[evt[event_signal_col] != 0]
    if not sig.empty:
        longs = sig[sig[event_signal_col] > 0]
        shorts = sig[sig[event_signal_col] < 0]

        if not longs.empty:
            fig.add_trace(
                go.Scatter(
                    x=longs["timestamp"],
                    y=longs["close"] if "close" in longs.columns else longs["price"],
                    mode="markers",
                    name="Long",
                    marker=dict(symbol="triangle-up", size=8, color="#1b9e77"),
                    hovertemplate="%{x}<br>Long<br>conf=%{text}<extra></extra>",
                    text=longs[event_conf_col].round(4).astype(str),
                )
            )
        if not shorts.empty:
            fig.add_trace(
                go.Scatter(
                    x=shorts["timestamp"],
                    y=shorts["close"] if "close" in shorts.columns else shorts["price"],
                    mode="markers",
                    name="Short",
                    marker=dict(symbol="triangle-down", size=8, color="#d95f02"),
                    hovertemplate="%{x}<br>Short<br>conf=%{text}<extra></extra>",
                    text=shorts[event_conf_col].round(4).astype(str),
                )
            )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
    )
    fig.write_html(output_html, include_plotlyjs="cdn")


def build_overlay_png(
    bars: pd.DataFrame,
    execution_outputs: Dict[str, pd.DataFrame],
    output_png: str,
    max_rows: int = 2000,
) -> None:
    df = bars.copy().sort_values("timestamp")
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(15, 7))

    up = df["close"] >= df["open"]
    dn = ~up

    ax.vlines(x, df["low"], df["high"], color="#777777", linewidth=0.7, alpha=0.6)
    ax.vlines(x[up], df.loc[up, "open"], df.loc[up, "close"], color="#2a9d8f", linewidth=2.0)
    ax.vlines(x[dn], df.loc[dn, "open"], df.loc[dn, "close"], color="#e76f51", linewidth=2.0)

    ax.plot(x, df["box_upper_edge"], color="#264653", linewidth=1.2, label="Box Upper")
    ax.plot(x, df["box_midline"], color="#6d597a", linewidth=1.0, linestyle="--", label="Box Mid")
    ax.plot(x, df["box_lower_edge"], color="#264653", linewidth=1.2, label="Box Lower")

    ts_to_idx = {t: i for i, t in enumerate(df["timestamp"].tolist())}
    for engine_name, color in [("false_break", "#1b9e77"), ("boundary", "#d95f02")]:
        edf = execution_outputs[engine_name]
        sig = edf[edf["exec_signal"] != 0]
        if sig.empty:
            continue

        pts = sig[sig["timestamp"].isin(ts_to_idx.keys())]
        xs = [ts_to_idx[t] for t in pts["timestamp"]]
        ys = pts["price"].tolist()
        marker = "^" if engine_name == "false_break" else "v"
        ax.scatter(xs, ys, s=28, marker=marker, color=color, label=f"{engine_name} entries", alpha=0.85)

    step = max(1, len(df) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(t)[:16] for t in df["timestamp"].iloc[::step]], rotation=30, ha="right")
    ax.set_title("K-line + Event Execution Overlay")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=130)
    plt.close(fig)
