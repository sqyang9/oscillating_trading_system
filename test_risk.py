import pandas as pd

from risk import RiskConfig, apply_risk_layer


def _bars_for_risk() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 100, 99, 98, 97, 96, 95, 95, 94, 94, 93, 93],
            "high": [101, 101, 100, 99, 98, 97, 96, 96, 95, 95, 94, 94],
            "low": [99, 96, 95, 94, 93, 92, 91, 91, 90, 90, 89, 89],
            "close": [100, 99, 98, 97, 96, 95, 94, 94, 93, 93, 92, 92],
            "box_width": [4.0] * 12,
            "box_valid": [True] * 12,
            "box_state": ["STABLE"] * 12,
        }
    )



def test_risk_price_stop_and_circuit_breaker_blocks_entries():
    bars = _bars_for_risk()

    exec_false_break = pd.DataFrame(
        {
            "timestamp": bars["timestamp"].iloc[[0, 2, 4, 6, 8, 10]].tolist(),
            "engine": "false_break",
            "raw_signal": [1, 1, 1, 1, 1, 1],
            "event_confidence": [0.8] * 6,
            "gate_pass": [True] * 6,
            "blocked_reason": [""] * 6,
            "exec_signal": [1, 1, 1, 1, 1, 1],
            "side": ["long"] * 6,
            "price": bars["close"].iloc[[0, 2, 4, 6, 8, 10]].tolist(),
        }
    )
    exec_boundary = exec_false_break.iloc[0:0].copy()

    cfg = RiskConfig(
        atr_window=2,
        atr_mult=0.5,
        box_stop_mult=0.4,
        min_stop_pct=0.001,
        max_hold_bars=5,
        circuit_breaker_loss_streak=2,
        circuit_breaker_pause_bars=3,
        side_cooldown_bars=0,
    )
    out = apply_risk_layer(
        bars,
        {"false_break": exec_false_break, "boundary": exec_boundary},
        cfg,
    )

    trades = out["trades_false_break"]
    logs = out["risk_log_false_break"]

    assert not trades.empty
    assert "price_stop" in set(trades["exit_reason"])
    assert ((logs["event"] == "circuit") & (logs["status"] == "pause")).any()
    assert ((logs["event"] == "entry") & (logs["status"] == "blocked")).any()



def test_risk_take_profit_exit_reason_is_recorded():
    ts = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 100.0, 102.0, 103.0, 103.0, 102.0],
            "high": [100.5, 102.5, 104.5, 103.5, 103.0, 102.5],
            "low": [99.5, 99.8, 101.5, 102.0, 101.5, 101.0],
            "close": [100.0, 102.0, 104.0, 103.0, 102.0, 101.5],
            "box_width": [6.0] * 6,
            "box_valid": [True] * 6,
            "box_state": ["STABLE"] * 6,
        }
    )

    exec_false_break = pd.DataFrame(
        {
            "timestamp": [ts[0]],
            "engine": ["false_break"],
            "raw_signal": [1],
            "event_confidence": [0.9],
            "gate_pass": [True],
            "blocked_reason": [""],
            "exec_signal": [1],
            "side": ["long"],
            "price": [100.0],
            "box_midline": [102.0],
            "box_upper_edge": [104.0],
            "box_lower_edge": [98.0],
        }
    )
    exec_boundary = exec_false_break.iloc[0:0].copy()

    cfg = RiskConfig(
        atr_window=2,
        atr_mult=5.0,
        box_stop_mult=5.0,
        min_stop_pct=0.001,
        max_hold_bars=5,
        early_progress_bars=5,
        early_progress_min_return=-1.0,
        take_profit_enabled=True,
        take_profit_mode="midline",
    )
    out = apply_risk_layer(
        bars,
        {"false_break": exec_false_break, "boundary": exec_boundary},
        cfg,
    )

    trades = out["trades_false_break"]
    assert len(trades) == 1
    assert trades.loc[0, "exit_reason"] == "take_profit_midline"
    assert float(trades.loc[0, "exit_price"]) == 102.0



def test_early_failure_uses_only_bars_seen_so_far():
    ts = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.2, 100.2, 100.2, 100.2],
            "low": [100.0, 99.7, 99.6, 98.8, 98.5],
            "close": [100.0, 99.95, 99.92, 99.0, 98.7],
            "box_width": [4.0] * 5,
            "box_valid": [True] * 5,
            "box_state": ["STABLE"] * 5,
        }
    )
    exec_false_break = pd.DataFrame(
        {
            "timestamp": [ts[0]],
            "engine": ["false_break"],
            "raw_signal": [1],
            "event_confidence": [0.9],
            "event_base_confirms": [2],
            "box_width_pct": [0.012],
            "entry_warmup": [False],
            "gate_pass": [True],
            "blocked_reason": [""],
            "exec_signal": [1],
            "side": ["long"],
            "price": [100.0],
            "box_midline": [101.0],
            "box_upper_edge": [103.0],
            "box_lower_edge": [97.0],
        }
    )
    exec_boundary = exec_false_break.iloc[0:0].copy()
    cfg = RiskConfig(
        atr_window=2,
        atr_mult=10.0,
        box_stop_mult=10.0,
        min_stop_pct=0.0001,
        max_hold_bars=10,
        early_progress_bars=10,
        early_progress_min_return=-1.0,
        early_failure_filter_enabled=True,
        early_failure_bars=2,
        early_failure_min_progress=0.003,
        early_failure_max_adverse=0.008,
        early_failure_scope="all",
    )
    full = apply_risk_layer(bars, {"false_break": exec_false_break, "boundary": exec_boundary}, cfg)["trades_false_break"]
    prefix = apply_risk_layer(bars.iloc[:3].copy(), {"false_break": exec_false_break, "boundary": exec_boundary}, cfg)["trades_false_break"]

    assert len(full) == 1
    assert full.loc[0, "exit_reason"] == "early_failure"
    assert full.loc[0, "exit_time"] == ts[3]
    assert prefix.loc[0, "exit_reason"] == "end_of_data"



def test_early_failure_width_scope_only_hits_target_bucket():
    ts = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    bars = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 8,
            "high": [100.0, 100.1, 100.1, 100.0, 100.0, 100.1, 100.2, 100.3],
            "low": [100.0, 99.0, 99.0, 99.0, 100.0, 99.0, 98.9, 98.8],
            "close": [100.0, 99.5, 99.4, 99.4, 100.0, 99.6, 99.7, 99.8],
            "box_width": [4.0] * 8,
            "box_valid": [True] * 8,
            "box_state": ["STABLE"] * 8,
        }
    )
    exec_false_break = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[4]],
            "engine": ["false_break", "false_break"],
            "raw_signal": [1, 1],
            "event_confidence": [0.8, 0.8],
            "event_base_confirms": [2, 2],
            "box_width_pct": [0.012, 0.020],
            "entry_warmup": [False, False],
            "gate_pass": [True, True],
            "blocked_reason": ["", ""],
            "exec_signal": [1, 1],
            "side": ["long", "long"],
            "price": [100.0, 100.0],
            "box_midline": [101.0, 101.0],
            "box_upper_edge": [103.0, 103.0],
            "box_lower_edge": [97.0, 97.0],
        }
    )
    exec_boundary = exec_false_break.iloc[0:0].copy()
    cfg = RiskConfig(
        atr_window=2,
        atr_mult=10.0,
        box_stop_mult=10.0,
        min_stop_pct=0.0001,
        max_hold_bars=3,
        early_progress_bars=10,
        early_progress_min_return=-1.0,
        side_cooldown_bars=0,
        early_failure_filter_enabled=True,
        early_failure_bars=2,
        early_failure_min_progress=0.003,
        early_failure_max_adverse=0.008,
        early_failure_scope="width_bucket_only",
        early_failure_width_min_pct=0.010,
        early_failure_width_max_pct=0.015,
    )
    trades = apply_risk_layer(bars, {"false_break": exec_false_break, "boundary": exec_boundary}, cfg)["trades_false_break"]

    assert len(trades) == 2
    assert trades.loc[0, "exit_reason"] == "early_failure"
    assert trades.loc[1, "entry_box_width_pct"] == 0.02
    assert trades.loc[1, "exit_reason"] != "early_failure"
