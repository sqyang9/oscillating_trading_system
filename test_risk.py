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

    # Frequent long attempts in a down move should cause repeated losses.
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