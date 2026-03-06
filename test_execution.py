import pandas as pd

from execution import ExecutionConfig, align_4h_to_1h, run_execution_engines


def _make_1h_with_events() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 00:00:00", periods=7, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 102, 103, 104, 105, 106],
            "high": [101, 102, 103, 104, 105, 106, 107],
            "low": [99, 100, 101, 102, 103, 104, 105],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5],
            "box_midline": [100] * 7,
            "box_upper_edge": [103] * 7,
            "box_lower_edge": [97] * 7,
            "box_width": [6] * 7,
            "box_state": ["STABLE"] * 7,
            "event_false_break_signal": [0, 1, 1, -1, 0, 0, 0],
            "event_false_break_confidence": [0, 0.8, 0.8, 0.9, 0, 0, 0],
            "event_box_init_signal": [0, 0, -1, 0, -1, 0, 0],
            "event_box_init_confidence": [0, 0, 0.6, 0, 0.65, 0, 0],
            "h4_range_usable": [False, False, False, True, True, True, True],
            "h4_box_state": ["INVALID", "INVALID", "INVALID", "STABLE", "STABLE", "STABLE", "STABLE"],
            "h4_box_width_pct": [0.02] * 7,
            "h4_method_agreement": [0.8] * 7,
            "h4_transition_risk": [0.2] * 7,
        }
    )
    return df


def test_align_4h_applies_only_after_close():
    bars_1h = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 00:00:00", periods=6, freq="1h", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    )
    bars_4h = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")],
            "range_usable": [True],
            "box_state": ["STABLE"],
            "box_width_pct": [0.02],
            "method_agreement": [0.8],
            "transition_risk": [0.2],
            "box_confidence": [0.7],
            "box_valid": [True],
        }
    )

    out = align_4h_to_1h(bars_1h, bars_4h)
    assert not bool(out.loc[2, "h4_range_usable"])
    assert bool(out.loc[4, "h4_range_usable"])


def test_execution_gating_and_cooldown_behavior():
    df = _make_1h_with_events()
    cfg = ExecutionConfig(side_cooldown_bars_false_break=2, side_cooldown_bars_boundary=2)
    outs = run_execution_engines(df, cfg)

    fb = outs["false_break"]
    # First 1/2 long signals are blocked due 4H unusable.
    assert int(fb.loc[1, "exec_signal"]) == 0
    assert "h4_range_unusable" in str(fb.loc[1, "blocked_reason"])

    # Signal at index 3 is first one with 4H usable and should pass.
    assert int(fb.loc[3, "exec_signal"]) == -1

    bd = outs["boundary"]
    # Boundary at index 2 blocked by 4H unusable, index 4 can pass.
    assert int(bd.loc[2, "exec_signal"]) == 0
    assert int(bd.loc[4, "exec_signal"]) in (-1, 0)