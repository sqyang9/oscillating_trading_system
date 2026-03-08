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
    assert int(fb.loc[1, "exec_signal"]) == 0
    assert "h4_range_unusable" in str(fb.loc[1, "blocked_reason"])

    assert int(fb.loc[3, "exec_signal"]) == -1

    bd = outs["boundary"]
    assert int(bd.loc[2, "exec_signal"]) == 0
    assert int(bd.loc[4, "exec_signal"]) in (-1, 0)



def test_expected_value_filter_blocks_low_reward_trade_without_future_leak():
    ts = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.5, 100.5, 120.0, 121.0],
            "low": [99.5, 99.5, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "box_midline": [100.2, 100.2, 120.0, 120.0],
            "box_upper_edge": [101.0, 101.0, 130.0, 130.0],
            "box_lower_edge": [99.0, 99.0, 90.0, 90.0],
            "box_width": [2.0, 2.0, 40.0, 40.0],
            "box_state": ["STABLE"] * 4,
            "event_false_break_signal": [0, 1, 0, 0],
            "event_false_break_confidence": [0.0, 0.9, 0.0, 0.0],
            "event_box_init_signal": [0, 0, 0, 0],
            "event_box_init_confidence": [0.0, 0.0, 0.0, 0.0],
            "h4_range_usable": [True] * 4,
            "h4_box_state": ["STABLE"] * 4,
            "h4_box_width_pct": [0.02] * 4,
            "h4_method_agreement": [0.8] * 4,
            "h4_transition_risk": [0.2] * 4,
        }
    )
    cfg = ExecutionConfig(
        expected_value_filter_enabled=True,
        expected_value_mode="to_midline",
        min_reward_to_cost_ratio=2.0,
        expected_value_round_trip_cost_rate=0.01,
    )

    full = run_execution_engines(df, cfg)["false_break"]
    prefix = run_execution_engines(df.iloc[:2].copy(), cfg)["false_break"]

    assert int(full.loc[1, "exec_signal"]) == 0
    assert int(prefix.loc[1, "exec_signal"]) == 0
    assert "expected_value_blocked" in str(full.loc[1, "blocked_reason"])
    assert float(full.loc[1, "reward_to_cost_ratio"]) < 2.0



def test_regime_lookback_switch_changes_labels():
    ts = pd.date_range("2024-01-01 00:00:00", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 103, 104, 105],
            "box_midline": [100] * 5,
            "box_upper_edge": [106] * 5,
            "box_lower_edge": [94] * 5,
            "box_width": [12] * 5,
            "box_state": ["STABLE"] * 5,
            "event_false_break_signal": [0] * 5,
            "event_false_break_confidence": [0.0] * 5,
            "event_box_init_signal": [0] * 5,
            "event_box_init_confidence": [0.0] * 5,
            "h4_range_usable": [True] * 5,
            "h4_box_state": ["STABLE"] * 5,
            "h4_box_width_pct": [0.02] * 5,
            "h4_method_agreement": [0.8] * 5,
            "h4_transition_risk": [0.2] * 5,
        }
    )

    short_cfg = ExecutionConfig(
        false_break_regime_mode="all",
        regime_lookback_hours=2,
        regime_up_low=0.015,
        regime_up_high=0.10,
    )
    long_cfg = ExecutionConfig(
        false_break_regime_mode="all",
        regime_lookback_hours=4,
        regime_up_low=0.015,
        regime_up_high=0.10,
    )

    short_out = run_execution_engines(df, short_cfg)["false_break"]
    long_out = run_execution_engines(df, long_cfg)["false_break"]

    assert short_out.loc[2, "fb_regime"] == "up"
    assert long_out.loc[2, "fb_regime"] == "warmup"



def test_warmup_trades_can_be_blocked_without_changing_regime_logic():
    ts = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0, 100.2, 100.1, 100.0],
            "box_midline": [101.0] * 4,
            "box_upper_edge": [103.0] * 4,
            "box_lower_edge": [97.0] * 4,
            "box_width": [6.0] * 4,
            "box_width_pct": [0.02] * 4,
            "box_state": ["STABLE"] * 4,
            "event_false_break_signal": [1, 0, 0, 0],
            "event_false_break_confidence": [0.9, 0.0, 0.0, 0.0],
            "event_false_break_base_confirms": [3, 0, 0, 0],
            "event_box_init_signal": [0, 0, 0, 0],
            "event_box_init_confidence": [0.0, 0.0, 0.0, 0.0],
            "h4_range_usable": [True] * 4,
            "h4_box_state": ["STABLE"] * 4,
            "h4_box_width_pct": [0.02] * 4,
            "h4_method_agreement": [0.8] * 4,
            "h4_transition_risk": [0.2] * 4,
        }
    )
    cfg = ExecutionConfig(false_break_regime_mode="down_only", regime_lookback_hours=24, allow_warmup_trades=False)
    out = run_execution_engines(df, cfg)["false_break"]

    assert out.loc[0, "fb_regime"] == "warmup"
    assert int(out.loc[0, "exec_signal"]) == 0
    assert "warmup_blocked" in str(out.loc[0, "blocked_reason"])



def test_bad_width_bucket_filter_only_blocks_target_bucket():
    ts = pd.date_range("2024-01-01 00:00:00", periods=4, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.0] * 4,
            "box_midline": [101.0] * 4,
            "box_upper_edge": [103.0] * 4,
            "box_lower_edge": [97.0] * 4,
            "box_width": [6.0] * 4,
            "box_width_pct": [0.012, 0.020, 0.020, 0.020],
            "box_state": ["STABLE"] * 4,
            "event_false_break_signal": [1, 1, 0, 0],
            "event_false_break_confidence": [0.9, 0.9, 0.0, 0.0],
            "event_false_break_base_confirms": [3, 3, 0, 0],
            "event_box_init_signal": [0, 0, 0, 0],
            "event_box_init_confidence": [0.0, 0.0, 0.0, 0.0],
            "h4_range_usable": [True] * 4,
            "h4_box_state": ["STABLE"] * 4,
            "h4_box_width_pct": [0.02] * 4,
            "h4_method_agreement": [0.8] * 4,
            "h4_transition_risk": [0.2] * 4,
        }
    )
    cfg = ExecutionConfig(
        false_break_regime_mode="all",
        allow_warmup_trades=True,
        bad_width_bucket_filter_enabled=True,
        side_cooldown_bars_false_break=0,
    )
    out = run_execution_engines(df, cfg)["false_break"]

    assert int(out.loc[0, "exec_signal"]) == 0
    assert "bad_width_bucket_blocked" in str(out.loc[0, "blocked_reason"])
    assert int(out.loc[1, "exec_signal"]) == 1
