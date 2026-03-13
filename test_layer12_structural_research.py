import pandas as pd

from event_first import FalseBreakConfig, detect_false_break_events
from layer12_structural_research_helpers import detect_false_break_events_research


def _base_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=9, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [95.0, 98.5, 100.5, 101.0, 96.0, 90.5, 92.0, 96.5, 96.5],
            "high": [96.0, 100.0, 103.0, 101.5, 96.5, 92.0, 97.5, 98.5, 97.0],
            "low": [94.0, 98.0, 99.5, 97.0, 89.2, 89.0, 91.8, 95.5, 95.0],
            "close": [95.0, 99.0, 101.0, 98.0, 95.0, 91.0, 96.0, 97.0, 96.0],
            "volume": [10.0] * 9,
            "segment_id": 0,
            "box_upper_edge": 100.0,
            "box_lower_edge": 90.0,
            "box_midline": 95.0,
            "box_width": 10.0,
            "box_valid": True,
            "box_confidence": 0.9,
            "touch_upper": False,
            "touch_lower": False,
            "pivot_upper_confirm": False,
            "pivot_lower_confirm": False,
            "repeated_upper_count": 0,
            "repeated_lower_count": 0,
        }
    )
    df.loc[3, "touch_upper"] = True
    df.loc[3, "repeated_upper_count"] = 2
    return df


def test_research_baseline_matches_production_signal_path():
    df = _base_df()
    cfg = FalseBreakConfig(min_confirmations=2, reentry_window=3)
    prod = detect_false_break_events(df, cfg)
    research = detect_false_break_events_research(df, cfg, anchor_policy="latest", reversal_policy="baseline")

    pd.testing.assert_series_equal(prod["event_false_break_signal"], research["event_false_break_signal"], check_names=False)
    pd.testing.assert_series_equal(prod["event_false_break_base_confirms"], research["event_false_break_base_confirms"], check_names=False)


def test_locked_anchor_variant_does_not_refresh_anchor_on_later_overshoot():
    ts = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 100.5, 101.0, 101.0, 100.8, 100.2],
            "high": [100.5, 102.0, 102.2, 101.2, 100.9, 100.5],
            "low": [99.5, 100.2, 100.4, 99.8, 99.7, 99.5],
            "close": [100.0, 101.5, 101.6, 100.4, 99.9, 100.0],
            "volume": [10.0] * 6,
            "segment_id": 0,
            "box_upper_edge": 100.0,
            "box_lower_edge": 95.0,
            "box_midline": 97.5,
            "box_width": 5.0,
            "box_valid": True,
            "box_confidence": 0.9,
            "touch_upper": [False, False, False, False, True, False],
            "touch_lower": False,
            "pivot_upper_confirm": False,
            "pivot_lower_confirm": False,
            "repeated_upper_count": [0, 0, 0, 0, 2, 0],
            "repeated_lower_count": 0,
        }
    )
    cfg = FalseBreakConfig(min_confirmations=2, reentry_window=2)

    latest = detect_false_break_events_research(df, cfg, anchor_policy="latest", reversal_policy="baseline")
    locked = detect_false_break_events_research(df, cfg, anchor_policy="first", reversal_policy="baseline")

    assert int(latest.loc[4, "event_false_break_signal"]) == -1
    assert int(locked.loc[4, "event_false_break_signal"]) == 0

