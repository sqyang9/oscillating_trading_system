import pandas as pd

from event_first import BoxInitiationConfig, FalseBreakConfig, detect_box_initiation_events, detect_false_break_events


def _base_df() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=9, freq="1h", tz="UTC")
    close = [95.0, 99.0, 101.0, 98.0, 95.0, 91.0, 96.0, 97.0, 96.0]
    open_ = [95.0, 98.5, 100.5, 101.0, 96.0, 90.5, 92.0, 96.5, 96.5]
    high = [96.0, 100.0, 103.0, 101.5, 96.5, 92.0, 97.5, 98.5, 97.0]
    low = [94.0, 98.0, 99.5, 97.0, 89.2, 89.0, 91.8, 95.5, 95.0]

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1.0,
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

    # False-break setup: overshoot up at index=2, re-entry short at index=3 with confirmations.
    df.loc[3, "touch_upper"] = True
    df.loc[3, "repeated_upper_count"] = 2

    # Boundary-init long setup near lower edge.
    df.loc[5, "touch_lower"] = True
    df.loc[5, "repeated_lower_count"] = 2
    return df


def test_false_break_detects_short_reentry():
    df = _base_df()
    out = detect_false_break_events(df, FalseBreakConfig(min_confirmations=2, reentry_window=3))

    assert int(out.loc[3, "event_false_break_signal"]) == -1
    assert out.loc[3, "event_false_break_confirms"] >= 2
    assert float(out.loc[3, "event_false_break_confidence"]) > 0.0


def test_box_initiation_detects_long_from_lower_edge():
    df = _base_df()
    out = detect_box_initiation_events(df, BoxInitiationConfig(min_confirmations=2))

    long_count = int((out["event_box_init_signal"] == 1).sum())
    assert long_count >= 1
    assert out["event_box_init_confidence"].max() > 0.0