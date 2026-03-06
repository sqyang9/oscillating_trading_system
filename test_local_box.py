import numpy as np
import pandas as pd

from local_box import LocalBoxConfig, attach_gap_segments, build_local_boxes, resample_5m_to_1h


def _make_hourly_bars(start: str, closes: list[float], freq: str = "1h") -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1.0,
        }
    )


def test_resample_5m_marks_gap_without_bridging():
    ts = pd.date_range("2024-01-01 00:00:00", periods=24, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1.0,
        }
    )
    # Remove one sub-bar inside the second hour.
    df = df.drop(index=16).reset_index(drop=True)

    out = resample_5m_to_1h(df)
    assert len(out) == 2
    assert bool(out.loc[1, "gap"])
    assert np.isnan(out.loc[1, "close"])


def test_local_box_resets_after_gap():
    first = _make_hourly_bars("2024-01-01", [100, 102, 104, 106, 108])
    second = _make_hourly_bars("2024-01-01 10:00:00", [80, 81, 82, 83])
    df = pd.concat([first, second], ignore_index=True)

    segmented = attach_gap_segments(df, expected_freq="1h")
    cfg = LocalBoxConfig(rolling_window=3, min_periods=1, pivot_left=1, pivot_right=1, drift_window=2)
    box = build_local_boxes(segmented, cfg)

    post_gap_idx = box.index[box["timestamp"] == pd.Timestamp("2024-01-01 10:00:00+00:00")][0]
    assert box.loc[post_gap_idx, "segment_id"] != box.loc[post_gap_idx - 1, "segment_id"]
    assert abs(box.loc[post_gap_idx, "q_upper_edge"] - box.loc[post_gap_idx, "close"]) < 1e-9


def test_local_box_causal_prefix_consistency():
    closes = [100 + np.sin(i / 3.0) * 2 + i * 0.03 for i in range(50)]
    full = _make_hourly_bars("2024-01-01", closes)

    cfg = LocalBoxConfig(rolling_window=8, min_periods=3, pivot_left=2, pivot_right=2, drift_window=4)

    full_box = build_local_boxes(attach_gap_segments(full, "1h"), cfg)
    cut = full.iloc[:35].copy()
    cut_box = build_local_boxes(attach_gap_segments(cut, "1h"), cfg)

    compare_cols = ["q_lower_edge", "q_upper_edge", "box_upper_edge", "box_lower_edge", "box_midline"]
    pd.testing.assert_frame_equal(
        full_box.loc[:34, compare_cols].reset_index(drop=True),
        cut_box.loc[:, compare_cols].reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=1e-10,
    )