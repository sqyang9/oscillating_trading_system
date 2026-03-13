import pandas as pd

from study_preserve_first_robustness import add_context_splits, build_time_split_definitions


def test_build_time_split_definitions_is_deterministic_and_includes_expected_windows():
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2020-01-15T00:00:00Z",
                    "2021-06-15T00:00:00Z",
                    "2022-06-15T00:00:00Z",
                    "2023-06-15T00:00:00Z",
                    "2024-06-15T00:00:00Z",
                    "2025-06-15T00:00:00Z",
                ],
                utc=True,
            )
        }
    )
    splits = build_time_split_definitions(trades)
    labels = [(row["split_type"], row["split_label"]) for row in splits]

    assert labels[0] == ("full", "full_sample")
    assert ("half", "early_half") in labels
    assert ("half", "late_half") in labels
    assert any(split_type == "rolling_36m" for split_type, _ in labels)


def test_add_context_splits_assigns_year_half_and_phase_labels():
    scored = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                [
                    "2020-01-15T00:00:00Z",
                    "2021-06-15T00:00:00Z",
                    "2023-06-15T00:00:00Z",
                    "2025-06-15T00:00:00Z",
                ],
                utc=True,
            ),
            "baseline_trade": [True, True, True, True],
        }
    )
    merged_times = pd.date_range("2019-01-01", periods=7 * 24 * 365, freq="1h", tz="UTC")
    merged = pd.DataFrame({"timestamp": merged_times, "close": range(1, len(merged_times) + 1)})

    enriched, meta = add_context_splits(scored, merged)

    assert list(enriched["year"]) == [2020, 2021, 2023, 2025]
    assert set(enriched["sample_half"]) == {"early_half", "late_half"}
    assert "phase_context" in enriched.columns
    assert set(meta["name"]) == {"median_entry_time", "phase_q1", "phase_q2"}
