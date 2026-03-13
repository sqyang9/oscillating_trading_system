import pandas as pd

from study_entry_edge_optimization import apply_entry_quality_score, build_score_spec, filter_mask_for_variant


def _scored_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "entry_time": pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
            "observation_class": ["strong", "strong", "fragile", "fragile", "other", "other"],
            "base_confirms": [3, 3, 2, 2, 3, 2],
            "confirm_richness": [3, 3, 2, 2, 2, 2],
            "reentry_strength": [0.7, 0.8, 0.2, 0.3, 0.5, 0.4],
            "box_confidence": [0.9, 0.85, 0.4, 0.45, 0.6, 0.55],
            "transition_risk": [0.1, 0.2, 0.7, 0.6, 0.4, 0.5],
            "body_reclaim_reversal": [True, True, False, False, True, False],
            "prev_reject_combo": [True, False, False, False, False, False],
            "engulfing_reversal": [False, True, False, False, False, False],
            "reversal_composite_score": [2, 2, 0, 1, 1, 1],
            "box_width_atr_mult": [2.0, 2.2, 0.8, 0.9, 1.4, 1.5],
            "edge_distance_atr": [0.4, 0.5, 1.1, 1.0, 0.8, 0.7],
            "overshoot_distance_atr": [0.8, 0.9, 0.2, 0.3, 0.5, 0.4],
            "anchor_shift_bars": [1, 2, 8, 7, 4, 5],
            "overshoot_bars": [1, 2, 5, 6, 3, 3],
            "fb_breakout_age_bars": [1, 1, 4, 5, 2, 2],
            "fb_reentry_vs_breakout_ratio": [1.3, 1.2, 0.8, 0.85, 1.0, 1.0],
            "volume_rank_24": [0.7, 0.8, 0.3, 0.35, 0.5, 0.5],
            "volume_hour_norm": [1.2, 1.1, 0.7, 0.75, 0.9, 0.95],
        }
    )
    return df


def test_score_spec_and_score_bucket_separate_strong_from_fragile():
    df = _scored_frame()
    spec = build_score_spec(df)
    scored = apply_entry_quality_score(df, spec)

    strong_mean = scored.loc[scored["observation_class"] == "strong", "entry_quality_score_pct"].mean()
    fragile_mean = scored.loc[scored["observation_class"] == "fragile", "entry_quality_score_pct"].mean()
    assert strong_mean > fragile_mean


def test_filter_masks_are_nested_from_light_to_best_effort():
    df = _scored_frame()
    spec = build_score_spec(df)
    scored = apply_entry_quality_score(df, spec)

    f1 = filter_mask_for_variant(scored, "fragile_entry_filter_1")
    f2 = filter_mask_for_variant(scored, "fragile_entry_filter_2")
    best = filter_mask_for_variant(scored, "best_effort_precision_variant")

    assert (f2 <= f1).all()
    assert (best >= f2).all()
