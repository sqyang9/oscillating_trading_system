import pandas as pd

from study_preserve_refinement_round import build_refinement_flags, filter_mask_for_variant


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
            "entry_time": pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
            "score_bucket": ["fragile", "fragile", "fragile", "fragile", "middle", "fragile"],
            "preserve_bucket": ["high", "high", "high", "low", "high", "high"],
            "suppress_bucket": ["low", "high", "middle", "high", "low", "middle"],
            "reentry_strength": [0.08, 0.20, 0.11, 0.20, 0.05, 0.11],
            "suppress_score_pct": [0.34, 0.70, 0.34, 0.70, 0.34, 0.34],
            "prev_reject_combo": [False, False, False, False, False, True],
            "box_width_pct": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
        }
    )


def test_refinement_flags_are_deterministic_and_nested():
    scored = build_refinement_flags(_frame())
    assert scored["flag_prior_remove"].tolist() == [False, False, False, True, False, False]
    assert scored["flag_weak_preserve_v1"].tolist() == [True, False, False, False, False, False]
    assert scored["flag_weak_preserve_v2"].tolist() == [True, False, True, False, False, False]
    assert scored["flag_weak_preserve_best"].tolist() == [True, False, True, False, False, False]


def test_variant_masks_extend_prior_rule_without_touching_non_fragile_rows():
    scored = build_refinement_flags(_frame())
    prior = filter_mask_for_variant(scored, "prior_preserve_first_variant")
    refined1 = filter_mask_for_variant(scored, "refined_preserve_variant_1")
    refined2 = filter_mask_for_variant(scored, "refined_preserve_variant_2")
    best = filter_mask_for_variant(scored, "best_effort_refined_preserve_variant")

    assert prior.sum() == 1
    assert refined1.sum() == 2
    assert refined2.sum() == 3
    assert best.sum() == 3
    assert not refined2.iloc[4]
