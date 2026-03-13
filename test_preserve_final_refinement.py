import pandas as pd

from study_preserve_final_refinement import guard_col_for_alias, mask_for_alias
from study_preserve_refinement_round import build_refinement_flags


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


def test_alias_guards_match_expected_columns():
    assert guard_col_for_alias("control") == "guard_prior_preserve_high"
    assert guard_col_for_alias("prior_preserve_first") == "guard_prior_preserve_high"
    assert guard_col_for_alias("refined_preserve_variant") == "guard_refined_preserve_1"
    assert guard_col_for_alias("experimental_narrow_candidate") == "guard_refined_preserve_2"


def test_alias_masks_match_expected_variant_shapes():
    scored = build_refinement_flags(_frame())
    assert mask_for_alias(scored, "control").sum() == 0
    assert mask_for_alias(scored, "prior_preserve_first").sum() == 1
    assert mask_for_alias(scored, "refined_preserve_variant").sum() == 2
    assert mask_for_alias(scored, "experimental_narrow_candidate").sum() == 3
