import pandas as pd

from study_fragile_label_refinement import apply_hopeless_proxy, assign_fragile_subtypes, build_hopeless_proxy_spec


def _fragile_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "entry_stop_distance": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "forward_mfe_6h": [0.002, 0.004, 0.012, 0.016, 0.007, 0.009],
            "forward_mae_6h": [0.014, 0.011, 0.003, 0.004, 0.008, 0.007],
            "fragile_trade": [True, True, True, True, True, True],
            "confirm_richness": [1, 2, 4, 4, 3, 2],
            "base_confirms": [1, 2, 4, 4, 3, 2],
            "reentry_strength": [0.10, 0.20, 0.80, 0.70, 0.40, 0.35],
            "transition_risk": [0.90, 0.80, 0.20, 0.25, 0.50, 0.60],
            "box_confidence": [0.20, 0.30, 0.85, 0.90, 0.55, 0.50],
            "h4_range_usable": [False, False, True, True, True, False],
            "method_agreement": [0.30, 0.35, 0.80, 0.75, 0.55, 0.45],
            "edge_distance_atr": [1.10, 1.00, 0.40, 0.35, 0.70, 0.65],
            "box_width_pct": [0.012, 0.013, 0.020, 0.018, 0.015, 0.014],
            "box_width_atr_mult": [0.80, 0.90, 2.00, 2.10, 1.40, 1.30],
            "overshoot_distance_atr": [0.20, 0.25, 0.80, 0.85, 0.45, 0.40],
            "reentry_distance_atr": [0.25, 0.30, 0.75, 0.80, 0.50, 0.45],
            "anchor_shift_bars": [8, 7, 2, 1, 5, 5],
            "overshoot_bars": [6, 5, 2, 2, 4, 4],
            "fb_breakout_age_bars": [5, 5, 1, 1, 3, 3],
            "body_reclaim_reversal": [False, False, True, True, False, False],
            "prev_reject_combo": [False, False, True, True, False, False],
            "engulfing_reversal": [False, False, True, False, False, False],
            "reversal_composite_score": [0, 1, 3, 3, 1, 1],
            "fb_reentry_vs_breakout_ratio": [0.80, 0.85, 1.30, 1.25, 1.00, 0.95],
            "volume_rank_24": [0.20, 0.25, 0.75, 0.80, 0.45, 0.40],
            "volume_hour_norm": [0.70, 0.75, 1.15, 1.20, 0.95, 0.90],
        }
    )


def test_assign_fragile_subtypes_uses_fixed_excursion_bands():
    labeled = assign_fragile_subtypes(_fragile_frame())
    assert list(labeled["fragile_subtype"]) == [
        "fragile_hopeless",
        "fragile_hopeless",
        "fragile_high_upside",
        "fragile_high_upside",
        "fragile_mixed",
        "fragile_mixed",
    ]


def test_hopeless_proxy_scores_hopeless_rows_above_high_upside_rows():
    labeled = assign_fragile_subtypes(_fragile_frame())
    focus = labeled[labeled["fragile_focus_label"].isin(["hopeless", "high_upside"])].copy()
    spec = build_hopeless_proxy_spec(focus)
    scored = apply_hopeless_proxy(labeled, spec)

    hopeless_mean = scored.loc[scored["fragile_focus_label"] == "hopeless", "hopeless_proxy_score_pct"].mean()
    upside_mean = scored.loc[scored["fragile_focus_label"] == "high_upside", "hopeless_proxy_score_pct"].mean()
    assert hopeless_mean > upside_mean
