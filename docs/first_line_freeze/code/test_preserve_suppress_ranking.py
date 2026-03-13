import pandas as pd

from study_preserve_suppress_ranking import (
    add_interaction_features,
    apply_dual_score,
    build_dual_score_spec,
    filter_mask_for_variant,
)


def _fragile_frame() -> pd.DataFrame:
    rows = []
    labels = ["hopeless", "hopeless", "hopeless", "high_upside", "high_upside", "high_upside", "other", "other", "other"]
    for i, label in enumerate(labels):
        hopeless = label == "hopeless"
        high_upside = label == "high_upside"
        rows.append(
            {
                "entry_time": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=i),
                "fragile_trade": True,
                "fragile_focus_label": label,
                "fragile_subtype": "fragile_hopeless" if hopeless else "fragile_high_upside" if high_upside else "fragile_mixed",
                "score_bucket": "fragile",
                "confirm_richness": 1 if hopeless else 4 if high_upside else 2,
                "base_confirms": 1 if hopeless else 4 if high_upside else 2,
                "reentry_strength": 0.15 if hopeless else 0.75 if high_upside else 0.35,
                "transition_risk": 0.80 if hopeless else 0.20 if high_upside else 0.50,
                "box_confidence": 0.30 if hopeless else 0.85 if high_upside else 0.55,
                "h4_range_usable": high_upside,
                "method_agreement": 0.35 if hopeless else 0.80 if high_upside else 0.50,
                "box_width_pct": 0.024 if hopeless else 0.014 if high_upside else 0.018,
                "box_width_atr_mult": 3.8 if hopeless else 2.1 if high_upside else 2.8,
                "edge_distance_atr": 0.95 if hopeless else 0.40 if high_upside else 0.70,
                "reentry_distance_atr": 0.90 if hopeless else 0.35 if high_upside else 0.60,
                "anchor_shift_bars": 7 if hopeless else 2 if high_upside else 5,
                "overshoot_bars": 5 if hopeless else 2 if high_upside else 3,
                "fb_breakout_age_bars": 4 if hopeless else 1 if high_upside else 2,
                "prev_reject_combo": high_upside,
                "body_reclaim_reversal": high_upside,
                "engulfing_reversal": high_upside,
                "reversal_composite_score": 0 if hopeless else 3 if high_upside else 1,
                "fb_reentry_vs_breakout_ratio": 0.85 if hopeless else 1.20 if high_upside else 1.00,
                "volume_rank_24": 0.25 if hopeless else 0.75 if high_upside else 0.45,
                "volume_hour_norm": 0.75 if hopeless else 1.15 if high_upside else 0.95,
            }
        )
    return pd.DataFrame(rows)


def test_dual_scores_rank_high_upside_and_hopeless_in_opposite_directions():
    df, _ = add_interaction_features(_fragile_frame())
    preserve_spec = build_dual_score_spec(df, positive_label="high_upside", target_name="preserve")
    suppress_spec = build_dual_score_spec(df, positive_label="hopeless", target_name="suppress")
    scored = apply_dual_score(df, preserve_spec, "preserve")
    scored = apply_dual_score(scored, suppress_spec, "suppress")

    preserve_high = scored.loc[scored["fragile_focus_label"] == "high_upside", "preserve_score_pct"].mean()
    preserve_hopeless = scored.loc[scored["fragile_focus_label"] == "hopeless", "preserve_score_pct"].mean()
    suppress_high = scored.loc[scored["fragile_focus_label"] == "hopeless", "suppress_score_pct"].mean()
    suppress_upside = scored.loc[scored["fragile_focus_label"] == "high_upside", "suppress_score_pct"].mean()

    assert preserve_high > preserve_hopeless
    assert suppress_high > suppress_upside


def test_preserve_first_variant_protects_preserve_high_bucket_rows():
    df, _ = add_interaction_features(_fragile_frame())
    preserve_spec = build_dual_score_spec(df, positive_label="high_upside", target_name="preserve")
    suppress_spec = build_dual_score_spec(df, positive_label="hopeless", target_name="suppress")
    scored = apply_dual_score(df, preserve_spec, "preserve")
    scored = apply_dual_score(scored, suppress_spec, "suppress")

    mask = filter_mask_for_variant(scored, "preserve_first_variant")
    assert not mask.loc[scored["preserve_bucket"] == "high"].any()
