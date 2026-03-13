from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestConfig, run_backtest
from layer12_structural_research_helpers import (
    FROZEN_LAYER12_CONTROL_CONFIG,
    ResearchVariant,
    add_forward_entry_quality,
    build_trade_audit,
    deep_copy_config,
    effect_size,
    load_closed_base_frames,
    parse_times,
    run_research_variant,
    save_yaml,
    summarize_returns,
    winner_sensitivity,
)
from risk import RiskConfig, apply_risk_layer
from study_entry_edge_optimization import (
    apply_entry_quality_score,
    build_execution_candidate_table,
    build_review_chart,
    build_score_spec,
)


CONTROL_VARIANT = ResearchVariant(name="control", anchor_policy="latest", reversal_policy="baseline")

HOPELESS_CUTOFF = 0.50
HIGH_UPSIDE_CUTOFF = 1.00

ENTRY_FEATURE_FAMILIES = {
    "confirm": ["confirm_richness", "base_confirms", "reentry_strength"],
    "context": ["transition_risk", "box_confidence", "h4_range_usable", "method_agreement"],
    "geometry": ["edge_distance_atr", "box_width_pct", "box_width_atr_mult", "overshoot_distance_atr", "reentry_distance_atr"],
    "anchor": ["anchor_shift_bars", "overshoot_bars", "fb_breakout_age_bars"],
    "reversal": ["body_reclaim_reversal", "prev_reject_combo", "engulfing_reversal", "reversal_composite_score"],
    "volume": ["fb_reentry_vs_breakout_ratio", "volume_rank_24", "volume_hour_norm"],
}

FAMILY_WEIGHTS = {
    "confirm": 1.0,
    "context": 1.0,
    "geometry": 1.0,
    "anchor": 0.75,
    "reversal": 0.75,
    "volume": 0.25,
}

SUBTYPE_FEATURES = [
    "mfe_stop_ratio",
    "mae_stop_ratio",
    "entry_precision_6h",
    "entry_precision_12h",
    "entry_precision_24h",
    "anchor_shift_bars",
    "overshoot_bars",
    "fb_breakout_age_bars",
    "reentry_strength",
    "box_width_pct",
    "box_width_atr_mult",
    "edge_distance_atr",
    "overshoot_distance_atr",
    "reentry_distance_atr",
    "confirm_richness",
    "base_confirms",
    "body_reclaim_reversal",
    "prev_reject_combo",
    "engulfing_reversal",
    "reversal_composite_score",
    "transition_risk",
    "box_confidence",
    "h4_range_usable",
    "method_agreement",
    "fb_reentry_vs_breakout_ratio",
    "volume_rank_24",
    "volume_hour_norm",
]

SECONDARY_MATRIX_COLS = [
    "entry_time",
    "fragile_subtype",
    "fragile_focus_label",
    "exit_reason",
    "net_return",
    "box_width_pct",
    "box_width_atr_mult",
    "edge_distance_atr",
    "overshoot_distance_atr",
    "reentry_distance_atr",
    "anchor_shift_bars",
    "overshoot_bars",
    "fb_breakout_age_bars",
    "body_reclaim_reversal",
    "prev_reject_combo",
    "engulfing_reversal",
    "reversal_composite_score",
    "fb_reentry_vs_breakout_ratio",
    "volume_rank_24",
    "volume_hour_norm",
    "transition_risk",
    "box_confidence",
    "confirm_richness",
    "reentry_strength",
    "market_phase",
    "session",
]


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_bool_like(series: pd.Series) -> bool:
    return str(series.dtype) == "bool"


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if _is_bool_like(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _feature_family(feature: str) -> str:
    for family, features in ENTRY_FEATURE_FAMILIES.items():
        if feature in features:
            return family
    return "other"


def add_fragile_realized_features(candidate: pd.DataFrame) -> pd.DataFrame:
    out = candidate.copy()
    entry_price = pd.to_numeric(out["close"], errors="coerce")
    stop_distance = pd.to_numeric(out["entry_stop_distance"], errors="coerce")
    out["entry_stop_rate"] = stop_distance / entry_price.replace(0.0, np.nan)
    out["mfe_stop_ratio"] = pd.to_numeric(out["forward_mfe_6h"], errors="coerce") / out["entry_stop_rate"].replace(0.0, np.nan)
    out["mae_stop_ratio"] = pd.to_numeric(out["forward_mae_6h"], errors="coerce") / out["entry_stop_rate"].replace(0.0, np.nan)
    out["upside_buffer_ratio"] = out["mfe_stop_ratio"] - out["mae_stop_ratio"]
    out["meaningful_upside_hit"] = out["mfe_stop_ratio"] >= HIGH_UPSIDE_CUTOFF
    out["poor_early_progress"] = out["mfe_stop_ratio"] <= HOPELESS_CUTOFF
    return out


def assign_fragile_subtypes(candidate: pd.DataFrame) -> pd.DataFrame:
    out = add_fragile_realized_features(candidate)
    fragile = out["fragile_trade"].fillna(False)
    hopeless = fragile & (out["mfe_stop_ratio"] <= HOPELESS_CUTOFF)
    high_upside = fragile & (out["mfe_stop_ratio"] >= HIGH_UPSIDE_CUTOFF)
    out["fragile_subtype"] = np.select(
        [hopeless, high_upside, fragile],
        ["fragile_hopeless", "fragile_high_upside", "fragile_mixed"],
        default="not_fragile",
    )
    out["fragile_focus_label"] = np.select(
        [out["fragile_subtype"] == "fragile_hopeless", out["fragile_subtype"] == "fragile_high_upside"],
        ["hopeless", "high_upside"],
        default="other",
    )
    return out

def build_subtype_tables(root: Path, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fragile = candidate[candidate["fragile_trade"]].copy()
    focus = fragile[fragile["fragile_focus_label"].isin(["hopeless", "high_upside"])].copy()

    counts = fragile.groupby(["fragile_subtype", "exit_reason"], dropna=False).agg(
        trades=("entry_time", "size"),
        avg_return=("net_return", "mean"),
        median_mfe_stop_ratio=("mfe_stop_ratio", "median"),
        median_mae_stop_ratio=("mae_stop_ratio", "median"),
    ).reset_index()
    counts.to_csv(root / "fragile_subtype_counts.csv", index=False)

    split_rows = []
    for feature in SUBTYPE_FEATURES:
        if feature not in fragile.columns:
            continue
        for label, group in fragile.groupby("fragile_subtype", dropna=False):
            values = _coerce_numeric(group[feature])
            split_rows.append(
                {
                    "feature": feature,
                    "fragile_subtype": label,
                    "trades": int(len(group)),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                }
            )
    feature_split = pd.DataFrame(split_rows)
    feature_split.to_csv(root / "fragile_subtype_feature_split.csv", index=False)

    hopeless = focus[focus["fragile_focus_label"] == "hopeless"]
    high_upside = focus[focus["fragile_focus_label"] == "high_upside"]
    effect_rows = []
    for feature in SUBTYPE_FEATURES:
        if feature not in focus.columns:
            continue
        hopeless_values = _coerce_numeric(hopeless[feature])
        upside_values = _coerce_numeric(high_upside[feature])
        effect_rows.append(
            {
                "feature": feature,
                "hopeless_mean": float(hopeless_values.mean()),
                "high_upside_mean": float(upside_values.mean()),
                "hopeless_median": float(hopeless_values.median()),
                "high_upside_median": float(upside_values.median()),
                "effect_size_high_upside_minus_hopeless": effect_size(upside_values, hopeless_values),
                "abs_median_gap": float(abs(upside_values.median() - hopeless_values.median())),
            }
        )
    effects = pd.DataFrame(effect_rows).sort_values(
        "effect_size_high_upside_minus_hopeless",
        key=lambda series: pd.to_numeric(series, errors="coerce").abs(),
        ascending=False,
    )
    effects.to_csv(root / "hopeless_vs_high_upside_effect_sizes.csv", index=False)

    rankings = effects.copy()
    rankings["rank"] = np.arange(1, len(rankings) + 1)
    rankings["family"] = rankings["feature"].map(_feature_family)
    rankings.to_csv(root / "fragile_subtype_feature_rankings.csv", index=False)

    write_md(
        root / "fragile_hopeless_vs_high_upside_definition_note.md",
        [
            "# Fragile Hopeless vs High-Upside Definition Note",
            "",
            "## Taxonomy",
            f"- `fragile_hopeless`: baseline-fragile trade with `mfe_stop_ratio <= {HOPELESS_CUTOFF:.2f}`.",
            f"- `fragile_high_upside`: baseline-fragile trade with `mfe_stop_ratio >= {HIGH_UPSIDE_CUTOFF:.2f}`.",
            f"- `fragile_mixed`: baseline-fragile trade with `{HOPELESS_CUTOFF:.2f} < mfe_stop_ratio < {HIGH_UPSIDE_CUTOFF:.2f}`.",
            "- `mfe_stop_ratio` uses forward 6h favorable excursion divided by entry stop rate.",
            "",
            "## Rationale",
            "- This is a post-hoc research taxonomy only. It does not define live trading logic.",
            "- The split is intentionally transparent: the label asks whether a fragile trade showed meaningful upside relative to its own stop distance soon after entry.",
        ],
    )
    return counts, feature_split, effects, focus


def build_interaction_audit(root: Path, focus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_pairs = [
        ("anchor_shift_bars", "box_confidence"),
        ("anchor_shift_bars", "reversal_composite_score"),
        ("box_width_atr_mult", "confirm_richness"),
        ("box_width_atr_mult", "box_confidence"),
        ("reversal_composite_score", "edge_distance_atr"),
        ("fb_reentry_vs_breakout_ratio", "anchor_shift_bars"),
        ("transition_risk", "reentry_strength"),
    ]

    audit_rows = []
    pair_rows = []
    hopeless_flag = (focus["fragile_focus_label"] == "hopeless").astype(float)
    base_rate = float(hopeless_flag.mean()) if len(hopeless_flag) else np.nan
    min_count = max(3, int(len(focus) * 0.08))

    for left, right in interaction_pairs:
        if left not in focus.columns or right not in focus.columns:
            continue
        left_series = _coerce_numeric(focus[left])
        right_series = _coerce_numeric(focus[right])
        left_cut = float(left_series.median()) if left_series.notna().any() else np.nan
        right_cut = float(right_series.median()) if right_series.notna().any() else np.nan
        left_state = np.where(left_series >= left_cut, "high", "low")
        right_state = np.where(right_series >= right_cut, "high", "low")

        local = focus[[left, right, "fragile_focus_label"]].copy()
        local["left_state"] = left_state
        local["right_state"] = right_state
        local["hopeless_flag"] = hopeless_flag.values
        grouped = local.groupby(["left_state", "right_state"], dropna=False).agg(
            trades=("hopeless_flag", "size"),
            hopeless_rate=("hopeless_flag", "mean"),
        ).reset_index()
        grouped["pair"] = f"{left} x {right}"
        grouped["base_hopeless_rate"] = base_rate
        grouped["delta_vs_base"] = grouped["hopeless_rate"] - base_rate
        grouped["supported"] = grouped["trades"] >= min_count
        audit_rows.extend(grouped.to_dict("records"))

        supported = grouped[grouped["supported"]].copy()
        if supported.empty:
            pair_gap = np.nan
            top_combo = ""
            bottom_combo = ""
        else:
            supported = supported.sort_values("hopeless_rate")
            pair_gap = float(supported["hopeless_rate"].iloc[-1] - supported["hopeless_rate"].iloc[0])
            top_combo = f"{supported['left_state'].iloc[-1]}|{supported['right_state'].iloc[-1]}"
            bottom_combo = f"{supported['left_state'].iloc[0]}|{supported['right_state'].iloc[0]}"
        pair_rows.append(
            {
                "pair": f"{left} x {right}",
                "left_feature": left,
                "right_feature": right,
                "supported_gap": pair_gap,
                "supported_combo_count": int(len(supported)),
                "top_combo": top_combo,
                "bottom_combo": bottom_combo,
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(root / "fragile_subtype_interaction_audit.csv", index=False)
    pair_rank = pd.DataFrame(pair_rows).sort_values("supported_gap", ascending=False, na_position="last")

    lines = ["# Interaction Rankings", "", "## Observation"]
    for row in pair_rank.head(6).itertuples(index=False):
        if pd.notna(row.supported_gap):
            lines.append(
                f"- {row.pair}: supported_gap={row.supported_gap:.3f} with top combo `{row.top_combo}` and bottom combo `{row.bottom_combo}`."
            )
    lines.extend(
        [
            "",
            "## Conclusion",
            "- Interaction effects are only treated as informative when the combo gap is supported by enough fragile-subtype observations.",
            "- Weak standalone variables remain tracked if they widen subtype separation in combination.",
        ]
    )
    write_md(root / "interaction_rankings.md", lines)
    return audit, pair_rank

def build_hopeless_proxy_spec(focus: pd.DataFrame) -> pd.DataFrame:
    hopeless = focus[focus["fragile_focus_label"] == "hopeless"]
    high_upside = focus[focus["fragile_focus_label"] == "high_upside"]
    rows = []
    for family, features in ENTRY_FEATURE_FAMILIES.items():
        best_row: dict[str, object] | None = None
        for feature in features:
            if feature not in focus.columns:
                continue
            if _is_bool_like(focus[feature]):
                hopeless_mean = float(hopeless[feature].astype(float).mean())
                upside_mean = float(high_upside[feature].astype(float).mean())
                direction = "high" if hopeless_mean >= upside_mean else "low"
                threshold = 0.5
                effect = hopeless_mean - upside_mean
            else:
                hopeless_values = pd.to_numeric(hopeless[feature], errors="coerce")
                upside_values = pd.to_numeric(high_upside[feature], errors="coerce")
                hopeless_med = float(hopeless_values.median())
                upside_med = float(upside_values.median())
                direction = "high" if hopeless_med >= upside_med else "low"
                threshold = (hopeless_med + upside_med) / 2.0
                effect = effect_size(hopeless_values, upside_values)
            row = {
                "family": family,
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "effect_size_hopeless_minus_high_upside": effect,
                "weight": FAMILY_WEIGHTS[family],
            }
            if best_row is None or abs(float(row["effect_size_hopeless_minus_high_upside"])) > abs(float(best_row["effect_size_hopeless_minus_high_upside"])):
                best_row = row
        if best_row is not None:
            rows.append(best_row)
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)


def apply_hopeless_proxy(candidate: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    out = candidate.copy()
    out["hopeless_proxy_score"] = 0.0
    out["hopeless_proxy_score_max"] = 0.0
    for row in spec.itertuples(index=False):
        feature = row.feature
        weight = float(row.weight)
        hit_name = f"hopeless_hit_{feature}"
        out[hit_name] = False
        if feature not in out.columns:
            continue
        if _is_bool_like(out[feature]):
            hit = out[feature] if row.direction == "high" else ~out[feature]
        else:
            series = pd.to_numeric(out[feature], errors="coerce")
            hit = series >= float(row.threshold) if row.direction == "high" else series <= float(row.threshold)
        out[hit_name] = hit.fillna(False)
        out["hopeless_proxy_score"] += hit.fillna(False).astype(float) * weight
        out["hopeless_proxy_score_max"] += weight
    out["hopeless_proxy_score_pct"] = out["hopeless_proxy_score"] / out["hopeless_proxy_score_max"].replace(0.0, np.nan)
    out["hopeless_proxy_bucket"] = np.select(
        [out["hopeless_proxy_score_pct"] >= 0.75, out["hopeless_proxy_score_pct"] <= 0.25],
        ["hopeless_like", "upside_like"],
        default="mixed_like",
    )
    return out


def build_secondary_matrix(root: Path, candidate: pd.DataFrame) -> pd.DataFrame:
    matrix = candidate[candidate["fragile_trade"]].copy()
    matrix = matrix[[col for col in SECONDARY_MATRIX_COLS if col in matrix.columns]]
    matrix.to_csv(root / "fragile_secondary_observation_matrix.csv", index=False)
    return matrix


def build_proxy_evaluation(root: Path, scored: pd.DataFrame) -> pd.DataFrame:
    fragile = scored[scored["fragile_trade"]].copy()
    evaluation = fragile.groupby("hopeless_proxy_bucket", dropna=False).agg(
        trades=("entry_time", "size"),
        hopeless_rate=("fragile_focus_label", lambda values: float((values == "hopeless").mean())),
        high_upside_rate=("fragile_focus_label", lambda values: float((values == "high_upside").mean())),
        avg_score=("hopeless_proxy_score_pct", "mean"),
    ).reset_index()
    evaluation.to_csv(root / "fragile_proxy_bucket_evaluation.csv", index=False)
    return evaluation


def write_focus_reports(
    root: Path,
    counts: pd.DataFrame,
    effects: pd.DataFrame,
    pair_rank: pd.DataFrame,
    proxy_eval: pd.DataFrame,
) -> None:
    top_features = effects.head(8)
    count_summary = counts.groupby("fragile_subtype", dropna=False)["trades"].sum().to_dict()

    write_md(
        root / "FRAGILE_SUBTYPE_AUDIT.md",
        [
            "# Fragile Subtype Audit",
            "",
            "## Observation",
            f"- fragile_hopeless trades: {int(count_summary.get('fragile_hopeless', 0))}.",
            f"- fragile_high_upside trades: {int(count_summary.get('fragile_high_upside', 0))}.",
            f"- fragile_mixed trades: {int(count_summary.get('fragile_mixed', 0))}.",
            *[
                f"- {row.feature}: effect(high_upside-hopeless)={row.effect_size_high_upside_minus_hopeless:.3f}."
                for row in top_features.itertuples(index=False)
            ],
            "",
            "## Conclusion",
            "- The subtype split is only considered meaningful if hopeless vs high-upside show repeatable separation on entry-time features, not just post-hoc excursion.",
            "- Mixed fragile trades remain intentionally separate to avoid forcing a false binary where overlap is high.",
        ],
    )

    comparison_lines = ["# Fragile Hopeless vs High-Upside", "", "## Observation"]
    for row in pair_rank.head(5).itertuples(index=False):
        if pd.notna(row.supported_gap):
            comparison_lines.append(f"- interaction {row.pair}: supported gap={row.supported_gap:.3f}.")
    comparison_lines.extend(["", "## Proxy Audit"])
    for row in proxy_eval.itertuples(index=False):
        comparison_lines.append(
            f"- bucket={row.hopeless_proxy_bucket}: trades={int(row.trades)}, hopeless_rate={row.hopeless_rate:.3f}, high_upside_rate={row.high_upside_rate:.3f}."
        )
    comparison_lines.extend(
        [
            "",
            "## Conclusion",
            "- A narrow hopeless-like proxy is only justified if the hopeless-like bucket is materially cleaner than the rest of the fragile set.",
            "- Secondary variables stay alive even if they contribute only through interactions or ranking, not standalone gates.",
        ]
    )
    write_md(root / "FRAGILE_HOPELESS_VS_HIGH_UPSIDE.md", comparison_lines)


def filter_mask_for_variant(scored: pd.DataFrame, variant_name: str) -> pd.Series:
    very_strong_hopeless_like = scored["hopeless_proxy_score_pct"].fillna(0.0) >= 0.90
    low_quality_gate = scored["score_bucket"].astype(str) == "fragile"
    anchor_risk_cut = pd.to_numeric(scored["anchor_shift_bars"], errors="coerce").median()
    anchor_risk = scored["anchor_shift_bars"].fillna(0.0) >= anchor_risk_cut
    weak_reversal = ~scored["body_reclaim_reversal"].fillna(False)

    if variant_name == "fragile_refinement_audit_only":
        return pd.Series(False, index=scored.index)
    if variant_name == "fragile_hopeless_filter_1":
        return scored["fragile_subtype"].eq("fragile_hopeless")
    if variant_name == "fragile_hopeless_filter_2":
        return low_quality_gate & very_strong_hopeless_like & anchor_risk & weak_reversal
    raise ValueError(f"Unknown variant: {variant_name}")


def run_refinement_variant(control_result: dict, scored: pd.DataFrame, variant_name: str) -> dict:
    config = deep_copy_config()
    merged = control_result["merged"].copy()
    execution_false_break = control_result["execution"]["false_break"].copy()
    execution_boundary = control_result["execution"]["boundary"].copy()

    filtered = execution_false_break.copy()
    mask = filter_mask_for_variant(scored, variant_name)
    filtered_ts = set(scored.loc[mask, "timestamp"].tolist())
    row_mask = filtered["timestamp"].isin(filtered_ts) & (filtered["exec_signal"] != 0)
    filtered.loc[row_mask, "exec_signal"] = 0
    filtered.loc[row_mask, "gate_pass"] = False
    filtered.loc[row_mask, "blocked_reason"] = filtered.loc[row_mask, "blocked_reason"].fillna("").astype(str).apply(
        lambda value: f"{value}|{variant_name}".strip("|")
    )

    execution_outputs = {
        "false_break": filtered,
        "boundary": execution_boundary,
        "combined": pd.concat([filtered, execution_boundary], ignore_index=True).sort_values(["timestamp", "engine"]).reset_index(drop=True),
    }
    risk_cfg = RiskConfig(**config["risk"])
    risk_outputs = apply_risk_layer(merged, execution_outputs, risk_cfg)
    bt_cfg = BacktestConfig(**config["backtest"])
    backtest_outputs = run_backtest(risk_outputs["trades_false_break"], risk_outputs["trades_boundary"], bt_cfg)
    return {
        "variant": variant_name,
        "execution": execution_outputs,
        "risk": risk_outputs,
        "backtest": backtest_outputs,
        "filtered_timestamps": filtered_ts,
    }

def build_variant_outputs(
    root: Path,
    control_audit: pd.DataFrame,
    scored: pd.DataFrame,
    variant_results: dict[str, dict],
    annualization_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    control_trades = parse_times(control_audit.copy(), ["entry_time", "exit_time"])
    control_total_return = float((1.0 + control_trades["net_return"]).prod() - 1.0) if not control_trades.empty else 0.0
    control_time_stop_max_sum = float(control_trades.loc[control_trades["exit_reason"] == "time_stop_max", "net_return"].sum())

    scored_lookup = scored[[
        "entry_time",
        "fragile_subtype",
        "fragile_focus_label",
        "hopeless_proxy_score_pct",
        "hopeless_proxy_bucket",
    ]].copy()

    summary_rows = []
    impact_rows = []
    retention_frames = []
    sensitivity_frames = []

    for name, result in variant_results.items():
        trades = parse_times(result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        summary = summarize_returns(trades, annualization_hours)
        summary_rows.append(
            {
                "variant": name,
                **summary,
                "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
                "time_stop_early_count": int((trades["exit_reason"] == "time_stop_early").sum()) if not trades.empty else 0,
                "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
                "state_stop_count": int((trades["exit_reason"] == "state_stop").sum()) if not trades.empty else 0,
            }
        )

        sensitivity = winner_sensitivity(trades, annualization_hours)
        sensitivity.insert(0, "variant", name)
        sensitivity_frames.append(sensitivity)

        if name == "control":
            continue
        filtered_ts = result["filtered_timestamps"]
        removed = control_trades[control_trades["entry_time"].isin(filtered_ts)].copy()
        retained = control_trades[~control_trades["entry_time"].isin(filtered_ts)].copy()
        if not removed.empty:
            removed = removed.merge(scored_lookup, on="entry_time", how="left")
        impact_rows.append(
            {
                "variant": name,
                "removed_trades": int(len(removed)),
                "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()) if not removed.empty else 0,
                "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()) if not removed.empty else 0,
                "removed_fragile_mixed": int((removed["fragile_subtype"] == "fragile_mixed").sum()) if not removed.empty else 0,
                "removed_non_fragile": int((removed["fragile_subtype"] == "not_fragile").sum()) if not removed.empty else 0,
                "removed_total_return_sum": float(removed["net_return"].sum()) if not removed.empty else 0.0,
                "delta_total_return_vs_control": summary["total_return"] - control_total_return,
                "delta_time_stop_max_sum_vs_control": (
                    float(retained.loc[retained["exit_reason"] == "time_stop_max", "net_return"].sum()) if not retained.empty else 0.0
                ) - control_time_stop_max_sum,
            }
        )
        merged_retention = control_trades.merge(scored_lookup, on="entry_time", how="left")
        merged_retention["variant"] = name
        merged_retention["retained"] = ~merged_retention["entry_time"].isin(filtered_ts)
        retention_frames.append(merged_retention)

    summary_df = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
    summary_df.to_csv(root / "variant_summary.csv", index=False)
    impact_df = pd.DataFrame(impact_rows)
    impact_df.to_csv(root / "fragile_refinement_trade_impact.csv", index=False)
    retained_df = pd.concat(retention_frames, ignore_index=True)
    retained_df.to_csv(root / "retained_vs_removed_fragile_subtypes.csv", index=False)
    winner_df = pd.concat(sensitivity_frames, ignore_index=True)
    winner_df.to_csv(root / "winner_sensitivity_by_variant.csv", index=False)
    return summary_df, impact_df, retained_df, winner_df


def write_summary_report(
    root: Path,
    counts: pd.DataFrame,
    effects: pd.DataFrame,
    pair_rank: pd.DataFrame,
    proxy_eval: pd.DataFrame,
    variant_summary: pd.DataFrame,
    impact_df: pd.DataFrame,
) -> None:
    count_map = counts.groupby("fragile_subtype", dropna=False)["trades"].sum().to_dict()
    best_feature = effects.iloc[0] if not effects.empty else None
    best_pair = pair_rank.iloc[0] if not pair_rank.empty else None
    control = variant_summary.loc[variant_summary["variant"] == "control"].iloc[0]
    audit_only = variant_summary.loc[variant_summary["variant"] == "fragile_refinement_audit_only"].iloc[0]
    filter1 = variant_summary.loc[variant_summary["variant"] == "fragile_hopeless_filter_1"].iloc[0]
    filter2 = variant_summary.loc[variant_summary["variant"] == "fragile_hopeless_filter_2"]
    filter2_row = filter2.iloc[0] if not filter2.empty else None

    hopeless_bucket = proxy_eval.loc[proxy_eval["hopeless_proxy_bucket"] == "hopeless_like"]
    hopeless_bucket_rate = float(hopeless_bucket["hopeless_rate"].iloc[0]) if not hopeless_bucket.empty else np.nan
    hopeless_bucket_upside = float(hopeless_bucket["high_upside_rate"].iloc[0]) if not hopeless_bucket.empty else np.nan
    filter1_impact = impact_df.loc[impact_df["variant"] == "fragile_hopeless_filter_1"]
    removed_hopeless = int(filter1_impact["removed_fragile_hopeless"].iloc[0]) if not filter1_impact.empty else 0
    removed_high_upside = int(filter1_impact["removed_fragile_high_upside"].iloc[0]) if not filter1_impact.empty else 0

    lines = [
        "# Fragile Label Refinement Summary",
        "",
        "## Final Answers",
        f"1. Fragile entries split into hopeless vs high-upside with meaningful separation: {'yes, but partial' if best_feature is not None and abs(float(best_feature.effect_size_high_upside_minus_hopeless)) >= 0.40 else 'overlap remains too large'}.",
        f"2. Strongest separating variable: {best_feature.feature} (effect={float(best_feature.effect_size_high_upside_minus_hopeless):.3f})." if best_feature is not None else "2. Strongest separating variable: insufficient evidence.",
        f"3. Interaction effects matter more than single features: {'yes in the top supported pairs' if best_pair is not None and pd.notna(best_pair.supported_gap) and float(best_pair.supported_gap) >= 0.25 else 'not clearly more than single features'}.",
        "4. Secondary observation variables that remain worth tracking: ATR geometry, anchor drift descriptors, richer reversal descriptors, and weak volume descriptors remain in the feature and interaction tables.",
        f"5. Narrow fragile-harvesting / filtering rule worth further testing: {'yes, as a very narrow hopeless-like suppression only' if np.isfinite(hopeless_bucket_rate) and hopeless_bucket_rate >= 0.65 and hopeless_bucket_upside <= 0.20 else 'not yet; overlap is still too large'}.",
        f"6. Overlap still too large to support safe refinement: {'yes' if not (np.isfinite(hopeless_bucket_rate) and hopeless_bucket_rate >= 0.65 and hopeless_bucket_upside <= 0.20) else 'no, but only for a narrow rule'}.",
        "7. The single most justified next optimization direction is another interpretable preserve-vs-suppress ranking pass inside already-admitted trades, using subtype-aware interactions before any broader filter promotion.",
        "",
        "## Observation",
        f"- fragile_hopeless={int(count_map.get('fragile_hopeless', 0))}, fragile_high_upside={int(count_map.get('fragile_high_upside', 0))}, fragile_mixed={int(count_map.get('fragile_mixed', 0))}.",
        f"- control total_return={control['total_return']:.4f}; audit_only total_return={audit_only['total_return']:.4f}; filter_1 total_return={filter1['total_return']:.4f} (oracle hopeless-label upper bound).",
        f"- hopeless_like bucket hopeless_rate={hopeless_bucket_rate:.3f}, high_upside_rate={hopeless_bucket_upside:.3f}." if np.isfinite(hopeless_bucket_rate) else "- hopeless_like bucket was not populated strongly enough to judge.",
        f"- fragile_hopeless_filter_1 removed hopeless={removed_hopeless} and high_upside={removed_high_upside}.",
    ]
    if filter2_row is not None:
        lines.append(f"- fragile_hopeless_filter_2 total_return={filter2_row['total_return']:.4f}.")
    write_md(root / "FRAGILE_LABEL_REFINEMENT_SUMMARY.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated fragile label refinement research.")
    parser.add_argument("--outdir", default="outputs/fragile_label_refinement_round")
    args = parser.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(FROZEN_LAYER12_CONTROL_CONFIG, root / "frozen_control_config.yaml")

    cfg = deep_copy_config()
    _, boxes_1h, boxes_4h, enriched_1h = load_closed_base_frames(cfg)
    control_result = run_research_variant(cfg, boxes_1h, boxes_4h, enriched_1h, root / "control", CONTROL_VARIANT)

    control_audit = add_forward_entry_quality(build_trade_audit(control_result), control_result["merged"])
    control_audit.to_csv(root / "control_trade_audit.csv", index=False)

    candidate = build_execution_candidate_table(control_result, control_audit)
    coarse_spec = build_score_spec(candidate)
    candidate = apply_entry_quality_score(candidate, coarse_spec)
    coarse_spec.to_csv(root / "prior_round_entry_quality_spec.csv", index=False)
    labeled = assign_fragile_subtypes(candidate)
    labeled.to_csv(root / "control_entry_candidates_fragile_labeled.csv", index=False)

    counts, _, effects, focus = build_subtype_tables(root, labeled)
    _, pair_rank = build_interaction_audit(root, focus)

    spec = build_hopeless_proxy_spec(focus)
    spec.to_csv(root / "fragile_hopeless_proxy_spec.csv", index=False)
    scored = apply_hopeless_proxy(labeled, spec)
    scored.to_csv(root / "control_entry_candidates_hopeless_proxy.csv", index=False)

    proxy_eval = build_proxy_evaluation(root, scored)
    build_secondary_matrix(root, scored)
    write_focus_reports(root, counts, effects, pair_rank, proxy_eval)

    variant_results = {
        "control": {
            "variant": "control",
            "backtest": control_result["backtest"],
            "filtered_timestamps": set(),
        },
        "fragile_refinement_audit_only": run_refinement_variant(control_result, scored, "fragile_refinement_audit_only"),
        "fragile_hopeless_filter_1": run_refinement_variant(control_result, scored, "fragile_hopeless_filter_1"),
    }

    hopeless_like = proxy_eval.loc[proxy_eval["hopeless_proxy_bucket"] == "hopeless_like"]
    if not hopeless_like.empty:
        hopeless_rate = float(hopeless_like["hopeless_rate"].iloc[0])
        high_upside_rate = float(hopeless_like["high_upside_rate"].iloc[0])
        if hopeless_rate >= 0.70 and high_upside_rate <= 0.15:
            variant_results["fragile_hopeless_filter_2"] = run_refinement_variant(control_result, scored, "fragile_hopeless_filter_2")

    annualization_hours = int(cfg["backtest"]["annualization_hours"])
    variant_summary, impact_df, _, _ = build_variant_outputs(root, control_audit, scored, variant_results, annualization_hours)

    review = parse_times(control_audit.copy(), ["entry_time", "exit_time"])
    review = review.merge(scored[["entry_time", "fragile_subtype"]], on="entry_time", how="left")
    build_review_chart(
        control_result["merged"],
        review[review["fragile_subtype"] == "fragile_hopeless"].copy(),
        "Fragile Hopeless",
        root / "fragile_hopeless_review.html",
        root / "fragile_hopeless_review.png",
    )
    build_review_chart(
        control_result["merged"],
        review[review["fragile_subtype"] == "fragile_high_upside"].copy(),
        "Fragile High Upside",
        root / "fragile_high_upside_review.html",
        root / "fragile_high_upside_review.png",
    )

    write_summary_report(root, counts, effects, pair_rank, proxy_eval, variant_summary, impact_df)


if __name__ == "__main__":
    main()



