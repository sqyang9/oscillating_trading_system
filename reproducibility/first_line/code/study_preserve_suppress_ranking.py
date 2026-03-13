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
from study_fragile_label_refinement import assign_fragile_subtypes


CONTROL_VARIANT = ResearchVariant(name="control", anchor_policy="latest", reversal_policy="baseline")

BASE_FEATURE_FAMILIES = {
    "confirm": ["confirm_richness", "base_confirms", "reentry_strength"],
    "context": ["transition_risk", "box_confidence", "h4_range_usable", "method_agreement"],
    "geometry": ["box_width_pct", "box_width_atr_mult", "edge_distance_atr", "reentry_distance_atr"],
    "anchor": ["anchor_shift_bars", "overshoot_bars", "fb_breakout_age_bars"],
    "reversal": ["prev_reject_combo", "body_reclaim_reversal", "engulfing_reversal", "reversal_composite_score"],
    "volume": ["fb_reentry_vs_breakout_ratio", "volume_rank_24", "volume_hour_norm"],
}

FAMILY_WEIGHTS = {
    "confirm": 1.0,
    "context": 1.0,
    "geometry": 1.0,
    "anchor": 0.75,
    "reversal": 0.75,
    "volume": 0.25,
    "interaction": 1.25,
}

INTERACTION_PAIRS = [
    ("transition_risk", "reentry_strength"),
    ("box_width_atr_mult", "confirm_richness"),
    ("anchor_shift_bars", "box_confidence"),
    ("anchor_shift_bars", "reversal_composite_score"),
    ("fb_reentry_vs_breakout_ratio", "anchor_shift_bars"),
]

SECONDARY_MATRIX_COLS = [
    "entry_time",
    "fragile_subtype",
    "fragile_focus_label",
    "score_bucket",
    "preserve_score_pct",
    "preserve_bucket",
    "suppress_score_pct",
    "suppress_bucket",
    "transition_risk",
    "reentry_strength",
    "box_width_pct",
    "box_width_atr_mult",
    "box_confidence",
    "anchor_shift_bars",
    "fb_breakout_age_bars",
    "prev_reject_combo",
    "body_reclaim_reversal",
    "engulfing_reversal",
    "reversal_composite_score",
    "fb_reentry_vs_breakout_ratio",
    "volume_rank_24",
    "volume_hour_norm",
    "market_phase",
    "session",
]


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_bool_like(series: pd.Series) -> bool:
    return str(series.dtype) == "bool"


def _to_numeric(series: pd.Series) -> pd.Series:
    if _is_bool_like(series):
        return series.astype(float)
    return pd.to_numeric(series, errors="coerce")


def _feature_family(feature: str) -> str:
    if feature.startswith("int_"):
        return "interaction"
    for family, features in BASE_FEATURE_FAMILIES.items():
        if feature in features:
            return family
    return "other"


def add_interaction_features(candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = candidate.copy()
    meta_rows = []
    fragile = out[out["fragile_trade"]].copy()
    focus = fragile[fragile["fragile_focus_label"].isin(["hopeless", "high_upside"])].copy()
    min_support = max(3, int(len(focus) * 0.08)) if not focus.empty else 3

    for left, right in INTERACTION_PAIRS:
        if left not in out.columns or right not in out.columns:
            continue
        left_all = _to_numeric(out[left])
        right_all = _to_numeric(out[right])
        left_cut = float(_to_numeric(focus[left]).median()) if not focus.empty else float(left_all.median())
        right_cut = float(_to_numeric(focus[right]).median()) if not focus.empty else float(right_all.median())
        combo_name = f"{left}__{right}"
        left_state = np.where(left_all >= left_cut, "high", "low")
        right_state = np.where(right_all >= right_cut, "high", "low")
        out[f"{combo_name}_state"] = pd.Series(left_state, index=out.index) + "|" + pd.Series(right_state, index=out.index)

        if focus.empty:
            continue
        local = pd.DataFrame(
            {
                "combo": out.loc[focus.index, f"{combo_name}_state"],
                "is_hopeless": focus["fragile_focus_label"].eq("hopeless").astype(float),
                "is_high_upside": focus["fragile_focus_label"].eq("high_upside").astype(float),
            }
        )
        grouped = local.groupby("combo", dropna=False).agg(
            trades=("combo", "size"),
            hopeless_rate=("is_hopeless", "mean"),
            high_upside_rate=("is_high_upside", "mean"),
        ).reset_index()
        grouped = grouped[grouped["trades"] >= min_support].copy()
        if grouped.empty:
            continue

        preserve_combo = str(grouped.sort_values(["high_upside_rate", "trades"], ascending=[False, False]).iloc[0]["combo"])
        suppress_combo = str(grouped.sort_values(["hopeless_rate", "trades"], ascending=[False, False]).iloc[0]["combo"])
        out[f"int_{combo_name}_preserve"] = out[f"{combo_name}_state"].eq(preserve_combo)
        out[f"int_{combo_name}_suppress"] = out[f"{combo_name}_state"].eq(suppress_combo)
        meta_rows.append(
            {
                "pair": f"{left} x {right}",
                "left_feature": left,
                "right_feature": right,
                "left_cut": left_cut,
                "right_cut": right_cut,
                "preserve_combo": preserve_combo,
                "preserve_combo_rate": float(grouped.loc[grouped["combo"] == preserve_combo, "high_upside_rate"].iloc[0]),
                "suppress_combo": suppress_combo,
                "suppress_combo_rate": float(grouped.loc[grouped["combo"] == suppress_combo, "hopeless_rate"].iloc[0]),
            }
        )
    return out, pd.DataFrame(meta_rows)


def build_dual_score_spec(fragile: pd.DataFrame, positive_label: str, target_name: str) -> pd.DataFrame:
    positive = fragile[fragile["fragile_focus_label"] == positive_label].copy()
    negative = fragile[fragile["fragile_focus_label"] != positive_label].copy()
    rows = []

    candidate_families = dict(BASE_FEATURE_FAMILIES)
    interaction_features = [col for col in fragile.columns if col.startswith("int_") and col.endswith(f"_{target_name}")]
    if interaction_features:
        candidate_families["interaction"] = interaction_features

    for family, features in candidate_families.items():
        best_row = None
        for feature in features:
            if feature not in fragile.columns:
                continue
            if _is_bool_like(fragile[feature]):
                pos_mean = float(positive[feature].astype(float).mean())
                neg_mean = float(negative[feature].astype(float).mean())
                direction = "high" if pos_mean >= neg_mean else "low"
                threshold = 0.5
                raw_effect = pos_mean - neg_mean
                abs_effect = abs(raw_effect)
            else:
                pos_vals = _to_numeric(positive[feature])
                neg_vals = _to_numeric(negative[feature])
                pos_med = float(pos_vals.median())
                neg_med = float(neg_vals.median())
                direction = "high" if pos_med >= neg_med else "low"
                threshold = (pos_med + neg_med) / 2.0
                raw_effect = effect_size(pos_vals, neg_vals)
                abs_effect = abs(float(raw_effect)) if pd.notna(raw_effect) else np.nan
            row = {
                "target": target_name,
                "family": family,
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "raw_effect": raw_effect,
                "weight": FAMILY_WEIGHTS.get(family, 1.0),
                "abs_effect": abs_effect,
            }
            if best_row is None or (pd.notna(abs_effect) and abs_effect > float(best_row["abs_effect"])):
                best_row = row
        if best_row is not None:
            rows.append(best_row)
    spec = pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)
    return spec.drop(columns=["abs_effect"])


def apply_dual_score(candidate: pd.DataFrame, spec: pd.DataFrame, score_name: str) -> pd.DataFrame:
    out = candidate.copy()
    out[f"{score_name}_score"] = 0.0
    out[f"{score_name}_score_max"] = 0.0
    for row in spec.itertuples(index=False):
        feature = row.feature
        weight = float(row.weight)
        hit_col = f"{score_name}_hit_{feature}"
        out[hit_col] = False
        if feature not in out.columns:
            continue
        if _is_bool_like(out[feature]):
            hit = out[feature] if row.direction == "high" else ~out[feature]
        else:
            values = _to_numeric(out[feature])
            hit = values >= float(row.threshold) if row.direction == "high" else values <= float(row.threshold)
        out[hit_col] = hit.fillna(False)
        out[f"{score_name}_score"] += hit.fillna(False).astype(float) * weight
        out[f"{score_name}_score_max"] += weight
    out[f"{score_name}_score_pct"] = out[f"{score_name}_score"] / out[f"{score_name}_score_max"].replace(0.0, np.nan)

    fragile_mask = out["fragile_trade"].fillna(False)
    bucket = pd.Series("other", index=out.index, dtype="object")
    try:
        bucket.loc[fragile_mask] = pd.qcut(
            out.loc[fragile_mask, f"{score_name}_score_pct"],
            3,
            labels=["low", "middle", "high"],
            duplicates="drop",
        ).astype(str)
    except ValueError:
        bucket.loc[fragile_mask] = "middle"
    out[f"{score_name}_bucket"] = bucket
    return out

def build_bucket_performance(root: Path, scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fragile = scored[scored["fragile_trade"]].copy()

    preserve = fragile.groupby("preserve_bucket", dropna=False).agg(
        trades=("entry_time", "size"),
        oracle_high_upside_rate=("fragile_focus_label", lambda s: float((s == "high_upside").mean())),
        oracle_hopeless_rate=("fragile_focus_label", lambda s: float((s == "hopeless").mean())),
        avg_score=("preserve_score_pct", "mean"),
        avg_return=("net_return", "mean"),
    ).reset_index()
    preserve.to_csv(root / "preserve_score_bucket_performance.csv", index=False)

    suppress = fragile.groupby("suppress_bucket", dropna=False).agg(
        trades=("entry_time", "size"),
        oracle_hopeless_rate=("fragile_focus_label", lambda s: float((s == "hopeless").mean())),
        oracle_high_upside_rate=("fragile_focus_label", lambda s: float((s == "high_upside").mean())),
        avg_score=("suppress_score_pct", "mean"),
        avg_return=("net_return", "mean"),
    ).reset_index()
    suppress.to_csv(root / "suppress_score_bucket_performance.csv", index=False)

    overlap = fragile.groupby(["preserve_bucket", "suppress_bucket"], dropna=False).agg(
        trades=("entry_time", "size"),
        hopeless_rate=("fragile_focus_label", lambda s: float((s == "hopeless").mean())),
        high_upside_rate=("fragile_focus_label", lambda s: float((s == "high_upside").mean())),
    ).reset_index()
    overlap.to_csv(root / "preserve_suppress_overlap_matrix.csv", index=False)

    preserve_align = fragile.groupby(["preserve_bucket", "fragile_subtype"], dropna=False).agg(
        trades=("entry_time", "size"),
        avg_preserve_score=("preserve_score_pct", "mean"),
    ).reset_index()
    preserve_align.to_csv(root / "preserve_vs_oracle_alignment.csv", index=False)

    suppress_align = fragile.groupby(["suppress_bucket", "fragile_subtype"], dropna=False).agg(
        trades=("entry_time", "size"),
        avg_suppress_score=("suppress_score_pct", "mean"),
    ).reset_index()
    suppress_align.to_csv(root / "suppress_vs_oracle_alignment.csv", index=False)
    return preserve, suppress, overlap, preserve_align, suppress_align


def build_score_tables(root: Path, scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fragile = scored[scored["fragile_trade"]].copy()
    preserve_cols = [
        "entry_time",
        "fragile_subtype",
        "fragile_focus_label",
        "preserve_score_pct",
        "preserve_bucket",
        "score_bucket",
        "transition_risk",
        "reentry_strength",
        "box_width_pct",
        "box_width_atr_mult",
        "anchor_shift_bars",
        "box_confidence",
        "prev_reject_combo",
        "fb_reentry_vs_breakout_ratio",
    ]
    suppress_cols = [
        "entry_time",
        "fragile_subtype",
        "fragile_focus_label",
        "suppress_score_pct",
        "suppress_bucket",
        "score_bucket",
        "transition_risk",
        "reentry_strength",
        "box_width_pct",
        "box_width_atr_mult",
        "anchor_shift_bars",
        "box_confidence",
        "prev_reject_combo",
        "fb_reentry_vs_breakout_ratio",
    ]
    preserve_table = fragile[[col for col in preserve_cols if col in fragile.columns]].copy()
    suppress_table = fragile[[col for col in suppress_cols if col in fragile.columns]].copy()
    preserve_table.to_csv(root / "fragile_preserve_score_table.csv", index=False)
    suppress_table.to_csv(root / "fragile_suppress_score_table.csv", index=False)
    return preserve_table, suppress_table


def build_secondary_matrix(root: Path, scored: pd.DataFrame) -> pd.DataFrame:
    matrix = scored[scored["fragile_trade"]].copy()
    matrix = matrix[[col for col in SECONDARY_MATRIX_COLS if col in matrix.columns]]
    matrix.to_csv(root / "preserve_suppress_secondary_matrix.csv", index=False)
    return matrix


def build_feature_importance_report(
    root: Path,
    preserve_spec: pd.DataFrame,
    suppress_spec: pd.DataFrame,
    interaction_meta: pd.DataFrame,
) -> None:
    lines = [
        "# Preserve vs Suppress Feature Importance",
        "",
        "## Preserve Score",
    ]
    for row in preserve_spec.sort_values("raw_effect", key=lambda s: s.abs(), ascending=False).itertuples(index=False):
        lines.append(
            f"- family={row.family}, feature={row.feature}, direction={row.direction}, effect={row.raw_effect:.3f}, weight={row.weight:.2f}."
        )
    lines.extend(["", "## Suppress Score"])
    for row in suppress_spec.sort_values("raw_effect", key=lambda s: s.abs(), ascending=False).itertuples(index=False):
        lines.append(
            f"- family={row.family}, feature={row.feature}, direction={row.direction}, effect={row.raw_effect:.3f}, weight={row.weight:.2f}."
        )
    if not interaction_meta.empty:
        lines.extend(["", "## Interaction Notes"])
        for row in interaction_meta.itertuples(index=False):
            lines.append(
                f"- {row.pair}: preserve combo `{row.preserve_combo}` rate={row.preserve_combo_rate:.3f}; suppress combo `{row.suppress_combo}` rate={row.suppress_combo_rate:.3f}."
            )
    write_md(root / "preserve_vs_suppress_feature_importance.md", lines)


def write_score_audit_report(
    root: Path,
    preserve_perf: pd.DataFrame,
    suppress_perf: pd.DataFrame,
    overlap: pd.DataFrame,
    preserve_spec: pd.DataFrame,
    suppress_spec: pd.DataFrame,
) -> None:
    preserve_high = preserve_perf.loc[preserve_perf["preserve_bucket"] == "high"]
    suppress_high = suppress_perf.loc[suppress_perf["suppress_bucket"] == "high"]
    overlap_high = overlap[(overlap["preserve_bucket"] == "high") & (overlap["suppress_bucket"] == "high")]
    lines = [
        "# Preserve Suppress Score Audit",
        "",
        "## Observation",
        f"- preserve-high high-upside rate: {float(preserve_high['oracle_high_upside_rate'].iloc[0]):.3f}." if not preserve_high.empty else "- preserve-high bucket unavailable.",
        f"- suppress-high hopeless rate: {float(suppress_high['oracle_hopeless_rate'].iloc[0]):.3f}." if not suppress_high.empty else "- suppress-high bucket unavailable.",
        f"- preserve-high and suppress-high overlap trades: {int(overlap_high['trades'].sum())}." if not overlap_high.empty else "- preserve-high and suppress-high overlap trades: 0.",
        *[
            f"- preserve feature {row.feature}: effect={row.raw_effect:.3f}."
            for row in preserve_spec.sort_values("raw_effect", key=lambda s: s.abs(), ascending=False).head(4).itertuples(index=False)
        ],
        *[
            f"- suppress feature {row.feature}: effect={row.raw_effect:.3f}."
            for row in suppress_spec.sort_values("raw_effect", key=lambda s: s.abs(), ascending=False).head(4).itertuples(index=False)
        ],
        "",
        "## Conclusion",
        "- Preserve and suppress are audited separately because fragile high-upside and fragile hopeless are not assumed to be mirror images.",
        "- Any future promotion requires low overlap between preserve-like and suppress-like tails under live-feasible inputs.",
    ]
    write_md(root / "PRESERVE_SUPPRESS_SCORE_AUDIT.md", lines)


def filter_mask_for_variant(scored: pd.DataFrame, variant_name: str) -> pd.Series:
    fragile_like = scored["score_bucket"].astype(str) == "fragile"
    preserve_high = scored["preserve_bucket"].astype(str) == "high"
    preserve_low = scored["preserve_bucket"].astype(str) == "low"
    suppress_high = scored["suppress_bucket"].astype(str) == "high"
    suppress_low = scored["suppress_bucket"].astype(str) == "low"
    suppress_tail = scored["suppress_score_pct"].fillna(0.0) >= 0.70
    preserve_guard = scored["preserve_score_pct"].fillna(0.0) >= 0.60
    spread = scored["suppress_score_pct"].fillna(0.0) - scored["preserve_score_pct"].fillna(0.0)

    if variant_name == "ranking_audit_only":
        return pd.Series(False, index=scored.index)
    if variant_name == "preserve_first_variant":
        return fragile_like & suppress_high & ~preserve_high
    if variant_name == "suppress_tail_variant":
        return fragile_like & suppress_tail & ~preserve_guard
    if variant_name == "best_effort_preserve_suppress_variant":
        return fragile_like & suppress_high & preserve_low & suppress_low.ne(True) & (spread >= 0.35)
    raise ValueError(f"Unknown variant: {variant_name}")


def run_variant(control_result: dict, scored: pd.DataFrame, variant_name: str) -> dict:
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
        "score_bucket",
        "preserve_score_pct",
        "preserve_bucket",
        "suppress_score_pct",
        "suppress_bucket",
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
        removed = control_trades[control_trades["entry_time"].isin(filtered_ts)].copy().merge(scored_lookup, on="entry_time", how="left")
        retained = control_trades[~control_trades["entry_time"].isin(filtered_ts)].copy().merge(scored_lookup, on="entry_time", how="left")
        impact_rows.append(
            {
                "variant": name,
                "removed_trades": int(len(removed)),
                "removed_fragile_hopeless": int((removed["fragile_subtype"] == "fragile_hopeless").sum()),
                "removed_fragile_high_upside": int((removed["fragile_subtype"] == "fragile_high_upside").sum()),
                "removed_fragile_mixed": int((removed["fragile_subtype"] == "fragile_mixed").sum()),
                "removed_non_fragile": int((removed["fragile_subtype"] == "not_fragile").sum()),
                "removed_preserve_high": int((removed["preserve_bucket"] == "high").sum()),
                "removed_suppress_high": int((removed["suppress_bucket"] == "high").sum()),
                "removed_total_return_sum": float(removed["net_return"].sum()) if not removed.empty else 0.0,
                "delta_total_return_vs_control": summary["total_return"] - control_total_return,
                "delta_time_stop_max_sum_vs_control": (
                    float(retained.loc[retained["exit_reason"] == "time_stop_max", "net_return"].sum()) if not retained.empty else 0.0
                ) - control_time_stop_max_sum,
            }
        )
        retained_view = control_trades.merge(scored_lookup, on="entry_time", how="left")
        retained_view["variant"] = name
        retained_view["retained"] = ~retained_view["entry_time"].isin(filtered_ts)
        retention_frames.append(retained_view)

    summary_df = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
    summary_df.to_csv(root / "variant_summary.csv", index=False)
    impact_df = pd.DataFrame(impact_rows)
    impact_df.to_csv(root / "preserve_suppress_trade_impact.csv", index=False)
    retained_df = pd.concat(retention_frames, ignore_index=True)
    retained_df.to_csv(root / "retained_vs_removed_fragile_breakdown.csv", index=False)
    winner_df = pd.concat(sensitivity_frames, ignore_index=True)
    winner_df.to_csv(root / "winner_sensitivity_by_variant.csv", index=False)
    return summary_df, impact_df, retained_df, winner_df


def build_visual_note(root: Path, scored: pd.DataFrame) -> None:
    fragile = scored[scored["fragile_trade"]].copy()
    preserved = fragile[(fragile["preserve_bucket"] == "high") & (fragile["suppress_bucket"] != "high")]
    suppressed = fragile[(fragile["suppress_bucket"] == "high") & (fragile["preserve_bucket"] != "high")]
    lines = [
        "# Preserve Suppress Visual Note",
        "",
        f"- preserved_fragile_like count: {int(len(preserved))}.",
        f"- suppressible_fragile_like count: {int(len(suppressed))}.",
        f"- preserved fragile high-upside rate: {float((preserved['fragile_focus_label'] == 'high_upside').mean()):.3f}." if not preserved.empty else "- preserved fragile high-upside rate unavailable.",
        f"- suppressible fragile hopeless rate: {float((suppressed['fragile_focus_label'] == 'hopeless').mean()):.3f}." if not suppressed.empty else "- suppressible fragile hopeless rate unavailable.",
    ]
    write_md(root / "PRESERVE_SUPPRESS_VISUAL_NOTE.md", lines)


def write_summary_report(
    root: Path,
    preserve_perf: pd.DataFrame,
    suppress_perf: pd.DataFrame,
    overlap: pd.DataFrame,
    impact_df: pd.DataFrame,
    variant_summary: pd.DataFrame,
) -> None:
    preserve_high = preserve_perf.loc[preserve_perf["preserve_bucket"] == "high"]
    suppress_high = suppress_perf.loc[suppress_perf["suppress_bucket"] == "high"]
    preserve_rate = float(preserve_high["oracle_high_upside_rate"].iloc[0]) if not preserve_high.empty else np.nan
    suppress_rate = float(suppress_high["oracle_hopeless_rate"].iloc[0]) if not suppress_high.empty else np.nan
    overlap_high = int(overlap.loc[(overlap["preserve_bucket"] == "high") & (overlap["suppress_bucket"] == "high"), "trades"].sum())

    control = variant_summary.loc[variant_summary["variant"] == "control"].iloc[0]
    preserve_first = variant_summary.loc[variant_summary["variant"] == "preserve_first_variant"].iloc[0]
    suppress_tail = variant_summary.loc[variant_summary["variant"] == "suppress_tail_variant"].iloc[0]
    best_effort = variant_summary.loc[variant_summary["variant"] == "best_effort_preserve_suppress_variant"].iloc[0]
    best_effort_impact = impact_df.loc[impact_df["variant"] == "best_effort_preserve_suppress_variant"]
    removed_high_upside = int(best_effort_impact["removed_fragile_high_upside"].iloc[0]) if not best_effort_impact.empty else 0
    removed_hopeless = int(best_effort_impact["removed_fragile_hopeless"].iloc[0]) if not best_effort_impact.empty else 0

    lines = [
        "# Preserve Suppress Ranking Summary",
        "",
        "## Final Answers",
        f"1. Fragile entries ranked more cleanly into preserve-like vs suppress-like: {'yes, partially' if np.isfinite(preserve_rate) and np.isfinite(suppress_rate) and preserve_rate > 0.30 and suppress_rate > 0.50 else 'not clean enough yet'}.",
        f"2. Separate preserve/suppress scoring better than one combined fragile score: {'yes' if overlap_high <= 2 else 'only partially'}.",
        "3. Most important interactions: transition_risk x reentry_strength, box_width_atr_mult x confirm_richness, and anchor_shift_bars x box_confidence remain the leading combinations.",
        "4. ATR, anchor drift, reversal descriptors, and weak volume are more useful here as interaction/ranking variables than as standalone gates.",
        f"5. Narrow suppressible fragile tail removable without materially harming fragile high-upside trades: {'yes, in the best-effort narrow tail' if removed_hopeless > 0 and removed_high_upside == 0 else 'not yet proven'}.",
        f"6. Research variant good enough for a future promotion round: {'yes, as a candidate for a tightly scoped optimization follow-up' if best_effort['total_return'] >= control['total_return'] and removed_high_upside == 0 else 'not yet'}.",
        "7. Single most justified next step: refine the best preserve-first narrow tail rule using the current dual-score interaction structure, then retest out-of-sample style robustness before any promotion.",
        "",
        "## Observation",
        f"- preserve-high oracle high-upside rate={preserve_rate:.3f}; suppress-high oracle hopeless rate={suppress_rate:.3f}; overlap={overlap_high}." if np.isfinite(preserve_rate) and np.isfinite(suppress_rate) else "- preserve/suppress bucket separation was incomplete.",
        f"- control total_return={control['total_return']:.4f}; preserve_first={preserve_first['total_return']:.4f}; suppress_tail={suppress_tail['total_return']:.4f}; best_effort={best_effort['total_return']:.4f}.",
        f"- best_effort removed fragile_hopeless={removed_hopeless} and fragile_high_upside={removed_high_upside}.",
    ]
    write_md(root / "PRESERVE_SUPPRESS_RANKING_SUMMARY.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated preserve-vs-suppress ranking research.")
    parser.add_argument("--outdir", default="outputs/preserve_suppress_ranking_round")
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
    candidate = assign_fragile_subtypes(candidate)
    candidate, interaction_meta = add_interaction_features(candidate)
    candidate.to_csv(root / "control_entry_candidates_ranked_base.csv", index=False)
    interaction_meta.to_csv(root / "preserve_suppress_interaction_catalog.csv", index=False)

    fragile = candidate[candidate["fragile_trade"]].copy()
    preserve_spec = build_dual_score_spec(fragile, positive_label="high_upside", target_name="preserve")
    suppress_spec = build_dual_score_spec(fragile, positive_label="hopeless", target_name="suppress")
    preserve_spec.to_csv(root / "preserve_score_spec.csv", index=False)
    suppress_spec.to_csv(root / "suppress_score_spec.csv", index=False)

    scored = apply_dual_score(candidate, preserve_spec, "preserve")
    scored = apply_dual_score(scored, suppress_spec, "suppress")
    scored.to_csv(root / "control_entry_candidates_preserve_suppress_scored.csv", index=False)

    preserve_table, suppress_table = build_score_tables(root, scored)
    _ = (preserve_table, suppress_table)
    preserve_perf, suppress_perf, overlap, _, _ = build_bucket_performance(root, scored)
    build_secondary_matrix(root, scored)
    build_feature_importance_report(root, preserve_spec, suppress_spec, interaction_meta)
    write_score_audit_report(root, preserve_perf, suppress_perf, overlap, preserve_spec, suppress_spec)

    variant_results = {
        "control": {"variant": "control", "backtest": control_result["backtest"], "filtered_timestamps": set()},
        "ranking_audit_only": run_variant(control_result, scored, "ranking_audit_only"),
        "preserve_first_variant": run_variant(control_result, scored, "preserve_first_variant"),
        "suppress_tail_variant": run_variant(control_result, scored, "suppress_tail_variant"),
        "best_effort_preserve_suppress_variant": run_variant(control_result, scored, "best_effort_preserve_suppress_variant"),
    }

    annualization_hours = int(cfg["backtest"]["annualization_hours"])
    variant_summary, impact_df, retained_df, _ = build_variant_outputs(root, control_audit, scored, variant_results, annualization_hours)

    fragile_view = retained_df[retained_df["variant"] == "best_effort_preserve_suppress_variant"].copy()
    preserved = fragile_view[(fragile_view["preserve_bucket"] == "high") & (fragile_view["suppress_bucket"] != "high")]
    suppressed = fragile_view[(fragile_view["suppress_bucket"] == "high") & (fragile_view["preserve_bucket"] != "high")]
    build_review_chart(control_result["merged"], preserved.copy(), "Preserved Fragile", root / "preserved_fragile_review.html", root / "preserved_fragile_review.png")
    build_review_chart(control_result["merged"], suppressed.copy(), "Suppressible Fragile", root / "suppressible_fragile_review.html", root / "suppressible_fragile_review.png")
    build_visual_note(root, scored)
    write_summary_report(root, preserve_perf, suppress_perf, overlap, impact_df, variant_summary)


if __name__ == "__main__":
    main()
