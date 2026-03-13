from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backtest import BacktestConfig, run_backtest
from layer12_structural_research_helpers import (
    FROZEN_LAYER12_CONTROL_CONFIG,
    ResearchVariant,
    add_common_bar_features,
    add_forward_entry_quality,
    build_trade_audit,
    deep_copy_config,
    effect_size,
    extract_overshoot_episodes,
    load_closed_base_frames,
    parse_times,
    run_research_variant,
    save_yaml,
    summarize_returns,
    winner_sensitivity,
)
from risk import RiskConfig, apply_risk_layer


CONTROL_VARIANT = ResearchVariant(name="control", anchor_policy="latest", reversal_policy="baseline")

CORE_FAMILY_FEATURES = {
    "confirm": ["base_confirms", "confirm_richness", "reentry_strength"],
    "context": ["box_confidence", "transition_risk"],
    "reversal": ["body_reclaim_reversal", "prev_reject_combo", "engulfing_reversal", "reversal_composite_score"],
}

SECONDARY_FAMILY_FEATURES = {
    "atr": ["box_width_atr_mult", "edge_distance_atr", "overshoot_distance_atr"],
    "anchor": ["anchor_shift_bars", "overshoot_bars", "fb_breakout_age_bars"],
    "volume": ["fb_reentry_vs_breakout_ratio", "volume_rank_24", "volume_hour_norm"],
}

FAMILY_WEIGHTS = {
    "confirm": 1.0,
    "context": 1.0,
    "reversal": 1.0,
    "atr": 0.5,
    "anchor": 0.5,
    "volume": 0.25,
}


def write_md(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_execution_candidate_table(control_result: dict, control_audit: pd.DataFrame) -> pd.DataFrame:
    merged = add_common_bar_features(control_result["merged"], control_result["config"])
    execution = parse_times(control_result["execution"]["false_break"].copy(), ["timestamp"])
    execution = execution[execution["exec_signal"] != 0].copy()

    episodes = extract_overshoot_episodes(control_result["merged"], control_result["config"])
    episode_cols = [
        "reentry_time",
        "anchor_shift_bars",
        "overshoot_bars",
        "first_anchor_age_at_reentry",
        "last_anchor_age_at_reentry",
        "first_within_window",
        "last_within_window",
    ]
    episodes = episodes[episodes["status"] == "reentry"][episode_cols].rename(columns={"reentry_time": "timestamp"})

    merge_cols = [
        "timestamp",
        "bar_index",
        "open",
        "high",
        "low",
        "close",
        "box_midline",
        "atr",
        "box_upper_edge",
        "box_lower_edge",
        "box_width",
        "box_width_pct",
        "box_width_atr_mult",
        "box_confidence",
        "transition_risk",
        "fb_breakout_age_bars",
        "fb_breakout_volume_ratio",
        "fb_reentry_volume_ratio",
        "fb_reentry_vs_breakout_ratio",
        "rev_engulf_short",
        "rev_engulf_long",
        "rev_body_reclaim_short",
        "rev_body_reclaim_long",
        "rev_prev_reject_short",
        "rev_prev_reject_long",
        "rev_composite_score_short",
        "rev_composite_score_long",
        "volume_rank_24",
        "volume_hour_norm",
        "volume_rel_20",
        "market_phase",
        "hour_utc",
        "session",
        "h4_range_usable",
        "h4_box_state",
        "method_agreement",
    ]
    candidate = execution.merge(merged[merge_cols], on="timestamp", how="left")
    candidate = candidate.merge(episodes, on="timestamp", how="left")
    for col in ["box_midline", "box_upper_edge", "box_lower_edge", "box_width", "box_width_pct"]:
        if col not in candidate.columns:
            if f"{col}_x" in candidate.columns:
                candidate[col] = candidate[f"{col}_x"]
            elif f"{col}_y" in candidate.columns:
                candidate[col] = candidate[f"{col}_y"]


    candidate["entry_time"] = candidate["timestamp"]
    candidate["base_confirms"] = candidate["event_base_confirms"]
    candidate["entry_side"] = np.where(candidate["exec_signal"] > 0, 1, -1)
    candidate["reentry_strength"] = np.where(
        candidate["entry_side"] > 0,
        (candidate["close"] - candidate["box_lower_edge"]) / candidate["box_width"].replace(0.0, np.nan),
        (candidate["box_upper_edge"] - candidate["close"]) / candidate["box_width"].replace(0.0, np.nan),
    )
    candidate["edge_distance"] = np.where(
        candidate["entry_side"] > 0,
        candidate["close"] - candidate["box_lower_edge"],
        candidate["box_upper_edge"] - candidate["close"],
    )
    candidate["edge_distance_atr"] = candidate["edge_distance"] / candidate["atr"].replace(0.0, np.nan)
    candidate["edge_distance_pct"] = candidate["edge_distance"] / candidate["close"].replace(0.0, np.nan)
    candidate["body_reclaim_reversal"] = np.where(candidate["entry_side"] > 0, candidate["rev_body_reclaim_long"], candidate["rev_body_reclaim_short"]).astype(bool)
    candidate["prev_reject_combo"] = np.where(candidate["entry_side"] > 0, candidate["rev_prev_reject_long"], candidate["rev_prev_reject_short"]).astype(bool)
    candidate["engulfing_reversal"] = np.where(candidate["entry_side"] > 0, candidate["rev_engulf_long"], candidate["rev_engulf_short"]).astype(bool)
    candidate["reversal_composite_score"] = np.where(candidate["entry_side"] > 0, candidate["rev_composite_score_long"], candidate["rev_composite_score_short"])
    candidate["confirm_richness"] = candidate["base_confirms"]

    trade_outcomes = control_audit[[
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_return",
        "hold_bars",
        "entry_precision_6h",
        "entry_precision_12h",
        "entry_precision_24h",
        "forward_mfe_6h",
        "forward_mae_6h",
        "overshoot_distance_atr",
        "reentry_distance_atr",
        "overshoot_distance_pct",
        "reentry_distance_pct",
        "entry_stop_distance",
        "engine",
    ]].copy()
    trade_outcomes["baseline_trade"] = True
    candidate = candidate.merge(trade_outcomes, on="entry_time", how="left")
    candidate["baseline_trade"] = candidate["baseline_trade"].fillna(False)
    candidate["fragile_trade"] = candidate["baseline_trade"] & candidate["exit_reason"].isin(["price_stop", "time_stop_early"])
    candidate["strong_trade"] = candidate["baseline_trade"] & candidate["exit_reason"].isin(["time_stop_max", "state_stop"]) & (candidate["net_return"] > 0)
    candidate["observation_class"] = np.select(
        [candidate["fragile_trade"], candidate["strong_trade"]],
        ["fragile", "strong"],
        default="other",
    )
    return candidate


def build_signature_tables(root: Path, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trade_df = candidate[candidate["baseline_trade"]].copy()
    focus = trade_df[trade_df["observation_class"].isin(["fragile", "strong"])].copy()

    features = [
        "entry_precision_6h",
        "entry_precision_12h",
        "entry_precision_24h",
        "forward_mfe_6h",
        "forward_mae_6h",
        "edge_distance_atr",
        "overshoot_distance_atr",
        "reentry_strength",
        "confirm_richness",
        "reversal_composite_score",
        "box_width_atr_mult",
        "anchor_shift_bars",
        "overshoot_bars",
        "fb_breakout_age_bars",
        "fb_reentry_vs_breakout_ratio",
        "volume_rank_24",
        "volume_hour_norm",
    ]

    split_rows = []
    for feature in features:
        for label, group in focus.groupby("observation_class", dropna=False):
            split_rows.append(
                {
                    "feature": feature,
                    "entry_class": label,
                    "trades": int(len(group)),
                    "mean": float(pd.to_numeric(group[feature], errors="coerce").mean()),
                    "median": float(pd.to_numeric(group[feature], errors="coerce").median()),
                }
            )
    feature_split = pd.DataFrame(split_rows)
    feature_split.to_csv(root / "fragile_entry_feature_split.csv", index=False)

    signature_rows = []
    fragile = focus[focus["observation_class"] == "fragile"]
    strong = focus[focus["observation_class"] == "strong"]
    for feature in features:
        f = pd.to_numeric(fragile[feature], errors="coerce")
        s = pd.to_numeric(strong[feature], errors="coerce")
        signature_rows.append(
            {
                "feature": feature,
                "fragile_mean": float(f.mean()),
                "strong_mean": float(s.mean()),
                "fragile_median": float(f.median()),
                "strong_median": float(s.median()),
                "effect_size_strong_minus_fragile": effect_size(s, f),
            }
        )
    signature = pd.DataFrame(signature_rows).sort_values("effect_size_strong_minus_fragile", key=lambda s: s.abs(), ascending=False)
    signature.to_csv(root / "fragile_vs_strong_entry_signature_table.csv", index=False)

    exit_reason_compare = trade_df.groupby("exit_reason", dropna=False).agg(
        trades=("entry_time", "size"),
        avg_net_return=("net_return", "mean"),
        median_entry_precision_6h=("entry_precision_6h", "median"),
        median_entry_precision_12h=("entry_precision_12h", "median"),
        median_entry_precision_24h=("entry_precision_24h", "median"),
        mean_forward_mfe_6h=("forward_mfe_6h", "mean"),
        mean_forward_mae_6h=("forward_mae_6h", "mean"),
        mean_reentry_strength=("reentry_strength", "mean"),
        mean_box_width_atr_mult=("box_width_atr_mult", "mean"),
        mean_anchor_shift_bars=("anchor_shift_bars", "mean"),
        mean_volume_rank_24=("volume_rank_24", "mean"),
    ).reset_index()
    exit_reason_compare.to_csv(root / "exit_reason_entry_signature_comparison.csv", index=False)
    return feature_split, signature, exit_reason_compare

def build_score_spec(candidate: pd.DataFrame) -> pd.DataFrame:
    focus = candidate[candidate["observation_class"].isin(["fragile", "strong"])].copy()
    strong = focus[focus["observation_class"] == "strong"]
    fragile = focus[focus["observation_class"] == "fragile"]
    rows = []
    for family_map, family_kind in [(CORE_FAMILY_FEATURES, "core"), (SECONDARY_FAMILY_FEATURES, "secondary")]:
        for family, features in family_map.items():
            for feature in features:
                if feature not in focus.columns:
                    continue
                if str(focus[feature].dtype) == "bool":
                    strong_mean = float(strong[feature].astype(float).mean())
                    fragile_mean = float(fragile[feature].astype(float).mean())
                    direction = "high" if strong_mean >= fragile_mean else "low"
                    threshold = 0.5
                    effect = strong_mean - fragile_mean
                else:
                    s = pd.to_numeric(strong[feature], errors="coerce")
                    f = pd.to_numeric(fragile[feature], errors="coerce")
                    strong_median = float(s.median())
                    fragile_median = float(f.median())
                    direction = "high" if strong_median >= fragile_median else "low"
                    threshold = (strong_median + fragile_median) / 2.0
                    effect = effect_size(s, f)
                rows.append(
                    {
                        "family": family,
                        "family_kind": family_kind,
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "effect_size": effect,
                        "weight": FAMILY_WEIGHTS[family],
                    }
                )
    spec = pd.DataFrame(rows)
    spec["abs_effect_size"] = pd.to_numeric(spec["effect_size"], errors="coerce").abs()
    spec = spec.sort_values(["family", "abs_effect_size"], ascending=[True, False])
    spec = spec.groupby("family", as_index=False).head(1).reset_index(drop=True)
    return spec.drop(columns=["abs_effect_size"])

def apply_entry_quality_score(candidate: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    out = candidate.copy()
    out["entry_quality_score"] = 0.0
    out["entry_quality_score_max"] = 0.0
    for row in spec.itertuples(index=False):
        feature = row.feature
        weight = float(row.weight)
        out[f"score_hit_{feature}"] = False
        if feature not in out.columns:
            continue
        if str(out[feature].dtype) == "bool":
            hit = out[feature] if row.direction == "high" else ~out[feature]
        else:
            series = pd.to_numeric(out[feature], errors="coerce")
            hit = series >= float(row.threshold) if row.direction == "high" else series <= float(row.threshold)
        out[f"score_hit_{feature}"] = hit.fillna(False)
        out[f"score_component_{row.family}"] = out.get(f"score_component_{row.family}", 0.0) + hit.fillna(False).astype(float) * weight
        out["entry_quality_score"] += hit.fillna(False).astype(float) * weight
        out["entry_quality_score_max"] += weight
    out["entry_quality_score_pct"] = out["entry_quality_score"] / out["entry_quality_score_max"].replace(0.0, np.nan)
    try:
        out["score_bucket"] = pd.qcut(out["entry_quality_score_pct"], 3, labels=["fragile", "middle", "strong"], duplicates="drop")
    except ValueError:
        out["score_bucket"] = pd.Series(["middle"] * len(out), index=out.index)
    return out


def build_score_outputs(root: Path, scored: pd.DataFrame, spec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist = scored.groupby(["score_bucket", "observation_class"], dropna=False).agg(
        rows=("entry_time", "size"),
        avg_score=("entry_quality_score_pct", "mean"),
    ).reset_index()
    dist.to_csv(root / "entry_quality_score_distribution.csv", index=False)

    trade_scored = scored[scored["baseline_trade"]].copy()
    bucket_perf = trade_scored.groupby("score_bucket", dropna=False).agg(
        trades=("entry_time", "size"),
        win_rate=("net_return", lambda x: float((x > 0).mean())),
        avg_return=("net_return", "mean"),
        total_return=("net_return", lambda x: float((1.0 + x).prod() - 1.0)),
        price_stop_rate=("exit_reason", lambda x: float((x == "price_stop").mean())),
        time_stop_max_rate=("exit_reason", lambda x: float((x == "time_stop_max").mean())),
        state_stop_rate=("exit_reason", lambda x: float((x == "state_stop").mean())),
    ).reset_index()
    bucket_perf.to_csv(root / "score_bucket_performance.csv", index=False)

    score_vs_exit = trade_scored.groupby("exit_reason", dropna=False).agg(
        trades=("entry_time", "size"),
        avg_score=("entry_quality_score_pct", "mean"),
        median_score=("entry_quality_score_pct", "median"),
        fragile_bucket_rate=("score_bucket", lambda x: float((x.astype(str) == "fragile").mean())),
        strong_bucket_rate=("score_bucket", lambda x: float((x.astype(str) == "strong").mean())),
    ).reset_index()
    score_vs_exit.to_csv(root / "score_vs_exit_reason.csv", index=False)

    spec.to_csv(root / "score_feature_spec.csv", index=False)
    return dist, bucket_perf, score_vs_exit


def filter_mask_for_variant(scored: pd.DataFrame, variant_name: str) -> pd.Series:
    low_score = scored["score_bucket"].astype(str) == "fragile"
    marginal_confirm = scored["base_confirms"].fillna(0) <= 2
    weak_reversal = ~scored["body_reclaim_reversal"].fillna(False)
    anchor_risk = scored["anchor_shift_bars"].fillna(0) >= scored["anchor_shift_bars"].fillna(0).median()
    low_context = scored["box_confidence"].fillna(0) <= scored["box_confidence"].median()

    if variant_name == "precision_rank_audit_only":
        return pd.Series(False, index=scored.index)
    if variant_name == "fragile_entry_filter_1":
        return low_score
    if variant_name == "fragile_entry_filter_2":
        return low_score & marginal_confirm & weak_reversal
    if variant_name == "best_effort_precision_variant":
        return low_score & ((marginal_confirm & weak_reversal) | (anchor_risk & low_context))
    raise ValueError(f"Unknown variant: {variant_name}")


def run_precision_variant(control_result: dict, scored: pd.DataFrame, variant_name: str) -> dict:
    config = deep_copy_config()
    merged = control_result["merged"].copy()
    execution_false_break = control_result["execution"]["false_break"].copy()
    execution_boundary = control_result["execution"]["boundary"].copy()

    filtered = execution_false_break.copy()
    score_view = scored.set_index("timestamp")
    mask = filter_mask_for_variant(scored, variant_name)
    filtered_ts = set(scored.loc[mask, "timestamp"].tolist())
    row_mask = filtered["timestamp"].isin(filtered_ts) & (filtered["exec_signal"] != 0)
    filtered.loc[row_mask, "exec_signal"] = 0
    filtered.loc[row_mask, "gate_pass"] = False
    filtered.loc[row_mask, "blocked_reason"] = filtered.loc[row_mask, "blocked_reason"].fillna("").astype(str).apply(
        lambda x: f"{x}|{variant_name}".strip("|")
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
        "filtered_mask": mask,
    }


def build_variant_outputs(root: Path, control_audit: pd.DataFrame, scored: pd.DataFrame, variant_results: dict[str, dict], annualization_hours: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    sensitivity_rows = []
    impact_rows = []
    retention_rows = []

    control_trades = parse_times(control_audit.copy(), ["entry_time", "exit_time"])
    control_total_return = float((1.0 + control_trades["net_return"]).prod() - 1.0) if not control_trades.empty else 0.0
    time_stop_max_contrib = float(control_trades.loc[control_trades["exit_reason"] == "time_stop_max", "net_return"].sum())

    for name, result in variant_results.items():
        trades = parse_times(result["backtest"]["trades_false_break"].copy(), ["entry_time", "exit_time"])
        summary = summarize_returns(trades, annualization_hours)
        rows.append(
            {
                "variant": name,
                **summary,
                "price_stop_count": int((trades["exit_reason"] == "price_stop").sum()) if not trades.empty else 0,
                "time_stop_early_count": int((trades["exit_reason"] == "time_stop_early").sum()) if not trades.empty else 0,
                "time_stop_max_count": int((trades["exit_reason"] == "time_stop_max").sum()) if not trades.empty else 0,
                "state_stop_count": int((trades["exit_reason"] == "state_stop").sum()) if not trades.empty else 0,
            }
        )
        ws = winner_sensitivity(trades, annualization_hours)
        ws.insert(0, "variant", name)
        sensitivity_rows.append(ws)

        if name != "control":
            filtered_ts = result["filtered_timestamps"]
            removed = control_trades[control_trades["entry_time"].isin(filtered_ts)].copy()
            retained = control_trades[~control_trades["entry_time"].isin(filtered_ts)].copy()
            impact_rows.append(
                {
                    "variant": name,
                    "removed_trades": int(len(removed)),
                    "removed_price_stop": int((removed["exit_reason"] == "price_stop").sum()) if not removed.empty else 0,
                    "removed_time_stop_early": int((removed["exit_reason"] == "time_stop_early").sum()) if not removed.empty else 0,
                    "removed_time_stop_max": int((removed["exit_reason"] == "time_stop_max").sum()) if not removed.empty else 0,
                    "removed_state_stop": int((removed["exit_reason"] == "state_stop").sum()) if not removed.empty else 0,
                    "removed_total_return_sum": float(removed["net_return"].sum()) if not removed.empty else 0.0,
                    "retained_time_stop_max_return_sum": float(retained.loc[retained["exit_reason"] == "time_stop_max", "net_return"].sum()) if not retained.empty else 0.0,
                    "delta_total_return_vs_control": summary["total_return"] - control_total_return,
                    "delta_time_stop_max_sum_vs_control": (float(retained.loc[retained["exit_reason"] == "time_stop_max", "net_return"].sum()) if not retained.empty else 0.0) - time_stop_max_contrib,
                }
            )
            for row in control_trades.itertuples(index=False):
                retention_rows.append(
                    {
                        "variant": name,
                        "entry_time": row.entry_time,
                        "exit_reason": row.exit_reason,
                        "net_return": row.net_return,
                        "retained": row.entry_time not in filtered_ts,
                    }
                )

    variant_summary = pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)
    variant_summary.to_csv(root / "variant_summary.csv", index=False)
    winner_df = pd.concat(sensitivity_rows, ignore_index=True)
    winner_df.to_csv(root / "winner_sensitivity_by_variant.csv", index=False)
    impact_df = pd.DataFrame(impact_rows)
    impact_df.to_csv(root / "fragile_filter_trade_impact.csv", index=False)
    retained_df = pd.DataFrame(retention_rows)
    retained_df = retained_df.merge(scored[["entry_time", "entry_quality_score_pct", "score_bucket"]], on="entry_time", how="left")
    retained_df.to_csv(root / "retained_vs_removed_baseline_trades.csv", index=False)
    return variant_summary, impact_df, retained_df, winner_df

def build_secondary_matrix(root: Path, scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "entry_time",
        "score_bucket",
        "entry_quality_score_pct",
        "observation_class",
        "exit_reason",
        "net_return",
        "box_width_pct",
        "box_width_atr_mult",
        "edge_distance_atr",
        "overshoot_distance_atr",
        "reentry_strength",
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
        "market_phase",
        "session",
    ]
    matrix = scored[[col for col in cols if col in scored.columns]].copy()
    matrix.to_csv(root / "secondary_observation_feature_matrix.csv", index=False)
    return matrix


def build_review_chart(bars: pd.DataFrame, trades: pd.DataFrame, label: str, html_path: Path, png_path: Path) -> None:
    if trades.empty:
        html_path.write_text("<html><body><p>No trades available.</p></body></html>", encoding="utf-8")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No trades available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        return

    sample = trades.sort_values("entry_time").head(40).copy()
    start = sample["entry_time"].min() - pd.Timedelta(hours=24)
    end = sample["exit_time"].max() + pd.Timedelta(hours=24)
    window = bars[(bars["timestamp"] >= start) & (bars["timestamp"] <= end)].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["box_upper_edge"], mode="lines", name="Box Upper", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["box_midline"], mode="lines", name="Box Mid", line=dict(width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=window["timestamp"], y=window["box_lower_edge"], mode="lines", name="Box Lower", line=dict(width=1)))

    entries = sample.copy()
    exits = sample.copy()
    stop_exits = exits[exits["exit_reason"] == "price_stop"]
    fig.add_trace(go.Scatter(x=entries["entry_time"], y=entries["entry_price"], mode="markers", name=f"{label} entry", marker=dict(size=9, symbol="diamond", color="#d95f02")))
    fig.add_trace(go.Scatter(x=exits["exit_time"], y=exits["exit_price"], mode="markers", name="Exit", marker=dict(size=8, symbol="x", color="#1b9e77")))
    if not stop_exits.empty:
        fig.add_trace(go.Scatter(x=stop_exits["exit_time"], y=stop_exits["exit_price"], mode="markers", name="Stop", marker=dict(size=10, symbol="x-thin", color="#e63946")))

    tradable = window[window["h4_range_usable"].fillna(False)]
    if not tradable.empty:
        fig.add_trace(go.Scatter(x=tradable["timestamp"], y=tradable["box_midline"], mode="markers", name="Tradable Window", marker=dict(size=3, color="rgba(70,130,180,0.35)")))

    fig.update_layout(title=f"{label} Entry Review", template="plotly_white", xaxis_rangeslider_visible=False)
    fig.write_html(str(html_path), include_plotlyjs="cdn")

    fig2, ax = plt.subplots(figsize=(14, 6))
    ax.plot(window["timestamp"], window["close"], linewidth=1.2, label="Close")
    ax.plot(window["timestamp"], window["box_upper_edge"], linewidth=0.8, label="Box Upper")
    ax.plot(window["timestamp"], window["box_midline"], linewidth=0.8, linestyle="--", label="Box Mid")
    ax.plot(window["timestamp"], window["box_lower_edge"], linewidth=0.8, label="Box Lower")
    ax.scatter(entries["entry_time"], entries["entry_price"], s=25, label=f"{label} entry")
    ax.scatter(exits["exit_time"], exits["exit_price"], s=25, marker="x", label="Exit")
    if not stop_exits.empty:
        ax.scatter(stop_exits["exit_time"], stop_exits["exit_price"], s=30, marker="x", color="#e63946", label="Stop")
    ax.set_title(f"{label} Entry Review")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig2.autofmt_xdate()
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=130)
    plt.close(fig2)


def write_reports(root: Path, signature: pd.DataFrame, score_spec: pd.DataFrame, score_perf: pd.DataFrame, variant_summary: pd.DataFrame) -> None:
    top_signature = signature.head(6)
    top_score = score_spec.sort_values("effect_size", key=lambda s: s.abs(), ascending=False)
    control = variant_summary.loc[variant_summary["variant"] == "control"].iloc[0]
    best_effort = variant_summary.loc[variant_summary["variant"] == "best_effort_precision_variant"].iloc[0]

    write_md(
        root / "ENTRY_EDGE_SIGNATURE_AUDIT.md",
        [
            "# Entry Edge Signature Audit",
            "",
            "## Observation",
            *[f"- {row.feature}: effect(strong-fragile)={row.effect_size_strong_minus_fragile:.3f}." for row in top_signature.itertuples(index=False)],
            "",
            "## Conclusion",
            "- A robust fragile-entry signature exists only if multiple entry-time features consistently separate fragile losers from strong winners.",
            "- Forward precision metrics are diagnostic only; ranking/filter logic uses entry-time features only.",
        ],
    )

    write_md(
        root / "ENTRY_QUALITY_SCORE_AUDIT.md",
        [
            "# Entry Quality Score Audit",
            "",
            "## Observation",
            *[f"- family={row.family}, feature={row.feature}, direction={row.direction}, effect={row.effect_size:.3f}, weight={row.weight:.2f}." for row in top_score.itertuples(index=False)],
            "",
            "## Conclusion",
            "- The score is an interpretable ranking layer on already-admitted entries, not a signal-expansion mechanism.",
            "- Secondary variables remain observation features unless they materially improve separation inside the ranked buckets.",
        ],
    )

    write_md(
        root / "ENTRY_EDGE_OPTIMIZATION_ROUND_SUMMARY.md",
        [
            "# Entry Edge Optimization Round Summary",
            "",
            "## Final Answers",
            f"1. Robust fragile-entry signature: {'yes' if top_signature['effect_size_strong_minus_fragile'].abs().max() > 0.5 else 'weak / not robust enough'}.",
            f"2. Strong vs fragile ranking separation: {'yes' if score_perf['price_stop_rate'].max() - score_perf['price_stop_rate'].min() > 0.10 else 'limited'}.",
            "3. Precision-aware suppression vs recovery expansion: precision-aware suppression is more justified in this round because it targets weak admitted entries instead of broadening admission.",
            "4. Secondary variables worth tracking: ATR geometry, anchor drift descriptors, richer reversal descriptors, and low-weight volume descriptors remain in the observation matrix.",
            "5. ATR geometry usefulness after isolation: tracked; review score spec and secondary matrix for whether ATR family contributes to separation.",
            "6. Anchor drift as risk label vs fix: tracked primarily as a risk/quality label unless a direct fix starts improving retained trade quality.",
            "7. Richer reversal descriptors ranking value: tracked as ranking features even though the prior broad admission variant failed.",
            "8. Volume conditional value: still weak unless it improves ranked-bucket separation conditionally.",
            f"9. Most justified next direction: {'best_effort_precision_variant' if best_effort['total_return'] >= control['total_return'] else 'continue with precision-aware ranking before any further admission changes'}.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated entry-edge optimization research.")
    parser.add_argument("--outdir", default="outputs/entry_edge_optimization_round")
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
    score_spec = build_score_spec(candidate)
    scored = apply_entry_quality_score(candidate, score_spec)
    scored.to_csv(root / "control_entry_candidates_scored.csv", index=False)

    _, signature, _ = build_signature_tables(root, scored)
    _, score_perf, _ = build_score_outputs(root, scored, score_spec)

    variant_results = {
        "control": {
            "backtest": control_result["backtest"],
            "filtered_timestamps": set(),
        }
    }
    for variant_name in ["precision_rank_audit_only", "fragile_entry_filter_1", "fragile_entry_filter_2", "best_effort_precision_variant"]:
        variant_results[variant_name] = run_precision_variant(control_result, scored, variant_name)

    annualization_hours = int(cfg["backtest"]["annualization_hours"])
    variant_summary, _, _, _ = build_variant_outputs(root, control_audit, scored, variant_results, annualization_hours)
    build_secondary_matrix(root, scored)

    fragile_review = scored[scored["fragile_trade"]].copy().rename(columns={"close": "entry_price"})
    strong_review = scored[scored["strong_trade"]].copy().rename(columns={"close": "entry_price"})
    fragile_review["exit_price"] = fragile_review["entry_price"] * (1.0 + fragile_review["entry_side"] * fragile_review["net_return"])
    strong_review["exit_price"] = strong_review["entry_price"] * (1.0 + strong_review["entry_side"] * strong_review["net_return"])
    build_review_chart(control_result["merged"], fragile_review, "Fragile", root / "fragile_entry_review.html", root / "fragile_entry_review.png")
    build_review_chart(control_result["merged"], strong_review, "Strong", root / "strong_entry_review.html", root / "strong_entry_review.png")

    write_reports(root, signature, score_spec, score_perf, variant_summary)
    print(f"Entry-edge optimization round complete. Outputs saved to: {root}")


if __name__ == "__main__":
    main()



