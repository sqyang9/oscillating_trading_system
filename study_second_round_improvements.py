"""Run second-round audit and targeted improvement experiments on the closed BTC/USDT dataset."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from risk import RiskConfig, compute_atr
from study_event_first import run_study
from research_context import ATTRITION_OVERLAP_NOTE, compute_tradable_context_mask
from study_first_round_improvements import (
    CLOSED_INPUT_4H,
    CLOSED_INPUT_5M,
    WIDTH_BINS,
    WIDTH_LABELS,
    _blocked_reason_breakdown,
    _deep_merge,
    _equity_curve_from_returns,
    _exit_reason_ratio,
    _force_closed_inputs,
    _group_stats,
    _load_yaml,
    _save_yaml,
    _summary,
)

H4_WIDTH_BINS = [0.0, 0.02, 0.03, 0.04, 0.05, 0.06, np.inf]
H4_WIDTH_LABELS = ["<=2.0%", "2.0-3.0%", "3.0-4.0%", "4.0-5.0%", "5.0-6.0%", ">6.0%"]
EARLY_HORIZONS = [1, 2, 3, 4, 6]


FROZEN_AUDITED_SECOND_ROUND_BASE_OVERRIDES = {
    # Intentionally frozen so the audited second-round baseline remains reproducible
    # even if the repository's current config.yaml defaults change later.
    "event_false_break": {
        "overshoot_width_mult": 0.08,
        "reentry_window": 6,
        "min_confirmations": 2,
        "recapture_to_mid_frac": 0.40,
        "momentum_lookback": 3,
        "reversal_wick_ratio": 0.45,
        "volume_confirmation_enabled": False,
        "volume_confirmation_mode": "require",
        "volume_lookback": 20,
        "volume_min_periods": 10,
        "breakout_volume_max_ratio": 1.15,
        "reentry_volume_min_ratio": 0.95,
        "reentry_vs_breakout_min_ratio": 1.05,
    },
    "execution": {
        "width_pct_min": 0.001,
        "width_pct_max": 0.06,
        "method_agreement_min": 0.34,
        "transition_risk_max": 0.65,
        "confidence_min_false_break": 0.45,
        "confidence_min_boundary": 0.50,
        "side_cooldown_bars_false_break": 4,
        "side_cooldown_bars_boundary": 6,
        "allow_states": ["STABLE", "SQUEEZE"],
        "false_break_regime_mode": "down_only",
        "regime_lookback_hours": 720,
        "regime_down_low": -0.20,
        "regime_down_high": -0.05,
        "regime_up_low": 0.05,
        "regime_up_high": 0.20,
        "false_break_up_long_only": False,
        "allow_warmup_trades": True,
        "bad_width_bucket_filter_enabled": False,
        "bad_width_bucket_min_pct": 0.010,
        "bad_width_bucket_max_pct": 0.015,
        "expected_value_filter_enabled": False,
        "expected_value_mode": "to_midline",
        "min_reward_to_cost_ratio": 1.5,
        "expected_value_round_trip_cost_rate": None,
    },
    "risk": {
        "atr_window": 14,
        "atr_mult": 1.10,
        "box_stop_mult": 0.60,
        "min_stop_pct": 0.002,
        "max_hold_bars": 30,
        "early_progress_bars": 16,
        "early_progress_min_return": 0.0003,
        "state_stop_grace_bars": 2,
        "side_cooldown_bars": 4,
        "circuit_breaker_loss_streak": 3,
        "circuit_breaker_pause_bars": 16,
        "take_profit_enabled": False,
        "take_profit_mode": "midline",
        "early_failure_filter_enabled": False,
        "early_failure_bars": 6,
        "early_failure_min_progress": 0.003,
        "early_failure_max_adverse": 0.008,
        "early_failure_scope": "all",
        "early_failure_confirm_threshold": 2,
        "early_failure_width_min_pct": 0.010,
        "early_failure_width_max_pct": 0.015,
    },
    "backtest": {
        "fee_bps_per_leg": 5,
        "slippage_bps_per_leg": 2,
        "half_spread_bps_per_leg": 1,
        "combine_mode": "equal_weight",
        "engine_weights": {"false_break": 1.00, "boundary": 0.00},
        "annualization_hours": 8760,
    },
}

VARIANTS = [
    ("baseline_current", copy.deepcopy(FROZEN_AUDITED_SECOND_ROUND_BASE_OVERRIDES)),
    ("variant_no_warmup", {"execution": {"allow_warmup_trades": False}}),
    (
        "variant_early_failure_global",
        {
            "risk": {
                "early_failure_filter_enabled": True,
                "early_failure_bars": 2,
                "early_failure_min_progress": 0.003,
                "early_failure_max_adverse": 0.006,
                "early_failure_scope": "all",
            }
        },
    ),
    (
        "variant_early_failure_targeted",
        {
            "risk": {
                "early_failure_filter_enabled": True,
                "early_failure_bars": 2,
                "early_failure_min_progress": 0.003,
                "early_failure_max_adverse": 0.006,
                "early_failure_scope": "low_confirm_only",
                "early_failure_confirm_threshold": 2,
            }
        },
    ),
    (
        "variant_bad_width_bucket_filter",
        {
            "execution": {
                "bad_width_bucket_filter_enabled": True,
                "bad_width_bucket_min_pct": 0.010,
                "bad_width_bucket_max_pct": 0.015,
            }
        },
    ),
    (
        "variant_bad_width_bucket_repair",
        {
            "risk": {
                "early_failure_filter_enabled": True,
                "early_failure_bars": 2,
                "early_failure_min_progress": 0.003,
                "early_failure_max_adverse": 0.006,
                "early_failure_scope": "width_bucket_only",
                "early_failure_width_min_pct": 0.010,
                "early_failure_width_max_pct": 0.015,
            }
        },
    ),
    (
        "variant_combo_best_effort",
        {
            "execution": {
                "allow_warmup_trades": False,
                "bad_width_bucket_filter_enabled": True,
                "bad_width_bucket_min_pct": 0.010,
                "bad_width_bucket_max_pct": 0.015,
            }
        },
    ),
]


def _parse_time(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True)
    return out



def _reward_rate(side: pd.Series, target: pd.Series, entry: pd.Series) -> pd.Series:
    return side * (target / entry - 1.0)



def _subcurve_total(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    curve = _equity_curve_from_returns(df.sort_values("exit_time"), return_col="net_return")
    return float(curve["equity"].iloc[-1] - 1.0) if not curve.empty else 0.0



def _winner_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ranked = df.sort_values("net_return", ascending=False).reset_index(drop=True)
    for n in [0, 3, 5, 10]:
        trimmed = ranked.iloc[n:].copy() if n > 0 else ranked.copy()
        s = _summary(trimmed.sort_values("exit_time"), 8760, return_col="net_return")
        rows.append({"remove_top_winners": n, **s})
    return pd.DataFrame(rows)



def _tradable_funnel(events: pd.DataFrame, execution_df: pd.DataFrame, min_confirms: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [
        "timestamp",
        "fb_overshoot_up",
        "fb_overshoot_down",
        "fb_reenter_from_up",
        "fb_reenter_from_down",
        "fb_volume_blocked",
        "event_false_break_base_confirms",
        "event_false_break_signal",
    ]
    evt = events[[c for c in cols if c in events.columns]].copy()
    ex = execution_df.copy()
    merged = evt.merge(ex, on="timestamp", how="inner")

    tradable = compute_tradable_context_mask(merged)

    candidate = tradable & (merged["fb_overshoot_up"].fillna(False) | merged["fb_overshoot_down"].fillna(False))
    confirmed = tradable & (merged["raw_signal"].fillna(0).astype(int) != 0)
    executed = tradable & (merged["exec_signal"].fillna(0).astype(int) != 0)

    funnel = pd.DataFrame(
        [
            {"stage": "tradable_context_bars", "count": int(tradable.sum()), "rate_vs_prev": np.nan},
            {
                "stage": "raw_false_break_candidates_in_tradable",
                "count": int(candidate.sum()),
                "rate_vs_prev": float(candidate.sum() / tradable.sum()) if tradable.sum() else 0.0,
            },
            {
                "stage": "confirmed_events_in_tradable",
                "count": int(confirmed.sum()),
                "rate_vs_prev": float(confirmed.sum() / candidate.sum()) if candidate.sum() else 0.0,
            },
            {
                "stage": "executed_trades_in_tradable",
                "count": int(executed.sum()),
                "rate_vs_prev": float(executed.sum() / confirmed.sum()) if confirmed.sum() else 0.0,
            },
        ]
    )

    confirm_fail = merged[candidate & ~confirmed].copy()
    if confirm_fail.empty:
        event_attrition = pd.DataFrame(columns=["reason", "reason_hits"])
    else:
        confirm_fail["missing_reentry"] = ~(
            confirm_fail["fb_reenter_from_up"].fillna(False) | confirm_fail["fb_reenter_from_down"].fillna(False)
        )
        confirm_fail["base_confirms_lt_min"] = confirm_fail["event_false_break_base_confirms"].fillna(0).astype(int) < min_confirms
        event_attrition = pd.DataFrame(
            [
                {"reason": "missing_reentry", "reason_hits": int(confirm_fail["missing_reentry"].sum())},
                {"reason": "base_confirms_lt_min", "reason_hits": int(confirm_fail["base_confirms_lt_min"].sum())},
                {"reason": "volume_blocked", "reason_hits": int(confirm_fail["fb_volume_blocked"].fillna(False).sum())},
            ]
        ).sort_values("reason_hits", ascending=False)

    blocked = merged[confirmed & ~executed]["blocked_reason"].fillna("").astype(str)
    counts: dict[str, int] = {}
    for text in blocked:
        for reason in text.split("|"):
            reason = reason.strip()
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
    execution_attrition = pd.DataFrame(
        [{"reason": k, "reason_hits": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    )

    return funnel, event_attrition, execution_attrition



def _build_trade_audit(variant_dir: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    trades = _parse_time(pd.read_csv(variant_dir / "trades_false_break.csv"), ["entry_time", "exit_time"])
    if trades.empty:
        return trades

    execution = _parse_time(pd.read_csv(variant_dir / "execution_false_break.csv"), ["timestamp"])
    bars = _parse_time(pd.read_csv(variant_dir / "events_merged.csv"), ["timestamp"])

    risk_cfg = RiskConfig(**cfg["risk"])
    bt_cost = 2.0 * (
        (cfg["backtest"]["fee_bps_per_leg"] + cfg["backtest"]["slippage_bps_per_leg"] + cfg["backtest"]["half_spread_bps_per_leg"])
        / 10000.0
    )

    mom = bars["close"].pct_change(int(cfg["event_false_break"].get("momentum_lookback", 3)), fill_method=None)
    bars["momentum_turn_flag"] = np.where(
        bars.get("event_false_break_side", "none").eq("short"),
        (mom < 0.0) & (mom.shift(1) >= 0.0),
        np.where(
            bars.get("event_false_break_side", "none").eq("long"),
            (mom > 0.0) & (mom.shift(1) <= 0.0),
            False,
        ),
    )
    candle_range = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    upper_wick = (bars["high"] - bars[["open", "close"]].max(axis=1)) / candle_range
    lower_wick = (bars[["open", "close"]].min(axis=1) - bars["low"]) / candle_range
    bars["relevant_wick_ratio"] = np.where(
        bars.get("event_false_break_side", "none").eq("short"), upper_wick, np.where(bars.get("event_false_break_side", "none").eq("long"), lower_wick, np.nan)
    )
    bars["reentry_strength"] = np.where(
        bars.get("event_false_break_side", "none").eq("short"),
        (bars["box_upper_edge"] - bars["close"]) / bars["box_width"].replace(0.0, np.nan),
        np.where(
            bars.get("event_false_break_side", "none").eq("long"),
            (bars["close"] - bars["box_lower_edge"]) / bars["box_width"].replace(0.0, np.nan),
            np.nan,
        ),
    )
    bars["atr"] = compute_atr(bars, risk_cfg.atr_window)

    ex_keep = [
        "timestamp",
        "fb_regime",
        "fb_regime_ret",
        "box_width_pct",
        "event_base_confirms",
        "entry_warmup",
        "reward_to_cost_ratio",
        "expected_reward_rate",
        "expected_cost_rate",
        "blocked_reason",
    ]
    ex_keep = [c for c in ex_keep if c in execution.columns]
    execution = execution[ex_keep].rename(columns={"timestamp": "entry_time"})

    bar_keep = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "box_midline",
        "box_upper_edge",
        "box_lower_edge",
        "box_width",
        "box_width_pct",
        "h4_box_width_pct",
        "event_false_break_side",
        "event_false_break_base_confirms",
        "event_false_break_confidence",
        "reentry_strength",
        "relevant_wick_ratio",
        "momentum_turn_flag",
        "atr",
    ]
    bar_keep = [c for c in bar_keep if c in bars.columns]
    entry_bars = bars[bar_keep].rename(columns={"timestamp": "entry_time"})

    audit = trades.merge(execution, on="entry_time", how="left", suffixes=("", "_exec"))
    audit = audit.merge(entry_bars, on="entry_time", how="left", suffixes=("", "_bar"))

    if "event_base_confirms" not in audit.columns or audit["event_base_confirms"].isna().all():
        audit["event_base_confirms"] = audit.get("event_false_break_base_confirms", np.nan)
    audit["warmup"] = audit.get("entry_warmup", False).fillna(False).astype(bool) | audit.get("fb_regime", "").eq("warmup")
    audit["box_width_pct"] = audit["box_width_pct"].fillna(audit.get("box_width_pct_bar", np.nan))
    audit["box_width_bucket"] = pd.cut(audit["box_width_pct"], bins=WIDTH_BINS, labels=WIDTH_LABELS, include_lowest=True, right=True)
    if "h4_box_width_pct" in audit.columns:
        audit["h4_width_bucket"] = pd.cut(audit["h4_box_width_pct"], bins=H4_WIDTH_BINS, labels=H4_WIDTH_LABELS, include_lowest=True, right=True)
    else:
        audit["h4_width_bucket"] = pd.Series(pd.Categorical([np.nan] * len(audit), categories=H4_WIDTH_LABELS))

    audit["event_type"] = np.where(audit["side"] > 0, "buy_fb", "sell_fb")
    audit["major_direction"] = audit["fb_regime"].map({"down": "short", "deep_down": "short", "up": "long", "strong_up": "long"}).fillna("neutral")
    audit["with_major_direction"] = (
        ((audit["side"] > 0) & audit["major_direction"].eq("long"))
        | ((audit["side"] < 0) & audit["major_direction"].eq("short"))
    )
    audit["reward_to_midline"] = _reward_rate(audit["side"], audit["box_midline"], audit["entry_price"])
    opposite_target = np.where(audit["side"] > 0, audit["box_upper_edge"], audit["box_lower_edge"])
    audit["reward_to_opposite_edge"] = _reward_rate(audit["side"], pd.Series(opposite_target, index=audit.index), audit["entry_price"])
    audit["reward_to_cost_midline"] = audit["reward_to_midline"] / bt_cost
    audit["reward_to_cost_opposite"] = audit["reward_to_opposite_edge"] / bt_cost
    audit["entry_reward_to_cost_ratio"] = audit["reward_to_cost_midline"]

    stop_dist = pd.concat(
        [
            cfg["risk"]["min_stop_pct"] * audit["entry_price"],
            cfg["risk"]["atr_mult"] * audit["atr"],
            cfg["risk"]["box_stop_mult"] * audit["box_width"],
        ],
        axis=1,
    ).max(axis=1)
    audit["entry_stop_distance"] = stop_dist
    audit["stop_dist_box_ratio"] = stop_dist / audit["box_width"].replace(0.0, np.nan)

    idx_map = {ts: i for i, ts in enumerate(bars["timestamp"].tolist())}
    for horizon in EARLY_HORIZONS:
        audit[f"mfe_{horizon}"] = np.nan
        audit[f"mae_{horizon}"] = np.nan

    for row in audit.itertuples(index=True):
        entry_idx = idx_map.get(row.entry_time)
        exit_idx = idx_map.get(row.exit_time)
        if entry_idx is None or exit_idx is None:
            continue
        for horizon in EARLY_HORIZONS:
            end_idx = min(entry_idx + horizon, exit_idx)
            if end_idx <= entry_idx:
                continue
            window = bars.iloc[entry_idx + 1 : end_idx + 1]
            if window.empty:
                continue
            hi = pd.to_numeric(window["high"], errors="coerce").max()
            lo = pd.to_numeric(window["low"], errors="coerce").min()
            if row.side > 0:
                mfe = max((hi / row.entry_price - 1.0), 0.0) if np.isfinite(hi) else np.nan
                mae = max((1.0 - lo / row.entry_price), 0.0) if np.isfinite(lo) else np.nan
            else:
                mfe = max((1.0 - lo / row.entry_price), 0.0) if np.isfinite(lo) else np.nan
                mae = max((hi / row.entry_price - 1.0), 0.0) if np.isfinite(hi) else np.nan
            audit.at[row.Index, f"mfe_{horizon}"] = mfe
            audit.at[row.Index, f"mae_{horizon}"] = mae

    return audit



def _price_stop_baseline_tables(audit: pd.DataFrame, annualization_hours: int) -> dict[str, pd.DataFrame]:
    price_stop = audit[audit["exit_reason"] == "price_stop"].copy()
    others = audit[audit["exit_reason"] != "price_stop"].copy()

    compare_cols = [
        "reward_to_midline",
        "reward_to_opposite_edge",
        "reward_to_cost_midline",
        "reward_to_cost_opposite",
        "reentry_strength",
        "relevant_wick_ratio",
        "stop_dist_box_ratio",
    ] + [f"mfe_{h}" for h in EARLY_HORIZONS] + [f"mae_{h}" for h in EARLY_HORIZONS]
    compare = []
    for col in compare_cols:
        if col not in audit.columns:
            continue
        compare.append(
            {
                "feature": col,
                "price_stop_mean": float(price_stop[col].mean()) if not price_stop.empty else np.nan,
                "non_price_stop_mean": float(others[col].mean()) if not others.empty else np.nan,
            }
        )

    slices = {
        "baseline_price_stop_by_warmup.csv": pd.DataFrame(
            [
                {
                    "warmup": label,
                    "trades": int(len(group)),
                    "price_stop_count": int((group["exit_reason"] == "price_stop").sum()),
                    "price_stop_rate": float((group["exit_reason"] == "price_stop").mean()),
                    "avg_return": float(group["net_return"].mean()),
                    "total_return": _subcurve_total(group),
                }
                for label, group in audit.groupby("warmup", dropna=False)
            ]
        ),
        "baseline_price_stop_by_width.csv": pd.DataFrame(
            [
                {
                    "box_width_bucket": str(label),
                    "trades": int(len(group)),
                    "price_stop_count": int((group["exit_reason"] == "price_stop").sum()),
                    "price_stop_rate": float((group["exit_reason"] == "price_stop").mean()),
                    "avg_return": float(group["net_return"].mean()),
                    "total_return": _subcurve_total(group),
                }
                for label, group in audit.groupby("box_width_bucket", dropna=False)
            ]
        ).sort_values("box_width_bucket"),
        "baseline_price_stop_by_h4_width.csv": pd.DataFrame(
            [
                {
                    "h4_width_bucket": str(label),
                    "trades": int(len(group)),
                    "price_stop_count": int((group["exit_reason"] == "price_stop").sum()),
                    "price_stop_rate": float((group["exit_reason"] == "price_stop").mean()),
                    "avg_return": float(group["net_return"].mean()),
                    "total_return": _subcurve_total(group),
                }
                for label, group in audit.groupby("h4_width_bucket", dropna=False)
            ]
        ).sort_values("h4_width_bucket"),
        "baseline_price_stop_by_confirms.csv": pd.DataFrame(
            [
                {
                    "event_base_confirms": int(label) if pd.notna(label) else -1,
                    "trades": int(len(group)),
                    "price_stop_count": int((group["exit_reason"] == "price_stop").sum()),
                    "price_stop_rate": float((group["exit_reason"] == "price_stop").mean()),
                    "avg_return": float(group["net_return"].mean()),
                    "total_return": _subcurve_total(group),
                }
                for label, group in audit.groupby("event_base_confirms", dropna=False)
            ]
        ).sort_values("event_base_confirms"),
        "baseline_price_stop_vs_other_means.csv": pd.DataFrame(compare),
        "baseline_price_stop_side_direction.csv": pd.DataFrame(
            [
                {
                    "event_type": evt,
                    "with_major_direction": bool(aligned),
                    "trades": int(len(group)),
                    "price_stop_count": int((group["exit_reason"] == "price_stop").sum()),
                    "price_stop_rate": float((group["exit_reason"] == "price_stop").mean()),
                    "avg_return": float(group["net_return"].mean()),
                    "total_return": _subcurve_total(group),
                }
                for (evt, aligned), group in audit.groupby(["event_type", "with_major_direction"], dropna=False)
            ]
        ).sort_values(["event_type", "with_major_direction"]),
    }
    return slices



def _width_bucket_focus(audit: pd.DataFrame) -> pd.DataFrame:
    focus = audit[audit["box_width_bucket"].isin(["0.5-1.0%", "1.0-1.5%", "1.5-2.5%"])]
    rows = []
    for bucket, group in focus.groupby("box_width_bucket", dropna=False):
        rows.append(
            {
                "box_width_bucket": str(bucket),
                "trades": int(len(group)),
                "win_rate": float((group["net_return"] > 0).mean()),
                "avg_return": float(group["net_return"].mean()),
                "total_return": _subcurve_total(group),
                "avg_hold_bars": float(group["hold_bars"].mean()),
                "median_reward_to_cost_midline": float(group["reward_to_cost_midline"].median()),
                "price_stop_ratio": float((group["exit_reason"] == "price_stop").mean()),
                "time_stop_max_ratio": float((group["exit_reason"] == "time_stop_max").mean()),
                "time_stop_early_ratio": float((group["exit_reason"] == "time_stop_early").mean()),
            }
        )
    return pd.DataFrame(rows)



def _variant_summary_row(name: str, audit: pd.DataFrame, execution_df: pd.DataFrame, annualization_hours: int) -> Dict[str, Any]:
    s = _summary(audit.sort_values("exit_time"), annualization_hours, return_col="net_return")
    blocked = execution_df[(execution_df["raw_signal"] != 0) & (execution_df["exec_signal"] == 0)] if not execution_df.empty else pd.DataFrame()
    price_stop = audit[audit["exit_reason"] == "price_stop"]
    time_stop_max = audit[audit["exit_reason"] == "time_stop_max"]
    early_failure = audit[audit["exit_reason"] == "early_failure"] if "exit_reason" in audit.columns else pd.DataFrame()
    return {
        "variant": name,
        **s,
        "price_stop_count": int(len(price_stop)),
        "price_stop_total_return": _subcurve_total(price_stop),
        "time_stop_max_count": int(len(time_stop_max)),
        "time_stop_max_total_return": _subcurve_total(time_stop_max),
        "early_failure_count": int(len(early_failure)),
        "warmup_trade_count": int(audit["warmup"].sum()) if "warmup" in audit.columns else 0,
        "filtered_signal_count": int(len(blocked)),
    }



def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:
    parser = argparse.ArgumentParser(description="Run second-round audit and targeted improvement experiments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outdir", default="outputs/second_round_improvements")
    args = parser.parse_args()

    base_cfg = _force_closed_inputs(_load_yaml(args.config))
    frozen_research_cfg = copy.deepcopy(base_cfg)
    _deep_merge(frozen_research_cfg, copy.deepcopy(FROZEN_AUDITED_SECOND_ROUND_BASE_OVERRIDES))
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    variant_rows = []
    exit_rows = []
    blocked_rows = []
    width_rows = []
    regime_rows = []
    winner_rows = []
    funnel_rows = []
    event_attrition_rows = []
    execution_attrition_rows = []

    baseline_audit: pd.DataFrame | None = None
    baseline_dir: Path | None = None
    baseline_cfg: Dict[str, Any] | None = None

    annualization_hours = int(base_cfg["backtest"].get("annualization_hours", 8760))

    for name, overrides in VARIANTS:
        variant_cfg = copy.deepcopy(frozen_research_cfg)
        _deep_merge(variant_cfg, overrides)
        variant_dir = out_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _save_yaml(variant_cfg, variant_dir / "config_variant.yaml")
        run_study(config_path=str(variant_dir / "config_variant.yaml"), outdir_override=str(variant_dir))

        execution_df = _parse_time(pd.read_csv(variant_dir / "execution_false_break.csv"), ["timestamp"])
        events = _parse_time(pd.read_csv(variant_dir / "events_merged.csv"), ["timestamp"])
        audit = _build_trade_audit(variant_dir, variant_cfg)
        audit.to_csv(variant_dir / "trade_audit_false_break.csv", index=False)

        variant_rows.append(_variant_summary_row(name, audit, execution_df, annualization_hours))

        exr = _exit_reason_ratio(audit)
        if not exr.empty:
            exr.insert(0, "variant", name)
            exit_rows.append(exr)

        blocked = _blocked_reason_breakdown(execution_df)
        if not blocked.empty:
            blocked.insert(0, "variant", name)
            blocked_rows.append(blocked)

        wb = _group_stats(audit, "box_width_bucket", annualization_hours)
        if not wb.empty:
            wb.insert(0, "variant", name)
            width_rows.append(wb)

        rb = _group_stats(audit, "fb_regime", annualization_hours)
        if not rb.empty:
            rb.insert(0, "variant", name)
            regime_rows.append(rb)

        ws = _winner_sensitivity(audit)
        ws.insert(0, "variant", name)
        winner_rows.append(ws)

        min_confirms = int(variant_cfg["event_false_break"].get("min_confirmations", 2))
        funnel, event_attrition, execution_attrition = _tradable_funnel(events, execution_df, min_confirms)
        funnel.insert(0, "variant", name)
        funnel_rows.append(funnel)
        if not event_attrition.empty:
            event_attrition.insert(0, "variant", name)
            event_attrition_rows.append(event_attrition)
        if not execution_attrition.empty:
            execution_attrition.insert(0, "variant", name)
            execution_attrition_rows.append(execution_attrition)

        if name == "baseline_current":
            baseline_audit = audit.copy()
            baseline_dir = variant_dir
            baseline_cfg = copy.deepcopy(variant_cfg)

    variant_summary = pd.DataFrame(variant_rows).sort_values("total_return", ascending=False)
    variant_summary.to_csv(out_root / "variant_summary.csv", index=False)
    if exit_rows:
        pd.concat(exit_rows, ignore_index=True).to_csv(out_root / "exit_reason_by_variant.csv", index=False)
    if blocked_rows:
        pd.concat(blocked_rows, ignore_index=True).to_csv(out_root / "blocked_reason_by_variant.csv", index=False)
    if width_rows:
        pd.concat(width_rows, ignore_index=True).to_csv(out_root / "false_break_width_bucket_by_variant.csv", index=False)
    if regime_rows:
        pd.concat(regime_rows, ignore_index=True).to_csv(out_root / "false_break_regime_bucket_by_variant.csv", index=False)
    if winner_rows:
        pd.concat(winner_rows, ignore_index=True).to_csv(out_root / "winner_sensitivity_by_variant.csv", index=False)
    if funnel_rows:
        pd.concat(funnel_rows, ignore_index=True).to_csv(out_root / "tradable_funnel_by_variant.csv", index=False)
    if event_attrition_rows:
        pd.concat(event_attrition_rows, ignore_index=True).to_csv(out_root / "event_attrition_by_variant.csv", index=False)
    if execution_attrition_rows:
        pd.concat(execution_attrition_rows, ignore_index=True).to_csv(out_root / "execution_attrition_by_variant.csv", index=False)

    if baseline_audit is not None and baseline_dir is not None and baseline_cfg is not None:
        for filename, table in _price_stop_baseline_tables(baseline_audit, annualization_hours).items():
            table.to_csv(out_root / filename, index=False)

        warmup_summary = pd.DataFrame(
            [
                {
                    "metric": "warmup_trades",
                    "value": int(baseline_audit["warmup"].sum()),
                },
                {
                    "metric": "warmup_entry_start",
                    "value": str(baseline_audit.loc[baseline_audit["warmup"], "entry_time"].min()) if baseline_audit["warmup"].any() else "",
                },
                {
                    "metric": "warmup_entry_end",
                    "value": str(baseline_audit.loc[baseline_audit["warmup"], "entry_time"].max()) if baseline_audit["warmup"].any() else "",
                },
                {
                    "metric": "warmup_total_return",
                    "value": _subcurve_total(baseline_audit[baseline_audit["warmup"]]),
                },
                {
                    "metric": "non_warmup_total_return",
                    "value": _subcurve_total(baseline_audit[~baseline_audit["warmup"]]),
                },
            ]
        )
        warmup_summary.to_csv(out_root / "baseline_warmup_summary.csv", index=False)

        baseline_audit[baseline_audit["warmup"]].to_csv(out_root / "baseline_warmup_trades.csv", index=False)
        _width_bucket_focus(baseline_audit).to_csv(out_root / "baseline_width_bucket_focus.csv", index=False)
        baseline_audit[baseline_audit["exit_reason"] == "time_stop_max"].sort_values("net_return", ascending=False).head(20).to_csv(
            out_root / "baseline_time_stop_max_top_contributors.csv", index=False
        )
        _winner_sensitivity(baseline_audit).to_csv(out_root / "baseline_winner_sensitivity.csv", index=False)

    lines = [
        "# Second-Round Experiment Summary",
        "",
        "Primary output directory: `outputs/second_round_improvements/`",
        "",
        f"Attrition semantics note: {ATTRITION_OVERLAP_NOTE}",
        "",
        "## Variant Ranking",
    ]
    for row in variant_summary.itertuples(index=False):
        lines.append(
            f"- `{row.variant}`: trades={row.trades}, total_return={row.total_return:.2%}, max_drawdown={row.max_drawdown:.2%}, price_stop_count={row.price_stop_count}, time_stop_max_count={row.time_stop_max_count}, early_failure_count={row.early_failure_count}"
        )
    _write_markdown(out_root / "SECOND_ROUND_EXPERIMENT_SUMMARY.md", lines)


    reproducibility_lines = [
        "# Reproducibility Note",
        "",
        "The second-round study now freezes the audited research baseline before applying any variant-specific overrides.",
        "This decouples baseline_current and all other variants from the repository's current config.yaml production defaults.",
        "",
        "Tradable context windows are defined by one shared non-event gate set used by both the audit funnel and the standalone chart.",
        f"Attrition semantics: {ATTRITION_OVERLAP_NOTE}",
    ]
    _write_markdown(out_root / "REPRODUCIBILITY_NOTE.md", reproducibility_lines)



if __name__ == "__main__":
    main()

