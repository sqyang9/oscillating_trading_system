# Second-Round Audit: False-Break Baseline

Audit dataset: `btc_data/closed/btc_usdt_swap_5m_closed.csv` and `btc_data/closed/btc_usdt_swap_4h_closed.csv`

Primary output directory: `outputs/second_round_improvements/`

## 1. Scope, reproducibility, and baseline under review

This audit now uses a frozen audited baseline inside `study_second_round_improvements.py`.
The second-round study no longer inherits the repository's current `config.yaml` production-candidate defaults.
That was the only reproducibility fix applied here; no strategy logic was redesigned.

Frozen audited baseline:

- `false_break only`
- `boundary weight = 0`
- `false_break_regime_mode = down_only`
- `regime_lookback_hours = 720`
- `allow_warmup_trades = true`
- `bad_width_bucket_filter_enabled = false`
- `volume_confirmation = off`
- `expected_value_filter = off`
- `take_profit = off`
- `early_failure = off`

Baseline result re-confirmed after rerun:

- Trades: `145`
- Win rate: `31.72%`
- Avg return/trade: `0.3450%`
- Total return: `+58.23%`
- Sharpe: `3.243`
- Max drawdown: `-16.88%`
- Avg hold: `16.65 bars`

Judgment:

- The audited baseline is now reproducible from the current codebase even though `config.yaml` has moved to a newer production-candidate state.
- The core second-round strategic conclusions did not change on rerun.

## 2. Price-stop root-cause decomposition

Files:

- `outputs/second_round_improvements/baseline_price_stop_by_warmup.csv`
- `outputs/second_round_improvements/baseline_price_stop_by_width.csv`
- `outputs/second_round_improvements/baseline_price_stop_by_h4_width.csv`
- `outputs/second_round_improvements/baseline_price_stop_by_confirms.csv`
- `outputs/second_round_improvements/baseline_price_stop_side_direction.csv`
- `outputs/second_round_improvements/baseline_price_stop_vs_other_means.csv`
- `outputs/second_round_improvements/baseline_current/trade_audit_false_break.csv`

### 2.1 What price-stop losers look like

Baseline `price_stop` count: `68 / 145` (`46.90%`).

Most important mean differences versus non-`price_stop` trades:

- `reward_to_midline`: `0.4058%` vs `0.6133%`
- `reward_to_opposite_edge`: `1.1859%` vs `1.7271%`
- `reward_to_cost_midline`: `2.536x` vs `3.833x`
- `reward_to_cost_opposite`: `7.412x` vs `10.794x`
- `stop_dist_box_ratio`: `0.6318` vs `0.6094%`
- `reentry_strength`: `0.2556` vs `0.2562`
- `relevant_wick_ratio`: `0.3331` vs `0.2513`

Early path is the real separator:

- `mfe_1`: `0.3130%` vs `0.6614%`
- `mfe_2`: `0.3660%` vs `0.9687%`
- `mfe_3`: `0.4044%` vs `1.1661%`
- `mfe_6`: `0.4740%` vs `1.7615%`
- `mae_1`: `0.4310%` vs `0.2415%`
- `mae_2`: `0.6140%` vs `0.2857%`
- `mae_3`: `0.7823%` vs `0.3241%`
- `mae_6`: `0.9668%` vs `0.3781%`

Judgment:

- `price_stop` is driven primarily by weak early trade path, not by obviously too-tight stop placement.
- Stop placement is secondary. `stop_dist_box_ratio` is only slightly worse for stop trades.
- The cleanest causal signature remains low early MFE and high early MAE.

### 2.2 Which slices are worst

Width bucket slice:

- `<=0.5%`: `5` trades, `price_stop_rate = 80.0%`
- `0.5-1.0%`: `26` trades, `price_stop_rate = 61.5%`
- `1.0-1.5%`: `22` trades, `price_stop_rate = 59.1%`, total return `-5.62%`
- `1.5-2.5%`: `60` trades, `price_stop_rate = 45.0%`
- `2.5-4.0%`: `22` trades, `price_stop_rate = 27.3%`
- `4.0-8.0%`: `10` trades, `price_stop_rate = 20.0%`

4H width slice:

- `<=2.0%`: `15` trades, `price_stop_rate = 80.0%`
- `2.0-3.0%`: `23` trades, `price_stop_rate = 47.8%`
- `3.0-4.0%`: `35` trades, `price_stop_rate = 45.7%`
- `4.0-5.0%`: `33` trades, `price_stop_rate = 39.4%`
- `5.0-6.0%`: `39` trades, `price_stop_rate = 41.0%`

Confirm-count slice:

- `2 confirms`: still profitable overall, but price-stop remains common
- `3 confirms`: lower avg return than expected, not a free fix
- `4 confirms`: sample too small to treat as stable evidence

Direction slice:

- Both `buy_fb` and `sell_fb` can fail via `price_stop`
- In the surviving `down` regime, `sell_fb` aligns with major direction and is stronger on average, but bad `buy_fb` is not the whole story

### 2.3 Early failure mode

`price_stop` hold-bar distribution is front-loaded:

- Hold `1-5` bars: `41 / 68`
- Hold `<=10` bars: `50 / 68`

This still supports the same causal reading:

- The main bad trades usually reveal themselves early.
- The second-round early-failure variants were directionally sensible, but still clipped too much profitable structure in live sequential reruns.

## 3. Warmup contamination audit

Files:

- `outputs/second_round_improvements/baseline_warmup_summary.csv`
- `outputs/second_round_improvements/baseline_warmup_trades.csv`
- `outputs/second_round_improvements/variant_summary.csv`

Warmup definition in current code:

- `fb_regime = warmup` when `close.pct_change(regime_lookback_hours)` is unavailable
- In the original logic, `_false_break_regime_gate` still allowed `warmup`

Observed warmup sample:

- Warmup trades: `9`
- Entry window: `2019-12-21 21:00 UTC` to `2020-01-08 16:00 UTC`
- Warmup total return: `-2.79%`
- Non-warmup total return: `+62.77%`

Direct no-warmup variant:

- Trades: `136`
- Total return: `+62.77%`
- Sharpe: `3.558`
- Max drawdown: `-16.88%`
- `price_stop` count: `62`
- Warmup trades: `0`

Judgment:

- Warmup remains dirty sample contamination.
- This conclusion survives the reproducibility rerun unchanged.

## 4. The 1.0%-1.5% bad width bucket

Files:

- `outputs/second_round_improvements/baseline_width_bucket_focus.csv`
- `outputs/second_round_improvements/false_break_width_bucket_by_variant.csv`

Baseline width focus:

- `0.5-1.0%`: `26` trades, avg `+0.3720%`, total `+9.69%`, `price_stop_ratio = 61.5%`
- `1.0-1.5%`: `22` trades, avg `-0.2564%`, total `-5.62%`, `price_stop_ratio = 59.1%`
- `1.5-2.5%`: `60` trades, avg `+0.5457%`, total `+34.75%`, `price_stop_ratio = 45.0%`

Important detail:

- The `1.0-1.5%` bucket is still `100%` inside the intended `down` regime.
- It still shows reasonable reward-to-cost geometry, so this is not just a pure space-too-small issue.

Second-round conclusion on this bucket remains:

- Filtering it out works much better than trying to repair it with a mild scoped early-failure rule.
- The repair variant improved baseline only slightly.
- The direct filter produced the strongest improvement in this round.

## 5. Long-hold winner stability

Files:

- `outputs/second_round_improvements/baseline_time_stop_max_top_contributors.csv`
- `outputs/second_round_improvements/baseline_winner_sensitivity.csv`
- `outputs/second_round_improvements/winner_sensitivity_by_variant.csv`

Baseline winner sensitivity:

- Remove top `3` winners: total return `+14.32%`
- Remove top `5` winners: total return `+1.80%`
- Remove top `10` winners: total return `-19.48%`

Combo best effort sensitivity:

- Remove top `3` winners: total return `+40.68%`
- Remove top `5` winners: total return `+25.27%`
- Remove top `10` winners: total return `-0.91%`

Judgment:

- The edge still depends on winners.
- The combo remains materially less fragile than the old baseline.
- The rerun did not overturn the earlier conclusion that the system is not a one-trade illusion, but is still tail-dependent.

## 6. Tradable-context funnel and attrition semantics

Files:

- `outputs/second_round_improvements/tradable_funnel_by_variant.csv`
- `outputs/second_round_improvements/event_attrition_by_variant.csv`
- `outputs/second_round_improvements/execution_attrition_by_variant.csv`
- `outputs/second_round_improvements/blocked_reason_by_variant.csv`

`tradable context window` now has one shared definition across the audit funnel and the standalone chart.
It means all of these non-event gates are passing:

- `gate_fb_regime`
- `gate_fb_side_bias`
- `gate_allow_warmup`
- `gate_bad_width_bucket`
- `gate_h4_range_usable`
- `gate_h4_state`
- `gate_h4_width`
- `gate_h4_method`
- `gate_h4_transition`

Baseline funnel inside tradable context windows:

- Tradable context bars: `7189`
- Raw false-break candidates: `4189` (`58.27%` of tradable bars)
- Confirmed events: `441` (`10.53%` of candidates)
- Executed trades: `244` (`55.33%` of confirmed)

Consistent-definition effect on filtered variants:

- `variant_bad_width_bucket_filter` tradable context bars are now `5910`
- `variant_combo_best_effort` tradable context bars are now `5550`

Those counts are lower than the older report because `gate_bad_width_bucket` is now part of tradable context everywhere.
That is a semantic consistency fix, not a strategy change.

Attrition semantics note:

- `event_attrition_by_variant.csv` and `execution_attrition_by_variant.csv` now use `reason_hits`
- These counts are overlapping reason hits, not mutually exclusive blocker totals

Baseline overlapping reason hits:

- Event layer: `base_confirms_lt_min = 3805`, `missing_reentry = 3635`
- Execution layer inside tradable context: `event_confidence_low = 138`, `overlap_open_position = 62`, `side_cooldown = 39`, `opposite_overlap_blocked = 16`

Judgment:

- The main loss stage is still event formation / confirmation, not execution.
- The wording is now aligned with what the tables actually measure.

## 7. Direct answers to the required questions

1. Main root cause after second round:
   - unchanged in substance
   - still low-quality false-break entries that become `price_stop` losers
   - now narrowed to warmup contamination plus a bad `1.0-1.5%` width segment, with weak early path as the clearest signature
2. Should warmup be removed from the default baseline:
   - yes, high confidence
3. Main source of `price_stop`:
   - primarily bad entry quality / bad early path
   - not mainly stop placement itself
4. `1.0-1.5%` width bucket:
   - direct filter is better than mild repair in this sample
5. Early-failure mechanism that preserves long-hold winners:
   - no tested early-failure variant was good enough for default-on
6. Remove top winners:
   - baseline edge weakens sharply after removing top `5-10`
   - combo remains materially less fragile
7. Most worth keeping from second round:
   - `allow_warmup_trades = false`
   - `bad_width_bucket_filter_enabled = true`
8. What changed because of this patch:
   - reproducibility is now explicit
   - tradable-context wording is now unified
   - attrition semantics are now explicit
   - the strategic conclusions themselves did not change
