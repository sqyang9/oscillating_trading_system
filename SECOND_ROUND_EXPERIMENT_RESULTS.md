# Second-Round Experiment Results

Primary output directory: `outputs/second_round_improvements/`

Reproducibility / consistency note:

- `baseline_current` is now frozen inside `study_second_round_improvements.py`
- the second-round study no longer inherits the repository's current `config.yaml` production-candidate defaults
- `tradable context window` now has one shared non-event gate definition across the audit funnel and the standalone chart
- `event_attrition_by_variant.csv` and `execution_attrition_by_variant.csv` report overlapping `reason_hits`, not mutually exclusive blocker totals

Core tables:

- `variant_summary.csv`
- `exit_reason_by_variant.csv`
- `blocked_reason_by_variant.csv`
- `false_break_width_bucket_by_variant.csv`
- `winner_sensitivity_by_variant.csv`
- `tradable_funnel_by_variant.csv`
- `event_attrition_by_variant.csv`
- `execution_attrition_by_variant.csv`

Core strategic result after rerun:

- unchanged
- the second-round baseline and main variant conclusions were reproduced from the current codebase after freezing the audited baseline

## 1. Variant comparison

| Variant | Trades | Win rate | Avg return | Total return | Sharpe | Max drawdown | Avg hold | Price stops | Time-stop-max | Warmup trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_current` | 145 | 31.72% | 0.3450% | 58.23% | 3.243 | -16.88% | 16.65 | 68 | 45 | 9 |
| `variant_no_warmup` | 136 | 33.09% | 0.3879% | 62.77% | 3.558 | -16.88% | 16.85 | 62 | 44 | 0 |
| `variant_early_failure_global` | 145 | 29.66% | 0.3171% | 52.10% | 3.069 | -14.58% | 16.06 | 60 | 42 | 9 |
| `variant_early_failure_targeted` | 145 | 31.03% | 0.3230% | 53.35% | 3.086 | -16.09% | 16.34 | 65 | 44 | 9 |
| `variant_bad_width_bucket_filter` | 126 | 35.71% | 0.5397% | 89.28% | 4.645 | -14.86% | 17.40 | 55 | 43 | 9 |
| `variant_bad_width_bucket_repair` | 145 | 31.72% | 0.3507% | 59.55% | 3.303 | -16.19% | 16.63 | 65 | 45 | 9 |
| `variant_combo_best_effort` | 117 | 37.61% | 0.6045% | 94.71% | 5.048 | -14.86% | 17.70 | 49 | 42 | 0 |

## 2. What improved and why

### 2.1 `variant_no_warmup`

Effect:

- removes the `9` warmup trades
- total return: `58.23% -> 62.77%`
- Sharpe: `3.243 -> 3.558`
- `price_stop`: `68 -> 62`

Interpretation:

- warmup was not part of the live edge
- removing it remains a clean sample-quality improvement

### 2.2 `variant_bad_width_bucket_filter`

Effect:

- trades: `145 -> 126`
- total return: `58.23% -> 89.28%`
- Sharpe: `3.243 -> 4.645`
- Max drawdown: `-16.88% -> -14.86%`
- `price_stop`: `68 -> 55`
- `time_stop_max`: `45 -> 43`

Interpretation:

- this remains the strongest single second-round result
- it removes a low-quality segment while keeping most of the profitable long-hold tail intact

### 2.3 `variant_bad_width_bucket_repair`

Effect:

- total return: `58.23% -> 59.55%`
- Max drawdown: `-16.88% -> -16.19%`
- `price_stop`: `68 -> 65`
- `time_stop_max`: unchanged at `45`

Interpretation:

- mild targeted repair helped a little
- it was still much weaker than directly filtering the bucket

### 2.4 Early-failure variants

Global early-failure:

- `price_stop`: `68 -> 60`
- Max drawdown: `-16.88% -> -14.58%`
- total return fell to `52.10%`
- `time_stop_max`: `45 -> 42`

Targeted early-failure:

- `price_stop`: `68 -> 65`
- total return fell to `53.35%`
- `time_stop_max`: `45 -> 44`

Interpretation:

- early-failure does detect some bad trades
- but it still clips enough profitable structure to fail the default-on bar
- it remains optional, not preferred

### 2.5 `variant_combo_best_effort`

Combo used:

- `allow_warmup_trades = false`
- `bad_width_bucket_filter_enabled = true`

Effect:

- trades: `117`
- win rate: `37.61%`
- total return: `94.71%`
- Sharpe: `5.048`
- Max drawdown: `-14.86%`
- `price_stop`: `49`
- `time_stop_max`: `42`
- warmup trades: `0`

Interpretation:

- this remains the best in-sample second-round result
- it reduces bad trades materially without destroying the profitable tail
- the rerun did not change that conclusion

## 3. Funnel interpretation

`tradable context window` now means the same thing in both the audit and the chart.
It requires these non-event gates to pass:

- `gate_fb_regime`
- `gate_fb_side_bias`
- `gate_allow_warmup`
- `gate_bad_width_bucket`
- `gate_h4_range_usable`
- `gate_h4_state`
- `gate_h4_width`
- `gate_h4_method`
- `gate_h4_transition`

Baseline funnel:

- tradable context bars: `7189`
- raw candidates in tradable context: `4189`
- confirmed events in tradable context: `441`
- executed trades in tradable context: `244`

Consistent-definition impact on filtered variants:

- `variant_bad_width_bucket_filter` tradable context bars are now `5910`
- `variant_combo_best_effort` tradable context bars are now `5550`

Those counts are lower than in the older report because `gate_bad_width_bucket` now belongs to the context definition everywhere.
That is a semantic fix, not a trading-logic change.

Attrition semantics:

- `event_attrition_by_variant.csv` and `execution_attrition_by_variant.csv` now use `reason_hits`
- those counts are overlapping reason hits, not exclusive blocker totals

Baseline overlapping reason hits:

- event: `base_confirms_lt_min = 3805`, `missing_reentry = 3635`
- execution within tradable context: `event_confidence_low = 138`, `overlap_open_position = 62`, `side_cooldown = 39`, `opposite_overlap_blocked = 16`

Interpretation:

- the main attrition still happens at event formation / confirmation, not execution
- the tables and wording are now aligned with that measurement semantics

## 4. Width-bucket before/after

Baseline:

- `1.0-1.5%`: `22` trades, total return `-5.62%`

Bad-width filter:

- `1.0-1.5%`: `0` trades
- overall total return improved to `89.28%`

Combo best effort:

- `1.0-1.5%`: `0` trades
- overall total return improved to `94.71%`

This remains the clearest second-round repair story.

## 5. Winner sensitivity

Baseline:

- remove top `3`: `+14.32%`
- remove top `5`: `+1.80%`
- remove top `10`: `-19.48%`

Bad-width filter:

- remove top `3`: `+36.76%`
- remove top `5`: `+21.78%`
- remove top `10`: `-3.68%`

Combo best effort:

- remove top `3`: `+40.68%`
- remove top `5`: `+25.27%`
- remove top `10`: `-0.91%`

Interpretation:

- the edge still depends on winners
- the combo remains materially less fragile than the old baseline

## 6. Preferred repository stance after the fix

This patch did not reopen optimization.
It only made the existing second-round artifacts reproducible and semantically consistent.

Therefore the strategic conclusion remains:

- `baseline_current` is reproducible again as the audited reference
- `variant_combo_best_effort` remains the best in-sample second-round candidate

## 7. Validation checklist

1. `baseline_current` is now reproducible from a frozen audited override set:
   - confirmed
2. `baseline_current` is no longer coupled to the current `config.yaml` state:
   - confirmed
3. `tradable context window` has one consistent definition across audit and chart:
   - confirmed
4. attrition counts are clearly presented as overlapping reason hits:
   - confirmed
5. main second-round strategic conclusion changed or not:
   - unchanged
