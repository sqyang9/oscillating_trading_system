# Third Round Audit

Scope: current merged `master` only. This round audited capture loss inside already-approved tradable context windows and tested only narrow, execution-local recovery branches. It did not change the production-candidate baseline.

## Baseline kept fixed

Authoritative baseline used in this round:
- `false_break` only
- `boundary` weight `0`
- `false_break_regime_mode = down_only`
- `regime_lookback_hours = 720`
- `allow_warmup_trades = false`
- `bad_width_bucket_filter_enabled = true`
- `bad_width_bucket_min_pct = 0.010`
- `bad_width_bucket_max_pct = 0.015`
- `volume_confirmation = off`
- `expected_value_filter = off`
- `take_profit = off`
- `early_failure = off`

Baseline backtest result on current `master`:
- Trades: `117`
- Total return: `+94.71%`
- Sharpe: `5.048`
- Max drawdown: `-14.86%`

## Capture audit inside tradable context

Candidate-centered funnel inside already tradable context windows:
- Tradable-context bars: `5550`
- Raw false-break candidates: `3233`
- Event-layer survivors: `412`
- Confirm-layer survivors: `283`
- Execution-layer survivors: `157`
- Final traded candidate rows in the audit sample: `96`

Important terminology:
- The `96` figure above is the candidate-linked trade count inside the capture audit sample.
- It is not the same object as the full sequential backtest trade count of `117`.
- The backtest result remains the authoritative strategy performance figure.

## Main audit conclusions

1. Tradable windows are not sparse.
   - The system already sees a large number of raw false-break candidates inside approved context.
   - The baseline is selective inside those windows rather than failing to find context.

2. Event-layer attrition is largest by count, but it is not the main recoverable pool.
   - The largest raw loss remains the no-reentry / no-event pool.
   - That pool is mostly weak ex post and should not be treated as obviously recoverable alpha.

3. The narrower recoverable-looking subset sits later in the funnel.
   - `event_missed` candidates reached midline before equivalent stop only about `24.14%` of the time.
   - `confirm_missed` candidates did so about `42.64%` of the time.
   - `execution_missed` candidates did so about `46.03%` of the time.
   - This supports a narrow late-stage audit, not broad loosening.

4. The tested recovery rules did not recover that subset cleanly enough.
   - `variant_near_miss_recovery_2` and `variant_soft_confirm_strong_context` changed nothing.
   - `variant_near_miss_recovery_1` and `variant_best_effort_capture_recovery` added trades but degraded edge quality.

5. The baseline remains the best default production-candidate configuration.
   - No third-round recovery variant beat the current `master` baseline.
   - The system should remain low-frequency and selective by default.

## Variant read

Variants with no effect:
- `variant_capture_audit_only`
- `variant_near_miss_recovery_2`
- `variant_soft_confirm_strong_context`

These variants reproduced the baseline exactly:
- Trades: `117`
- Total return: `+94.71%`
- Sharpe: `5.048`
- Max drawdown: `-14.86%`

Variants that admitted additional trades:
- `variant_near_miss_recovery_1`
- `variant_best_effort_capture_recovery`

Their common result:
- Trades: `127`
- Total return: `+74.48%`
- Sharpe: `4.003`
- Max drawdown: `-15.78%`
- `price_stop` count: `55` vs baseline `49`
- `time_stop_max` count: `41` vs baseline `42`
- Baseline trades retained: `107`
- Baseline trades lost: `10`
- Newly added trades: `20`
- Recovered missed-good candidates: `7`
- Newly introduced bad trades: `13`

Judgment:
- Added trades mostly diluted quality.
- The tested recovery rules recovered fewer good trades than bad trades they introduced.

## Trading-point review

Practical read of the entries and exits:
- Entries are directionally and structurally reasonable; they are not random prints.
- Good trades are often entered near usable local reversal zones after real re-entry behavior.
- Losing trades are usually not "stopped out unfairly"; many are simply a bit early or too weak to mature.
- The edge does not come from perfect point precision.
- The edge comes from avoiding major trend damage and holding the right winners long enough.
- Exits are coarse but still defensible for this system shape, and keeping `time_stop_max` winners intact remains important.

## Winner-structure protection

The current baseline still depends on a meaningful winner tail, but it is stronger than the failed recovery variants:
- Baseline after removing top `10` winners: about `-0.91%`
- Recovery variant after removing top `10` winners: about `-12.42%`

Interpretation:
- Third-round recovery did not reduce tail dependence in a helpful way.
- It weakened the remaining distribution instead.

## Terminology guardrails

- `new price_stop count` means the change in the total number of `price_stop` exits versus baseline.
- `newly introduced bad trades` means newly admitted trades whose ex-post audit was poor; it is not identical to the `price_stop` delta.
- Attrition tables use overlapping reason hits, not mutually exclusive blocker totals.
- `tradable context window` refers only to the already-approved baseline context gate set; third round did not broaden that definition.

## Final conclusion

Third round did not identify a recovery rule that improves on the current `master` baseline without damaging edge quality. The production-candidate baseline remains unchanged, and third-round recovery logic remains research-only and default-off.
