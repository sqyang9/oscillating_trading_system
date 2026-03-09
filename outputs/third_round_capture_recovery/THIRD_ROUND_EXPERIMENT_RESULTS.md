# Third Round Experiment Results

Primary result: no third-round recovery variant beat the current `master` baseline.

## Baseline

Authoritative baseline result:
- Trades: `117`
- Win rate: `37.61%`
- Avg return per trade: `0.6045%`
- Total return: `+94.71%`
- Sharpe: `5.048`
- Max drawdown: `-14.86%`

## Variant summary

No-change variants:
- `variant_capture_audit_only`
- `variant_near_miss_recovery_2`
- `variant_soft_confirm_strong_context`

Each of those reproduced baseline exactly.

Degrading variants:
- `variant_near_miss_recovery_1`
- `variant_best_effort_capture_recovery`

Their shared result:
- Trades: `127`
- Win rate: `34.65%`
- Avg return per trade: `0.4719%`
- Total return: `+74.48%`
- Sharpe: `4.003`
- Max drawdown: `-15.78%`

## Why the degrading variants failed

Relative to baseline:
- `price_stop` count rose from `49` to `55`
- `time_stop_max` count fell from `42` to `41`
- baseline trades retained: `107`
- baseline trades lost: `10`
- newly admitted trades: `20`
- recovered missed-good candidates: `7`
- newly introduced bad trades: `13`

Interpretation:
- The added trades were not clean recoveries of good missed opportunities.
- They mostly added low-quality entries faster than they recovered genuine misses.

## Capture interpretation

Inside already tradable context:
- raw candidates are numerous
- event-layer attrition is largest by count
- but that event-missed pool is not the main recoverable pool

Ex-post quality read:
- `event_missed` good-mid-before-stop rate: `24.14%`
- `confirm_missed` good-mid-before-stop rate: `42.64%`
- `execution_missed` good-mid-before-stop rate: `46.03%`

Conservative read:
- the recoverable-looking subset is narrow
- it appears later in the funnel
- the tested recovery rules did not isolate it well enough

## Trading-quality interpretation

- Entry logic is directionally reasonable.
- Profitable trades are often entered near useful local reversal zones.
- Losing trades are usually early / weak-confirmation failures, not obvious stop-placement accidents.
- The system's strength is not perfect point precision.
- The system's strength is filtering major trend damage and holding the right winners.
- Current weakness is not mainly stop placement; it is that a subset of entries is still slightly early or not selective enough.

## Terminology and semantics

- `tradable context window` refers to the same baseline-approved context definition used throughout the third-round audit outputs.
- Attrition counts should be read as overlapping reason hits where applicable, not mutually exclusive blocker totals.
- `newly introduced bad trades` is an ex-post new-trade audit concept, not a synonym for `price_stop` count.

## Final conclusion

Third round did not identify a recovery rule that improves on the current `master` baseline without damaging edge quality. The production-candidate baseline remains unchanged, and third-round recovery logic remains research-only and default-off.
