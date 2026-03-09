# Third Round Publish Note

## Status

This package is a clean close-out of the post-third-round state using the current GitHub `master` production-candidate baseline only.

## Baseline status

The production-candidate baseline remains unchanged:
- `false_break` only
- `boundary = 0`
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

## Third-round conclusion

Third round was a negative-but-useful result:
- no third-round recovery variant beat the current `master` baseline
- only a narrow ex-post recoverable subset exists among missed candidates
- tested recovery rules added more bad trades than good recovered trades
- third-round recovery logic remains research-only and default-off
- the system should remain low-frequency and selective

## Fourth-round pre-audit status

The fourth-round pre-audit is forensic and explanatory only:
- `risk_dropped_after_execution` is mostly a sequencing / policy bucket
- it is driven mainly by `position_overlap`, `risk_side_cooldown`, and `circuit_pause`
- this is primarily a lifecycle / naming / process-clarity issue, not a strategy redesign or hidden large optimization opportunity

## What this package includes

- finalized third-round reports
- fourth-round pre-audit note and CSV
- third-round study runner
- necessary code/tests already used in the round
- `outputs/third_round_capture_recovery/*`
- visual review outputs for the current best baseline

## What did not change

- no production defaults changed
- no new trading logic was promoted
- no new optimization round was opened
- no regime or strategy-family redesign was performed
