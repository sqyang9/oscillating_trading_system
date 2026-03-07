# Second-Round Experiment Summary

Primary output directory: `outputs/second_round_improvements/`

Attrition semantics note: These counts are overlapping reason hits and should not be interpreted as mutually exclusive blocker totals.

## Variant Ranking
- `variant_combo_best_effort`: trades=117, total_return=94.71%, max_drawdown=-14.86%, price_stop_count=49, time_stop_max_count=42, early_failure_count=0
- `variant_bad_width_bucket_filter`: trades=126, total_return=89.28%, max_drawdown=-14.86%, price_stop_count=55, time_stop_max_count=43, early_failure_count=0
- `variant_no_warmup`: trades=136, total_return=62.77%, max_drawdown=-16.88%, price_stop_count=62, time_stop_max_count=44, early_failure_count=0
- `variant_bad_width_bucket_repair`: trades=145, total_return=59.55%, max_drawdown=-16.19%, price_stop_count=65, time_stop_max_count=45, early_failure_count=3
- `baseline_current`: trades=145, total_return=58.23%, max_drawdown=-16.88%, price_stop_count=68, time_stop_max_count=45, early_failure_count=0
- `variant_early_failure_targeted`: trades=145, total_return=53.35%, max_drawdown=-16.09%, price_stop_count=65, time_stop_max_count=44, early_failure_count=5
- `variant_early_failure_global`: trades=145, total_return=52.10%, max_drawdown=-14.58%, price_stop_count=60, time_stop_max_count=42, early_failure_count=11
