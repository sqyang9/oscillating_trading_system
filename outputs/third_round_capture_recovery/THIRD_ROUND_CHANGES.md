# Third Round Changes

This round was intentionally narrow.

## What was added

- A third-round capture audit runner:
  - `study_third_round_capture_recovery.py`
- Two narrow, execution-local recovery branches in `execution.py`
  - `capture_recovery_exec_conf_low_*`
  - `capture_recovery_confirm_near_miss_*`
- Focused execution tests for those branches

## What was not changed

- No regime redesign
- No boundary re-enable
- No warmup re-enable
- No bad-width filter removal
- No simple TP promotion
- No global volume require
- No broad confirm loosening
- No production-default change

## Default behavior

All third-round recovery logic remains:
- switchable
- local
- interpretable
- default-off

The production-candidate baseline on `master` was left unchanged throughout this round.

## Why the logic was not promoted

The recovery branches were tested against one narrow question: can they recover a small set of good missed trades inside already tradable windows without damaging the current winner structure?

Observed result:
- one branch had no effect
- one branch had no effect
- the permissive branch added trades but reduced total return and Sharpe
- added trades increased `price_stop` count and did not improve the winner structure

## Terminology cleanup

Third-round reports were tightened so they now state explicitly:
- event-layer attrition is largest by count, but not the main recoverable pool
- confirm / execution misses contain the narrow recoverable-looking subset
- overlapping reason hits are not mutually exclusive blocker totals
- the baseline remains the best default production-candidate configuration

## Final status

Third-round logic stays in the codebase for research comparison only. It is not promoted to default behavior.
