# Second-Round Master Mismatch Audit

## Scope

This note documents the mismatch discovered after the second-round reproducibility patch was merged.
Third-round work is paused until this mismatch is reconciled on `master`.

## Core mismatch

The repository currently contains committed second-round reports and committed second-round outputs, but `master` does not contain the full strategy/research code needed to reproduce or even execute the same second-round baseline cleanly.

That means the repository was in an internally inconsistent state:

- reports and outputs reflected the real second-round production-candidate baseline
- committed strategy code on `master` did not fully implement that baseline
- committed research code on `master` already assumed fields and helpers that the committed strategy code did not produce

## Exact gaps identified

### 1. Event layer mismatch

`study_second_round_improvements.py` expects event-layer fields such as:

- `event_false_break_base_confirms`
- `fb_volume_blocked`

But committed `master` event code did not generate them.

### 2. Execution layer mismatch

Second-round baseline and reports rely on execution-layer support for:

- `allow_warmup_trades`
- `bad_width_bucket_filter_enabled`
- `bad_width_bucket_min_pct`
- `bad_width_bucket_max_pct`
- execution metadata used by the second-round audit, including `box_width_pct`, `event_base_confirms`, and `entry_warmup`

Committed `master` execution code did not include that support.

### 3. Risk layer mismatch

Committed `master` second-round research code expects risk/trade metadata used by the audit and variant logic, including:

- `entry_base_confirms`
- `entry_box_width_pct`
- `entry_warmup`
- early-failure support used by second-round repair variants

Committed `master` risk code did not include those capabilities.

### 4. Runner/config mismatch

Committed `master` config did not represent the actual second-round production-candidate baseline.
It was missing the second-round execution/risk/event options required by the baseline used in the audited outputs.

Committed `master` runner/research wiring was also incomplete for the code paths already assumed by the second-round study stack.

### 5. Missing research dependency

Committed `master` already included `study_second_round_improvements.py`, but it imported `study_first_round_improvements.py`, which was not committed on `master`.
So the second-round research entrypoint on `master` was not self-contained.

## Minimal reconciliation required

To make `master` internally consistent, the minimum patch is:

- `event_first.py`
- `execution.py`
- `risk.py`
- `study_event_first.py`
- `study_first_round_improvements.py`
- `config.yaml`
- `config.production.yaml`
- `test_event_first.py`
- `test_execution.py`
- `test_risk.py`
- `pytest.ini`

This patch does not introduce third-round ideas.
It only publishes the actual second-round baseline code that the committed reports/outputs already rely on.

## Authoritative post-second-round master baseline

After reconciliation, the intended authoritative baseline on `master` is:

- `false_break only`
- `boundary weight = 0`
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

## Implication for next work

Third-round capture-rate work should start only after this reconciliation patch lands on `master`.
At that point, `master` will finally match the real post-second-round baseline represented by the committed reports and outputs.
