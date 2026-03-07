# Second-Round Changes

This file now documents the minimal reproducibility / consistency patch applied after the original second-round strategy work.
No strategy redesign, no parameter retuning, and no new exploratory optimization were introduced in this patch.

## 1. Baseline reproducibility fix

File:

- `study_second_round_improvements.py`

What changed:

- `baseline_current` no longer inherits the repository's current `config.yaml` defaults
- the script now defines an explicit frozen audited baseline override set
- all second-round variants are now built on top of that frozen audited research base before their own variant-specific overrides are applied

Why:

- `config.yaml` had already been moved to a newer production-candidate state
- without freezing the audited baseline, rerunning the second-round script would silently change the meaning of `baseline_current`

Result:

- the audited second-round baseline is reproducible again from the current codebase
- the rerun reproduced the same core baseline metrics and the same main variant ranking

## 2. Unified tradable-context definition

Files:

- `study_second_round_improvements.py`
- `make_tradeable_bs_equity_chart.py`
- `research_context.py` (new shared helper)

What changed:

- `tradable context window` now uses one shared non-event gate definition in both the audit funnel and the standalone chart
- the shared gate set is:
  - `gate_fb_regime`
  - `gate_fb_side_bias`
  - `gate_allow_warmup`
  - `gate_bad_width_bucket`
  - `gate_h4_range_usable`
  - `gate_h4_state`
  - `gate_h4_width`
  - `gate_h4_method`
  - `gate_h4_transition`

Why:

- the same phrase previously meant different things in the audit output and the visualization
- that created definition drift in `tradable_funnel_by_variant.csv` versus the chart

Result:

- the phrase now has one meaning across both files
- tradable-context counts for bad-width-filter variants changed because the context definition is now consistent everywhere
- this is a semantics fix, not a strategy-logic change

## 3. Attrition semantics clarification

Files:

- `study_second_round_improvements.py`
- `SECOND_ROUND_AUDIT.md`
- `SECOND_ROUND_EXPERIMENT_RESULTS.md`
- `outputs/second_round_improvements/event_attrition_by_variant.csv`
- `outputs/second_round_improvements/execution_attrition_by_variant.csv`

What changed:

- attrition tables now use `reason_hits`
- markdown outputs now explicitly state that these are overlapping reason hits, not mutually exclusive blocker totals

Why:

- multiple event-layer failure reasons can apply to the same candidate
- multiple execution blocked reasons can apply to the same event
- the old wording was too easy to misread as exclusive causal totals

Result:

- table labels and report wording now match the actual measurement semantics

## 4. Regenerated outputs and outcome

Regenerated:

- `outputs/second_round_improvements/*`
- `SECOND_ROUND_AUDIT.md`
- `SECOND_ROUND_EXPERIMENT_RESULTS.md`
- `outputs/second_round_improvements/REPRODUCIBILITY_NOTE.md`

Outcome:

- core second-round conclusions remained unchanged
- baseline reproducibility is restored
- tradable-context wording is now consistent
- attrition wording is now explicit

## 5. What this patch intentionally did not change

- no changes to event logic
- no changes to execution gate behavior other than freezing the audited research baseline inside the study runner
- no changes to risk logic
- no changes to parameter values used by the audited baseline
- no new optimization or search
