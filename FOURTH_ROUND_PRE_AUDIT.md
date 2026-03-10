# Fourth Round Pre-Audit

Scope: forensic review of `risk_dropped_after_execution` only. This is not a new strategy-optimization round.

## Question

Is `risk_dropped_after_execution` mostly:
- a real remaining strategy inefficiency,
- or a sequencing / overlap / accounting-policy bucket that mainly needs clarification?

## Exact lifecycle being audited

Baseline false-break backtest funnel:
- `raw_signal`: `3303`
- `after_execution_gate`: `180`
- `risk_entry_accepted`: `117`
- `risk_entry_blocked`: `63`
- `closed_trade`: `117`

So the audited bucket is:
- `63` execution-admitted entries that were then blocked at risk entry
- share of execution-admitted candidates: `35.0%` (`63 / 180`)

This is not an executed-trade failure bucket. It is a post-execution-admission, pre-risk-entry block bucket.

## Taxonomy

Observed blocked-entry reasons:
- `position_overlap`: `43`
- `risk_side_cooldown`: `13`
- `circuit_pause`: `7`

Conservative taxonomy:
- `sequencing_overlap_policy`: `43`
- `sequencing_cooldown_policy`: `13`
- `risk_policy_circuit_pause`: `7`

Interpretation:
- `56 / 63` (`88.9%`) are sequencing / policy effects from one-position, cooldown, or overlap handling.
- only `7 / 63` (`11.1%`) are circuit-pause policy blocks.

## Context clustering

Side:
- long: `32`
- short: `31`

Regime:
- all `63` occur in `down`

Width bucket:
- `1.5-2.5%`: `38`
- `0.5-1.0%`: `17`
- `2.5-4.0%`: `4`
- `<=0.5%`: `3`
- `4.0-8.0%`: `1`

Confirm count:
- `2`: `23`
- `3`: `34`
- `4`: `6`

These blocks do not point to a new broad bad-context cluster. They mostly sit in the same surviving baseline context that already generates the valid trades.

## Ex-post quality

Average ex-post quality by blocked-entry reason:

| Reason | good_mid_before_stop | good_opp_before_stop | mean MFE(2) | mean MAE(2) |
| --- | ---: | ---: | ---: | ---: |
| `position_overlap` | `30.23%` | `27.91%` | `0.41%` | `0.33%` |
| `risk_side_cooldown` | `38.46%` | `38.46%` | `0.50%` | `0.27%` |
| `circuit_pause` | `28.57%` | `14.29%` | `0.54%` | `0.32%` |

Read conservatively:
- blocked entries are not uniformly worthless
- but they are not obviously cleaner than executed trades either
- this looks like moderate opportunity cost from sequential policy, not a clear hidden alpha pool

## Overlap and sequencing read

For `position_overlap` blocks, the active trade at block time was:
- later a `winner`: `28`
- later an `other_loser`: `10`
- later a `price_stop_loser`: `5`

Interpretation:
- most overlap blocks happen while another position is already live
- a large share occur while that live position eventually proves worth holding
- this supports "expected sequential policy trade-off" more than "broken execution path"

For cooldown / circuit cases:
- prior exits are mostly `price_stop` or `time_stop_early`
- this is consistent with deliberate post-loss throttling, not a hidden event-engine defect

## Answers

1. Is `risk_dropped_after_execution` mostly a real strategy issue or mostly an accounting / sequencing issue?
   - Mostly a sequencing / policy issue.
   - The label reads more dramatically than the actual lifecycle.

2. Does it materially affect understanding of the current baseline?
   - No, not materially.
   - It explains some foregone entries, but it does not overturn the current baseline read.

3. Is there any small safe fix worth carrying forward?
   - The safest improvement is semantic clarity.
   - A future low-risk cleanup would be to rename this bucket in research outputs to something closer to `risk_entry_blocked_after_execution_gate`.
   - No strategy-logic change is justified from this audit alone.

4. Or should it simply be clarified and left alone?
   - Clarify it and leave the strategy logic alone for now.

## Bottom line

`risk_dropped_after_execution` is primarily a semantics / sequencing-policy bucket, not evidence of a clean remaining optimization. It should be interpreted as execution-admitted signals later blocked by overlap, cooldown, or circuit policy, not as trades that were validly opened and then unfairly discarded.
