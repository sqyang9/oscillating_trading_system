# Reproducibility Note

This patch is limited to the second-round reproducibility / consistency fix.

It does three things:

- freezes the audited second-round baseline inside `study_second_round_improvements.py` so reruns no longer inherit current `config.yaml` defaults
- unifies `tradable context window` across the second-round audit funnel and the standalone tradable-context chart
- clarifies that attrition tables report overlapping `reason_hits`, not mutually exclusive blocker totals

Rerun result:

- Trades: `145`
- Total Return: `+58.23%`
- Sharpe: `3.243`
- MaxDD: `-16.88%`

Core second-round conclusions remain unchanged.
