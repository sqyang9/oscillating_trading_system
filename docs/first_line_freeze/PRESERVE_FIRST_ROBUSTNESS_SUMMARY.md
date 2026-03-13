# Preserve First Robustness Summary

## Final Answers
1. Preserve-first robust across time or mostly sample-local: sample-local or unstable.
2. Improvement broad-based or concentrated: broad enough for follow-up, but not uniform.
3. Consistently removes the same kind of fragile hopeless trades: mostly yes.
4. Preserves fragile high-upside trades reliably across splits: yes in this sample audit.
5. Reduces structural fragility or only headline metrics: some structural improvement.
6. Preserve-high purity good enough for a future promotion round: not yet, but close enough for one narrow follow-up.
7. Recommended next step: one more preserve refinement pass, then a tightly scoped promotion attempt only if purity and split stability both hold.

## Observation
- full-sample control total_return=0.9471; preserve_first total_return=1.0191.
- early_half preserve_first total_return=1.0441; late_half preserve_first total_return=-0.0122.
- removed fragile_hopeless=5; removed fragile_high_upside=0.
- mean preserve-high oracle high-upside rate=0.319; mean preserve-high hopeless contamination=0.220; mean suppress-high hopeless rate=0.623.
