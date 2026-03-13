# Preserve Final Refinement Summary

## Final Answers
1. Did preserve-high purity improve? Yes. Prior=0.350, refined=0.389.
2. Did hopeless contamination decline? Yes. Prior=0.200, refined=0.111.
3. Did fragile_high_upside remain preserved? Yes. Removed fragile_high_upside=0.
4. Is improvement stable across splits? Yes, modestly. Late-half total return moved from -0.0408 to 0.0043; refined beat control in 5 yearly slices.
5. Is result strong enough for promotion-readiness audit? No. The full-sample result improved from 0.9471 to 1.0409, but preserve-high purity only reached 0.389.

## Plain-Language Comparison
- Control total return was 0.9471. Prior preserve-first was 1.0191. Refined preserve variant was 1.0409.
- Sharpe moved from 5.048 to 5.547.
- Max drawdown moved from -0.1486 to -0.1260.
- Trade count moved from 117 to 110.
- price_stop count moved from 49 to 46.
- time_stop_max count moved from 42 to 43.
- The refined preserve variant removed 7 hopeless trades, 1 mixed trades, and 0 fragile high-upside trades.

## Recommendation
- Do not move to promotion-readiness yet.
- Stop this optimization line here unless there is a strong reason to spend one more cycle on preserve-side purity. The result improved, but the preserve side is still not clean enough to justify a promotion audit.
