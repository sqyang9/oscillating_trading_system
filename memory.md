## 1. Project state
- Repository contains a frozen low-frequency selective false-break production baseline plus multiple completed research rounds for the first completed line.
- The first completed research line is closed. It delivered a real but modest research improvement through preserve-side refinement, but it is not promotion-ready.
- Future work should start as a separate 1H opportunity-expansion line and must not be mixed into this archived line.

## 2. Frozen production baseline
Authoritative control baseline:
- false_break only
- boundary = 0
- false_break_regime_mode = down_only
- regime_lookback_hours = 720
- allow_warmup_trades = false
- bad_width_bucket_filter_enabled = true
- bad_width_bucket_min_pct = 0.010
- bad_width_bucket_max_pct = 0.015
- volume_confirmation = off
- expected_value_filter = off
- take_profit = off
- early_failure = off

## 3. Best result from first completed research line
- Best result is a research-best refined preserve variant, not a promoted production baseline.
- Approximate headline metrics: total return 1.0409, Sharpe 5.547, max drawdown -0.1260, trade count 110.
- price_stop count fell versus control.
- fragile high-upside removals stayed at zero.
- preserve-high purity improved to about 0.389, but that remained too low for promotion.

## 4. Final conclusion of first line
- The first line succeeded as research.
- It found a real but modest edge improvement through fragile/preserve refinement.
- It is not promotion-ready.
- This line is frozen here and should not be further optimized by default.
- Do not continue preserve-side grinding unless this line is explicitly reopened later.

## 5. What was tested and what did NOT validate
- Broad recovery expansion did not validate.
- Volume as a hard gate did not validate.
- Richer reversal expansion did not validate in broad form.
- Anchor locking alone did not improve returns.
- ATR-normalized geometry improved explanatory power but did not yet prove direct trading superiority.
- Fragile/preserve/suppress work did produce real research improvement, but it stopped short of promotion.

## 6. Active constraints for future work
- Do not contaminate the frozen production baseline.
- Do not reopen broad recovery logic in the next thread.
- Do not confuse not-yet-proven with invalid.
- Keep ATR, anchor, reversal, and weak volume features alive as observation variables where relevant.
- Separate new research lines cleanly and keep their outputs isolated.

## 7. Next research line
- Next line: separate 1H opportunity-expansion and location-edge research.
- Focus: richer mean-reversion opportunity discovery rather than more preserve refinement.
- Example motifs: FVG, imbalance, displacement, retracement, range-edge confluence, reversal location structure.
- This is a new line, not a continuation of the preserve-refinement line.

## 8. File / artifact map
- Top-level memory: memory.md
- First-line archive note: FIRST_LINE_FINAL_ARCHIVE.md
- Published first-line docs and tables: docs/first_line_freeze/
- Final close-out script: study_preserve_final_refinement.py
- Supporting first-line studies: study_layer12_structural_research.py, study_entry_edge_optimization.py, study_fragile_label_refinement.py, study_preserve_suppress_ranking.py, study_preserve_first_robustness.py, study_preserve_refinement_round.py
- Final and supporting tests: test_layer12_structural_research.py, test_entry_edge_optimization.py, test_fragile_label_refinement.py, test_preserve_suppress_ranking.py, test_preserve_first_robustness.py, test_preserve_refinement_round.py, test_preserve_final_refinement.py

## 9. Research discipline reminders
- Isolate each research line.
- Freeze control arms explicitly.
- Prefer interpretable logic over black-box fitting.
- Always include intuitive return and drawdown presentation.
- Stop optimization lines when incremental gain becomes too small or impure.
