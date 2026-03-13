# First Line Reproducibility Bundle

This bundle reproduces the completed first research line for the low-frequency selective false-break system.

## One-command run
PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File reproducibility/first_line/run_repro.ps1
```

If dependencies are already installed, the shorter command is:
```powershell
python reproducibility/first_line/run_repro.py
```

## What it does
- runs the frozen control baseline and the final preserved-line variants
- writes fresh outputs into `reproducibility/first_line/reproducibility_outputs/first_line`
- compares the reproduced `variant_summary.csv`, `preserve_purity_table.csv`, and `robustness_check.csv` against the published expected results
- prints `REPRODUCIBILITY_CHECK=PASS` on success

## Included files
- `frozen_control_config.yaml`: frozen authoritative control settings for this bundle
- `code/`: minimal strategy and first-line research modules needed by the final refinement runner
- `btc_data/closed/`: local closed 5m and 4h BTC/USDT snapshot used by this line
- `expected/`: expected summary tables for result verification
- `run_repro.py`: Python runner and result checker
- `run_repro.ps1`: one-command installer plus runner

## Expected headline result
- control total return about `0.9471`
- refined preserve total return about `1.0409`
- control max drawdown about `-0.1486`
- refined max drawdown about `-0.1260`

## Scope limits
- This bundle is for the first completed low-frequency line only.
- It does not publish the full `outputs/` tree.
- It does not start the separate new 1H research line.
