# Oscillating Trading System

Event-first, causal BTC/USDT (OKX) oscillating range trading framework.

## First-Line Status
- The first completed low-frequency false-break research line is frozen and archived.
- The authoritative production baseline remains the frozen control profile documented in `memory.md`.
- The research-best refined preserve variant was not promoted to production.

## Reproduce First-Line Results
- Minimal reproducibility bundle: `reproducibility/first_line/`
- One-command run (PowerShell): `powershell -ExecutionPolicy Bypass -File reproducibility/first_line/run_repro.ps1`
- Direct Python run: `python reproducibility/first_line/run_repro.py`
- Expected reproduced headline result: control total return about `0.9471`, refined preserve total return about `1.0409`

## Freeze Docs
- Durable project memory: `memory.md`
- First-line archive summary: `FIRST_LINE_FINAL_ARCHIVE.md`
- Published freeze bundle: `docs/first_line_freeze/`

## Modules
- `local_box.py`: causal 1H/4H local box features
- `event_first.py`: false-break and box-init event families
- `execution.py`: execution gating and signal engines
- `risk.py`: causal risk exits and circuit controls
- `backtest.py`: sequential cost-aware backtest
- `study_event_first.py`: end-to-end study runner
- `study_false_break_ab.py`: A/B variant runner
- `research_false_break_regime.py`: regime deep research
- `make_trend_segment_bs_html.py`: standalone trend-segment + B/S interactive HTML

## Tests
- `python -m pytest -q`

## Default production profile
Use `config.production.yaml` (`down_only + A`, boundary backtest weight 0).
