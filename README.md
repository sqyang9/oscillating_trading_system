# Oscillating Trading System

Event-first, causal BTC/USDT (OKX) oscillating range trading framework.

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
