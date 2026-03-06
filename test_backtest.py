import math

import pandas as pd

from backtest import BacktestConfig, build_funnel, run_backtest


def _trades(engine: str, rows: list[tuple[str, str, int, float, float, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "engine": engine,
                "entry_time": e,
                "exit_time": x,
                "side": s,
                "entry_price": ep,
                "exit_price": xp,
                "hold_bars": h,
                "gross_return": g,
                "exit_reason": r,
                "entry_confidence": 0.8,
            }
            for e, x, s, ep, xp, h, g, r in rows
        ]
    )


def test_backtest_costs_and_combined_compounding():
    tf = _trades(
        "false_break",
        [
            ("2024-01-01 00:00:00+00:00", "2024-01-01 03:00:00+00:00", 1, 100, 102, 3, 0.02, "time_stop_max"),
            ("2024-01-01 04:00:00+00:00", "2024-01-01 06:00:00+00:00", -1, 103, 101, 2, 0.0194174757, "state_stop"),
        ],
    )
    tb = _trades(
        "boundary",
        [
            ("2024-01-01 01:00:00+00:00", "2024-01-01 05:00:00+00:00", 1, 100, 99, 4, -0.01, "price_stop"),
        ],
    )

    cfg = BacktestConfig(fee_bps_per_leg=5, slippage_bps_per_leg=2, half_spread_bps_per_leg=1)
    out = run_backtest(tf, tb, cfg)

    f = out["trades_false_break"]
    assert not f.empty
    expected_net = 0.02 - 0.0016
    assert math.isclose(float(f.loc[0, "net_return"]), expected_net, rel_tol=1e-9)

    combined = out["summary"]["combined"]
    assert combined["trades"] == 3
    assert "combined" in out["summary"]


def test_build_funnel_counts_stages():
    ex_fb = pd.DataFrame({"raw_signal": [1, 1, 0], "exec_signal": [1, 0, 0]})
    ex_bd = pd.DataFrame({"raw_signal": [0, -1], "exec_signal": [0, -1]})

    lg_fb = pd.DataFrame(
        {
            "event": ["entry", "entry"],
            "status": ["accepted", "blocked"],
        }
    )
    lg_bd = pd.DataFrame(
        {
            "event": ["entry"],
            "status": ["accepted"],
        }
    )

    tr_fb = pd.DataFrame({"x": [1]})
    tr_bd = pd.DataFrame({"x": [1, 2]})

    funnel = build_funnel(ex_fb, ex_bd, lg_fb, lg_bd, tr_fb, tr_bd)
    assert int(funnel[(funnel["engine"] == "false_break") & (funnel["stage"] == "raw_signal")]["count"].iloc[0]) == 2
    assert int(funnel[(funnel["engine"] == "boundary") & (funnel["stage"] == "closed_trade")]["count"].iloc[0]) == 2