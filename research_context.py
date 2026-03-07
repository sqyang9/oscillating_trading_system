"""Shared research-time definitions for tradable context and attrition semantics."""

from __future__ import annotations

import pandas as pd

# Non-event context gates only. Event confirmation and execution overlap/cooldown are excluded.
TRADABLE_CONTEXT_GATE_COLS = [
    "gate_fb_regime",
    "gate_fb_side_bias",
    "gate_allow_warmup",
    "gate_bad_width_bucket",
    "gate_h4_range_usable",
    "gate_h4_state",
    "gate_h4_width",
    "gate_h4_method",
    "gate_h4_transition",
]

ATTRITION_OVERLAP_NOTE = (
    "These counts are overlapping reason hits and should not be interpreted as mutually exclusive blocker totals."
)


def compute_tradable_context_mask(df: pd.DataFrame) -> pd.Series:
    work = df.copy()
    for col in TRADABLE_CONTEXT_GATE_COLS:
        if col not in work.columns:
            work[col] = False
    return work[TRADABLE_CONTEXT_GATE_COLS].fillna(False).astype(bool).all(axis=1)
