# Trade Review Master Note

## What this package shows

The master review package is a human-inspection artifact for the current best baseline only. The aggregated HTML and PNG show:
- price over time
- highlighted tradable windows based on the shared baseline tradable-context definition
- long and short entry markers
- exit markers with stop-loss exits (`price_stop`) visually separated from other exit types
- box upper / lower edges and midline
- equity curve in the lower panel of the master review HTML

## How to read it

- shaded blue windows: approved tradable context intervals
- green `B` / upward markers: long entries
- orange `S` / downward markers: short entries
- red `x`: `price_stop` exits
- teal `x`: `time_stop_max` exits
- yellow `x`: `time_stop_early` exits
- purple `x`: `state_stop` exits

The segmented review files provide zoomed local inspection of the same baseline behavior when the full-history chart is too compressed.

## Interpretation

The charts are intended to answer a practical question: do entries and exits inside approved windows look sensible by eye?

Current visual answer:
- yes, the tradable windows look structurally plausible
- entries mostly occur inside those windows and often near local reversal zones
- many losing trades look slightly early or weak rather than obviously misplaced
- exits are coarse but still defensible
- visual inspection continues to support the current selective baseline rather than a denser default
