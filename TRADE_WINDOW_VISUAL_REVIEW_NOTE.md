# Trade Window Visual Review Note

## Scope

Visual review was performed on the current best baseline only, using:
- full-history overview
- segmented local zoom charts
- highlighted tradable context windows
- entry and exit markers
- local box upper / lower edges and midline

## What the charts show

1. Tradable windows look structurally sensible.
   - Highlighted windows cluster around local range / reversal conditions inside the approved `down_only` regime backbone.
   - They do not appear sprayed across obviously hostile trend-chase zones.

2. Entries generally align with those windows.
   - Buy and sell points are usually placed near box edges or near local reversal attempts inside the approved windows.
   - They are not visually random and they are not mostly appearing outside the highlighted context.

3. The system does not look visually too dense.
   - `62` review segments were generated.
   - `35` segments contain trades and `27` contain no trades.
   - Total trade count across the segment index sums exactly to `117`, matching the full backtest baseline.
   - This supports the interpretation that the strategy is selective rather than under-filtered.

4. Where trades fail, they often look slightly early rather than obviously mislocated.
   - In dense segments such as `segment_038`, entries still occur inside valid windows, but the price path is noisier and several trades fail before the move matures.
   - This visually matches the earlier analytical conclusion that many losers are weak / early entries rather than stop-placement accidents.

5. Winner handling still looks defensible.
   - In cleaner segments such as `segment_045`, some entries sit near useful local reversal zones and are allowed to extend into `time_stop_max` winners.
   - Visually, the strategy still appears to rely on holding the better trades rather than forcing high point precision.

## Practical read

- Entries look directionally reasonable.
- Exits look coarse but defensible.
- The system does not look obviously too sparse.
- It also does not justify becoming denser by default.
- Visual inspection supports the conclusion to keep the current baseline selective.

## Caution

The segment charts are for human trading-behavior review, not a replacement for the quantitative reports. The key conclusion remains unchanged: the current master baseline stays the best production-candidate default, and third-round recovery logic remains default-off.
