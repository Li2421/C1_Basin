# Four-Way Cyclic Intersection

Canonical agent order is `A, B, C, D = North→South, East→West, South→North,
West→East`.  The conflict region is an open square: all four straight routes
cross it and no corridor, signal, priority bit, or right-of-way rule exists.

The outer workspace boundary is deliberately buffered outside the broad start
distribution; it is a safety boundary, not an additional approach bottleneck.

`observation()` returns `[4,18]`: own position, prior applied velocity,
own-relative goal, then the other three position/velocity-relative blocks in
canonical global order.  Boundary geometry is recoverable from absolute
position and the fixed workspace extent; no coordination label is observed.

The centralized expert searches all 24 crossing permutations, realizes them
with collision-free time separation, and validates them in the actual plant.
Its `CrossingOrder` is analysis/expert metadata only, never policy input.

Run the Gate A/B pilot:

```bash
python -m four_way_intersection.pilot --count 30
pytest -q four_way_intersection/tests
```
