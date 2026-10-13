# Joint four-scene physical-only critic: fresh confirmation

B15 means at least 15/16 successes; every row uses 32 independent physical states.

| Scene | Safety | Joint eta-only | Joint state | Same-scene mismatch mean | Frozen LOSO state | Frozen within state |
|---|---:|---:|---:|---:|---:|---:|
| toy_give_way | 16/32 | 25/32 | 27/32 | 22.86/32 | 12/32 | 28/32 |
| double_bottleneck | 7/32 | 32/32 | 32/32 | 32.00/32 | 32/32 | 32/32 |
| four_way_intersection | 0/32 | 30/32 | 29/32 | 29.73/32 | 31/32 | 30/32 |
| ring_exchange | 22/32 | 26/32 | 27/32 | 27.00/32 | 7/32 | 30/32 |

All paired state-bootstrap intervals, mean success, collisions and numerical upper/lower bounds are in results.json. Models and 100 same-scene derangements were frozen before labels. This selected-only confirmation has no full-pool oracle; the independent development cache has complete K16 and fixed-bank diagnostics.
