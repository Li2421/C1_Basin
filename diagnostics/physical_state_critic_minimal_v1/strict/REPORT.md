# Cache-first physical-only critic pilot (strict source)

This is an exploratory, zero-new-rollout result on historical FF execution outcomes.
Both critics used the same cached source labels and K16 target candidates;
the state critic received masked physical entities plus eta, with Flow/controller channels zero.
All eight predictions were frozen before this script opened target truths.
Historical target labels were already opened in earlier studies, and source state coverage
is limited/heterogeneous; do not read this as fresh-state formal confirmation.

The strict-source variant excludes all-scene-generator-derived source proposals, following the archived strict LOSO source rule.

| Panel | Train scenes | Test N | Safety | Q(eta) | Q(s,eta) | Oracle | State−eta B15 pp [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| within_toy | toy_giveway | 48 | n/a | 29/48 | 43/48 | 48/48 | +29.2 [+14.6, +43.8] |
| within_db | double_bottleneck | 24 | 4/24 | 24/24 | 24/24 | 24/24 | +0.0 [+0.0, +0.0] |
| within_four | four_way_intersection | 24 | 1/24 | 21/24 | 21/24 | 24/24 | +0.0 [-12.5, +12.5] |
| within_ring | ring_exchange | 60 | 42/60 | 12/60 | 50/60 | 60/60 | +63.3 [+51.7, +75.0] |
| loso_toy | double_bottleneck, four_way_intersection, ring_exchange | 48 | n/a | 30/48 | 33/48 | 48/48 | +6.2 [-12.5, +25.0] |
| loso_db | toy_giveway, four_way_intersection, ring_exchange | 24 | 4/24 | 16/24 | 23/24 | 24/24 | +29.2 [+8.3, +50.0] |
| loso_four | toy_giveway, double_bottleneck, ring_exchange | 24 | 1/24 | 24/24 | 22/24 | 24/24 | -8.3 [-20.8, +0.0] |
| loso_ring | toy_giveway, double_bottleneck, four_way_intersection | 60 | 42/60 | 28/60 | 48/60 | 60/60 | +33.3 [+18.3, +48.3] |

Toy Safety is n/a: its historical extra controller was MAC_ONLY and has different
success semantics. DB/Four/Ring Safety is hard-safety Q16 from the shared database.
Mean success, oracle regret, collisions, numerical bounds and paired state-level
statistics are in each panel's metrics.json. Full-scale paused evidence remains intact.

| Panel | Observed state B15 | Mean B15 after 32 physical-state derangements |
|---|---:|---:|
| within_toy | 43/48 | 31.3/48 |
| within_db | 24/24 | 24.0/24 |
| within_four | 21/24 | 21.0/24 |
| within_ring | 50/60 | 49.7/60 |
| loso_toy | 33/48 | 33.0/48 |
| loso_db | 23/24 | 23.1/24 |
| loso_four | 22/24 | 22.1/24 |
| loso_ring | 48/60 | 47.8/60 |

State derangement retains every target eta pool and frozen critic weight; a drop
indicates the selector uses the physical state rather than only a global eta tendency.

| Panel | Q(s,eta) B15 | Wider Q(eta) B15 | State−wider eta pp [95% CI] |
|---|---:|---:|---:|
| within_toy | 43/48 | 33/48 | +20.8 [+6.2, +35.4] |
| within_db | 24/24 | 24/24 | +0.0 [+0.0, +0.0] |
| within_four | 21/24 | 21/24 | +0.0 [+0.0, +0.0] |
| within_ring | 50/60 | 49/60 | +1.7 [-5.0, +8.3] |
| loso_toy | 33/48 | 30/48 | +6.2 [-12.5, +25.0] |
| loso_db | 23/24 | 15/24 | +33.3 [+12.5, +54.2] |
| loso_four | 22/24 | 23/24 | -4.2 [-16.7, +8.3] |
| loso_ring | 48/60 | 15/60 | +55.0 [+40.0, +68.4] |

The wider eta-only control uses the same source labels and validation rule
with 32→128→64 eta/head widths matching the state critic's downstream head.
