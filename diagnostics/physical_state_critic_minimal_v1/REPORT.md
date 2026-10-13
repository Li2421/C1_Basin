# Cache-first physical-only critic pilot (enriched source)

This is an exploratory, zero-new-rollout result on historical FF execution outcomes.
Both critics used the same cached source labels and K16 target candidates;
the state critic received masked physical entities plus eta, with Flow/controller channels zero.
All eight predictions were frozen before this script opened target truths.
Historical target labels were already opened in earlier studies, and source state coverage
is limited/heterogeneous; do not read this as fresh-state formal confirmation.

The enriched-source variant is ineligible for strict LOSO because some source proposals may have ancestry in an all-scene generator; its LOSO rows are exploratory only.

| Panel | Train scenes | Test N | Safety | Q(eta) | Q(s,eta) | Oracle | State−eta B15 pp [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| within_toy | toy_giveway | 48 | n/a | 29/48 | 43/48 | 48/48 | +29.2 [+14.6, +43.8] |
| within_db | double_bottleneck | 24 | 4/24 | 24/24 | 24/24 | 24/24 | +0.0 [+0.0, +0.0] |
| within_four | four_way_intersection | 24 | 1/24 | 21/24 | 21/24 | 24/24 | +0.0 [+0.0, +0.0] |
| within_ring | ring_exchange | 60 | 42/60 | 50/60 | 57/60 | 60/60 | +11.7 [+5.0, +20.0] |
| loso_toy | double_bottleneck, four_way_intersection, ring_exchange | 48 | n/a | 35/48 | 38/48 | 48/48 | +6.2 [-8.3, +20.8] |
| loso_db | toy_giveway, four_way_intersection, ring_exchange | 24 | 4/24 | 14/24 | 14/24 | 24/24 | +0.0 [-16.7, +16.7] |
| loso_four | toy_giveway, double_bottleneck, ring_exchange | 24 | 1/24 | 24/24 | 23/24 | 24/24 | -4.2 [-12.5, +0.0] |
| loso_ring | toy_giveway, double_bottleneck, four_way_intersection | 60 | 42/60 | 0/60 | 7/60 | 60/60 | +11.7 [+5.0, +20.0] |

Toy Safety is n/a: its historical extra controller was MAC_ONLY and has different
success semantics. DB/Four/Ring Safety is hard-safety Q16 from the shared database.
Mean success, oracle regret, collisions, numerical bounds and paired state-level
statistics are in each panel's metrics.json. Full-scale paused evidence remains intact.

| Panel | Observed state B15 | Mean B15 after 32 physical-state derangements |
|---|---:|---:|
| within_toy | 43/48 | 31.3/48 |
| within_db | 24/24 | 24.0/24 |
| within_four | 21/24 | 21.0/24 |
| within_ring | 57/60 | 55.8/60 |
| loso_toy | 38/48 | 38.0/48 |
| loso_db | 14/24 | 13.9/24 |
| loso_four | 23/24 | 23.1/24 |
| loso_ring | 7/60 | 7.0/60 |

State derangement retains every target eta pool and frozen critic weight; a drop
indicates the selector uses the physical state rather than only a global eta tendency.

| Panel | Q(s,eta) B15 | Wider Q(eta) B15 | State−wider eta pp [95% CI] |
|---|---:|---:|---:|
| within_toy | 43/48 | 33/48 | +20.8 [+6.2, +35.4] |
| within_db | 24/24 | 24/24 | +0.0 [+0.0, +0.0] |
| within_four | 21/24 | 21/24 | +0.0 [+0.0, +0.0] |
| within_ring | 57/60 | 54/60 | +5.0 [+0.0, +11.7] |
| loso_toy | 38/48 | 41/48 | -6.2 [-20.8, +6.2] |
| loso_db | 14/24 | 15/24 | -4.2 [-25.0, +12.5] |
| loso_four | 23/24 | 23/24 | +0.0 [+0.0, +0.0] |
| loso_ring | 7/60 | 11/60 | -6.7 [-20.0, +6.7] |

The wider eta-only control uses the same source labels and validation rule
with 32→128→64 eta/head widths matching the state critic's downstream head.
