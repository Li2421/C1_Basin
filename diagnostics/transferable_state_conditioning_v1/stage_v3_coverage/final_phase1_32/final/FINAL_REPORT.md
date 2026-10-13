# Transferable physical-only Q(s,η): phase-1 confirmation

**Verdict: `CROSS_SCENE_STATE_CONDITIONING_NOT_ESTABLISHED`.**

All numbers below use frozen, naturally sampled physical states. The stochastic K16 and common fixed-η bank are distinct candidate panels.

This is the prospectively frozen first-32-state tranche per scene. A positive or borderline result requires expansion to the remaining untouched TEST states before a full positive claim.

## Source-only model choice

All critic inputs use the masked physical-entity encoding: physical positions, velocities, goals, fixed map geometry and η only. Flow, response/context, policy and controller channels are zeroed. Both critics used identical source rollout labels; the eta-only head has the same downstream width as the state critic.

| Panel | Chosen state objective | Source-val concat regret | Source-val residual regret | Source-val eta-only regret |
|---|---|---:|---:|---:|
| within_toy_give_way | state_concat | 0.0000 | 0.0078 | 0.0391 |
| within_double_bottleneck | state_concat | 0.0000 | 0.0000 | 0.0000 |
| within_four_way_intersection | state_residual | 0.0000 | 0.0000 | 0.0000 |
| within_ring_exchange | state_concat | 0.0000 | 0.0156 | 0.0156 |
| loso_toy_give_way | state_concat | 0.0052 | 0.0156 | 0.0156 |
| loso_double_bottleneck | state_concat | 0.0156 | 0.0182 | 0.0208 |
| loso_four_way_intersection | state_residual | 0.0156 | 0.0156 | 0.0312 |
| loso_ring_exchange | state_residual | 0.0104 | 0.0026 | 0.0104 |

## Within-scene frozen stochastic K16: B15 / physical states

| Scene | Safety | Q(η) | matched Q(s,η) | mismatched state mean | Oracle | state−eta B15 95% CI | state−mismatch B15 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toy (32) | 20/32 | 27/32 | 30/32 | 25.0/32 | 31/32 | +9.4 pp [+0.0, +18.8] | +15.5 pp [+8.3, +23.9] |
| DB (32) | 4/32 | 32/32 | 32/32 | 32.0/32 | 32/32 | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] |
| Four-Way (32) | 0/32 | 30/32 | 30/32 | 30.0/32 | 32/32 | +0.0 pp [+0.0, +0.0] | +0.0 pp [+0.0, +0.0] |
| Ring (32) | 22/32 | 29/32 | 30/32 | 30.3/32 | 32/32 | +3.1 pp [-6.2, +12.5] | -0.9 pp [-6.8, +4.3] |

## Critic-only LOSO, frozen stochastic K16: B15 / physical states

| Held-out scene | Safety | Q(η) | matched Q(s,η) | mismatched state mean | Oracle | state−eta B15 95% CI | state−mismatch B15 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toy (32) | 20/32 | 22/32 | 17/32 | 17.0/32 | 31/32 | -15.6 pp [-34.4, +0.0] | +0.0 pp [+0.0, +0.0] |
| DB (32) | 4/32 | 29/32 | 31/32 | 30.8/32 | 32/32 | +6.2 pp [-6.2, +18.8] | +0.6 pp [-0.7, +2.4] |
| Four-Way (32) | 0/32 | 31/32 | 28/32 | 28.0/32 | 32/32 | -9.4 pp [-21.9, +0.0] | +0.0 pp [+0.0, +0.0] |
| Ring (32) | 22/32 | 5/32 | 6/32 | 6.0/32 | 32/32 | +3.1 pp [-12.5, +18.8] | +0.1 pp [-4.9, +4.3] |

## Common fixed-η bank: independent evaluation half (8 seeds)

| Panel | Q(η) | matched Q(s,η) | mismatched | best fixed, discovery | best fixed, posthoc evaluation | Oracle | state−bootstrap-reselected best fixed 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| within_toy_give_way (32) | 55.5% | 96.1% | 49.6% | 90.2% | 90.2% | 100.0% | +5.9 pp [-6.2,+18.4] |
| within_double_bottleneck (32) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | +0.0 pp [+0.0,+0.0] |
| within_four_way_intersection (32) | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | +0.0 pp [+0.0,+0.0] |
| within_ring_exchange (32) | 82.4% | 82.4% | 82.4% | 82.4% | 82.4% | 87.9% | +0.0 pp [+0.0,+0.0] |
| loso_toy_give_way (32) | 44.5% | 44.5% | 44.5% | 90.2% | 90.2% | 100.0% | -45.7 pp [-63.3,-29.3] |
| loso_double_bottleneck (32) | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | +0.0 pp [+0.0,+0.0] |
| loso_four_way_intersection (32) | 93.0% | 93.0% | 93.0% | 100.0% | 100.0% | 100.0% | -7.0 pp [-12.1,-3.1] |
| loso_ring_exchange (32) | 0.4% | 0.4% | 0.4% | 82.4% | 82.4% | 87.9% | -82.0 pp [-91.4,-71.5] |

## State-conditioning mechanism on the common bank

| Panel | state-effect corr. | strong reversal pairs | switch accuracy Q(η) | switch accuracy Q(s,η) | selection entropy (bits) | matched−mismatch eval Q 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| within_toy_give_way | 0.707 | 47 | 51.5% | 95.3% | 0.97 | +46.5 pp [+39.0, +52.3] |
| within_double_bottleneck | -0.142 | 0 | n/a | n/a | -0.00 | +0.0 pp [+0.0, +0.0] |
| within_four_way_intersection | 0.051 | 0 | n/a | n/a | -0.00 | +0.0 pp [+0.0, +0.0] |
| within_ring_exchange | -0.072 | 8 | 74.4% | 72.2% | -0.00 | +0.0 pp [+0.0, +0.0] |
| loso_toy_give_way | -0.066 | 47 | 50.7% | 52.4% | -0.00 | +0.0 pp [+0.0, +0.0] |
| loso_double_bottleneck | 0.000 | 0 | n/a | n/a | -0.00 | +0.0 pp [+0.0, +0.0] |
| loso_four_way_intersection | 0.024 | 0 | n/a | n/a | -0.00 | +0.0 pp [+0.0, +0.0] |
| loso_ring_exchange | 0.023 | 8 | 41.4% | 54.1% | -0.00 | +0.0 pp [+0.0, +0.0] |

Equal-scene-weighted LOSO matched−mismatch mean success: +0.3 pp, 95% state-bootstrap CI [-0.9, +1.3], 4 scenes / 128 physical states.

Equal-scene-weighted LOSO Q(s,η)−Safety mean success: +3.4 pp, 95% state-bootstrap CI [-3.3, +10.1].

The LOSO statement is limited to **critic** transfer conditional on frozen historical proposal generators. Those generators previously used scene-specific training data; this is not end-to-end zero-shot transfer of the whole pipeline. The common fixed bank removes generator response context from candidate selection, while the frozen FF execution pipeline still uses Flow to execute actions.

Every panel's JSON records mean success, oracle regret, B15 bounds, collisions, numerical failures, deadlocks, timeouts, all 16 fixed-policy outcomes, adaptive-demand and reversal groups, state-level paired bootstrap intervals, and individual failure cases. Exact per-seed evidence is in the shared rollout DB and the frozen raw/NPZ artifacts.
