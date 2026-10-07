# OrthoFlow3 basin-learning and selector-matrix v1

## Primary status

**BASIN_SELECTOR_MATRIX_DATASET_REQUIRES_APPROVAL**

The experiment stopped at its mandatory cost preflight. No new rollout was launched and no network was trained.

## Verified-ball inventory

Only six labels satisfy all eligibility conditions under the corrected continuous `E_bridge`:

| Split | Available | Minimum | Shortfall |
|---|---:|---:|---:|
| TRAIN | 3 | 24 | 21 |
| VAL | 1 | 8 | 7 |
| TEST | 2 | 8 | 6 |
| Total | 6 | 40 | 34 |

All six centers are 64/64, all balls are contained in `E_bridge`, and their 18 promoted inside points are 64/64. Radius mean/median/min/max is 0.24537 / 0.25766 / 0.01594 / 0.35362. Source-group overlap across splits is zero.

Historical disconnected-domain zero-radius balls were rejected. The axis-aligned ellipsoid labels are forbidden because that representation produced two confirmed internal false inclusions.

## Frozen low-J baseline

The existing `G_LOWJ` reference remains frozen at checkpoint SHA256:

`bd660db3ac501e5e77755af65cee5c01ba7d30810a4cf6001170bdbcebbb05d7`

It was not retrained or evaluated in this preflight.

## Exact budget consequence

The accepted six-state continuous-ball build used 12,688 new continuations and 3,498,278 physical steps. To fill the split shortages without outcome-based selection requires 34 additional labels.

| Projection | New continuations | Physical steps | Idealized 6-shard rollout wall time |
|---|---:|---:|---:|
| Empirical accepted-protocol scaling | 71,899 | 19,823,575 | 74.3 min |
| Frozen component-wise preflight scaling | 74,936 | 20,719,804 | 77.4 min |

The automatic limits are 30,000 continuations and 12,000,000 steps. Label generation alone is approximately 2.40–2.50 times the continuation cap and 1.65–1.73 times the physical-step cap. The downstream predicted-ball audit, selector matrix, perturbation tests, and Fresh-WIDE evaluation are not included in these figures.

Six shards are authorized when the server is idle and would keep the label-build wall time below three hours. They do not change the continuation or physical-step overrun.

## Scientific decision

Training a 214-D learner from the six available states is explicitly forbidden and scientifically underresolved. The controlled comparisons `G_LOWJ` vs `G_CENTER_DIRECT`, `H_REG` vs `H_MARGIN`, and CENTER vs DEF vs TRADEOFF therefore cannot yet be answered.

Approval would need to cover at least the 34-state verified-ball expansion—approximately 74,936 new continuations and 20.72 million physical steps—plus a separately frozen allowance for downstream validation. Robust B63, limiting-ray certification, and independent inside-ball validation should remain unchanged.
