# Oracle-domain resolution for this audit

The historical 424 labels do not share one continuous search domain.

- The 413 base/startup states used a finite frozen oracle set: exact eta zero followed, when needed, by eight active candidates from `gphi_training_dataset_v3/protocol.json`. V1/V2/V3 use robust seeds 95210001–95210064; startup uses 95710001–95710064.
- The 11 appended strict-deadlock states used the separately frozen `strict_deadlock_success_basin_capacity_v1` active grid/refinement/Sobol search with domain `[0.5,1.25] × [-0.5,0.5] × [0,0.75]` and seeds 95310001–95310064.
- The learned predictor bounds `[0,-0.53125,-0.125]` to `[1.25,0.5,0.75]` are explicitly a union normalization envelope. They are not an historical oracle search domain.

Consequently:

1. Cached Stage E membership uses each state's actual source-oracle candidate semantics. No new continuous-domain claim is made.
2. Fresh-WIDE Stage G uses the original **general-WIDE training oracle**: test eta zero first; only if zero is not B63, test the exact eight active candidates below with the frozen success-first, minimum-mean-success-J_def, lexicographic tie-break rule.

```text
(0.5703125, -0.375,   -0.125)
(0.578125,  -0.40625, -0.125)
(0.578125,  -0.34375, -0.125)
(0.40625,   -0.5,      0.0)
(0.4375,    -0.53125,  0.0)
(0.375,     -0.4375,  -0.0625)
(0.40625,   -0.4375,  -0.0625)
(1.0,        0.0,      0.25)
```

This choice was frozen before Stage G oracle outcomes. It directly measures capacity of the oracle supervision family underlying 413/424 training states, but `NO_ROBUST_ETA_FOUND` means “none in this frozen finite original search,” not proof that the continuous 3-D family has no solution.
