# J_def total / per-step / duration decomposition

## Scope and exact identities

This is a post-hoc analysis of existing trajectories only; no rollout, controller call, optimization, or training was performed. Metrics are computed only for successful rollouts belonging to high-confidence `SUCCESS_CELL`s under the frozen criterion.

For every successful rollout:

```text
J_def_total    = dt * sum_k ||delta_u[k]||^2
T              = N * dt
J_def_stepmean = (1/N) * sum_k ||delta_u[k]||^2
J_def_rms      = sqrt(J_def_stepmean)
```

Therefore `J_def_total = T * J_def_stepmean` exactly, and RMS has exactly the same ranking as stepmean. The optional active fraction uses `||delta_u|| > 0.01` action units, chosen as 2% of the frozen per-agent `vmax=0.5`; it is diagnostic only.

## Cell-level Spearman results

| state | success cells | rho(total,T) | rho(total,stepmean) | rho(total,RMS) | rho(total,peak) | total-vs-step rank rho | log-duration share | top-5 overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D1 | 111 | 0.604 | 0.889 | 0.889 | 0.934 | 0.889 | 0.221 | 3/5 |
| D2 | 133 | -0.042 | 0.909 | 0.909 | 0.913 | 0.909 | -0.027 | 3/5 |
| D4 | 133 | -0.190 | 0.890 | 0.890 | 0.909 | 0.890 | -0.095 | 4/5 |

The complete 5×5 matrices at both successful-rollout and eta-cell-mean levels are in `correlations.json`.

## Minima comparison

| state | eta_total* | mean total | eta_stepmean* | mean stepmean | coincide | total* rank by step | step* rank by total |
|---|---|---:|---|---:|---:|---:|---:|
| D1 | (0.5703125, -0.375, -0.125) | 0.234068 | (0.578125, -0.3125, -0.1875) | 0.020929 | NO | 2/111 | 8/111 |
| D2 | (0.359375, -0.5, 0.0) | 0.202133 | (0.359375, -0.5, 0.0) | 0.012687 | YES | 1/133 | 1/133 |
| D4 | (0.375, -0.40625, -0.0625) | 0.191477 | (0.359375, -0.4375, -0.0625) | 0.011573 | NO | 2/133 | 4/133 |

## Representative cells

| example | eta | total | stepmean | RMS | duration | peak |
|---|---|---:|---:|---:|---:|---:|
| D1 total minimum | (0.5703125, -0.375, -0.125) | 0.234068 | 0.021061 | 0.145083 | 11.115 | 0.313459 |
| D2 total minimum | (0.359375, -0.5, 0.0) | 0.202133 | 0.012687 | 0.112572 | 15.953 | 0.308911 |
| D4 total minimum | (0.375, -0.40625, -0.0625) | 0.191477 | 0.011583 | 0.107454 | 16.715 | 0.253492 |
| duration-assisted cell within lowest-total quintile | (0.5, -0.625, 0.0) | 0.214568 | 0.017192 | 0.131101 | 12.481 | 0.384868 |
| long episode among gentlest per-step quintile | (0.375, -0.40625, -0.09375) | 0.198812 | 0.011632 | 0.107669 | 17.330 | 0.234258 |

## Answers

1. Duration is not the main driver. D1 shows a moderate positive duration association, but stepmean is substantially stronger; D2/D4 have near-zero or negative total-duration rank correlation because longer successful episodes also tend to use gentler corrections.
2. The total-cost minima are also extremely gentle per step: their stepmean ranks are 2/111, 1/133, and 2/133 for D1/D2/D4.
3. Eta rankings change modestly rather than substantially: total-versus-stepmean rho is 0.889–0.909, with top-five overlaps of 3/5, 3/5, and 4/5. D1 and D4 minima do not exactly coincide, but remain near the top under the other metric.
4. Only successful rollouts are aggregated, so early deadlock/timeout cannot manufacture a low total here. The representative duration-assisted cell tests the strongest rank reversal within the lowest-total quintile; it does not overturn the overall per-step-dominated ordering.
5. Long, gentle successful episodes exist (representative row above), and total cost appropriately charges them for sustained intervention.

The duration coupling is mathematically and semantically consistent with accumulated intervention energy. It is measurable—especially for D1—but not dominant, and it does not pull any total-cost minimum toward an aggressive per-step policy. On the existing successful data there is no evidence of a pathological “terminate quickly with aggressive correction” mechanism controlling the minima.

## Conclusion

**TOTAL_COST_SEMANTICALLY_CLEAN**

The current primary objective remains unchanged. This analysis does not recommend replacing it with stepmean and does not introduce a new optimization objective.
