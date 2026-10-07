# OrthoFlow3 analytical minimum-deformation stability audit

## Decision

**`ANALYTIC_MINDEF_INADEQUATE`.** All 24 frozen states (12 anchors plus their deterministic `t+4` neighbors) have exactly two confirmed B63 eta candidates after 144 new enrichment continuations. All 12 local pairs are evaluable. No network was trained.

True `J_def` is accumulated in `run_subset_arms.py:143-176` as `dt * sum ||u_exec-u_safe||²` from the original absolute episode time to terminal/horizon 850. Canonical aggregation gates on B63 first, averages J only over successful continuations, then minimizes mean successful J with eta-index tie-breaking.

| proxy | active within-state Spearman / Kendall | median / mean / p90 regret ratio | within 5% / 10% / 20% | mean local eta movement |
|---|---:|---:|---:|---:|
| raw eta norm | -0.400 / -0.400 | 1.174 / 1.336 / 1.902 | 30% / 40% / 50% | 0.141 |
| normalized eta norm | -0.400 / -0.400 | 1.174 / 1.336 / 1.902 | 30% / 40% / 50% | 0.141 |
| start Gram energy | -1.000 / -1.000 | 1.412 / 1.477 / 1.902 | 10% / 20% / 30% | 0.187 |
| one-step executed correction | -1.000 / -1.000 | 1.412 / 1.477 / 1.902 | 10% / 20% / 30% | 0.187 |

True-J selection movement is exactly 0 on all 12 `t+4` pairs, whereas analytical selectors introduce mean movement 0.141--0.187. There are no large true-J jumps, so near-tie-switch attribution is not applicable.

## Geometry and projection

Median Gram diagonals `(M11,M22,M33)` are `(0.5, 0.002354, 0.5)`; median cross terms `(M12,M13,M23)` are `(2.65e-13, -0.202, -0.000196)`. Goal and flow-perpendicular are essentially orthogonal (`cos12` median 6.5e-12), but goal/relative coupling remains material (`cos13` median -0.616). In active states, diagonal terms account for a median 87.4% of absolute Gram contributions and cross terms 12.6%; the dimensions are interpretable but not fully separable.

The second solver is a Euclidean projection onto a closed convex feasible set and `u_safe` is feasible, so non-expansiveness legitimately gives `||delta u|| <= ||g||` up to numerical tolerance. Empirically the ratio has median 0.441, P95 0.793, maximum 1.000000, with zero violations above `1+1e-7`.

## Implication

The high pooled correlations on all states are driven by trivial zero-sufficient cases. On ACTIVE-required states the simple start-local proxies do not rank full-horizon deformation and incur meaningful regret. A learned `J(h,eta)` is therefore scientifically justified as a later controlled possibility, though this audit does not train it. The proposed `Q + analytical R + Direct-eta G` pipeline needs revision: feasibility Q may still be useful, but these analytical proxies should not be the sole deformation objective.

Smallest next experiment: promote two additional predeclared low-frontier Sobol candidates on the same 10 ACTIVE-required states, repeat the ranking/regret audit, and only then decide whether to train J.
