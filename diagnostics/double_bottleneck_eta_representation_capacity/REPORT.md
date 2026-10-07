# Double-Bottleneck eta representation-capacity audit

Date: 2026-09-25  
Scope: frozen S-XL-128 MACFlow, frozen four-agent environment, frozen first and second hard-safety projections, fixed 850-step horizon, and diagnostic episode-level eta representations only. No model was trained. No `G_phi`, safety change, horizon change, mode label, phase label, per-step free eta, or environment change was introduced.

## Executive result and registered stop

The experiment reached the user's explicit H0 stop condition before a full structural attribution was scientifically valid.

The previous 3D study found one shared point, A004, that rescued `15/61` safe timeouts. A denser 256-point P0 search on the pre-registered pilot set found six additional successful P0 points. Applying those six fixed points unchanged to all 61 frozen timeout episodes rescued `32/61`; the union with A004 rescued:

```text
37 / 61 = 60.66%
```

This includes 22 episodes not rescued by A004. All 510 full-population inheritance rollouts were collision-free. The same observed P0 set preserved at least one successful parameter for `16/24` baseline-success controls.

This more than doubles the prior observed target coverage and materially changes the earlier `15/61` conclusion. Per the requested protocol—“if dense 3D search dramatically changes the conclusion, stop before interpreting representation capacity”—the promoted Agent6 full sweep was interrupted at `2,839/10,965` rollouts, isolated under `incomplete_after_h0_stop/`, and excluded from every scientific count below.

**Primary conclusion: H0 SEARCH INSUFFICIENCY.** The prior 65-point 3D result was not dense enough to support rejection of the current 3D representation.

![Full-population P0 coverage](figures/h0_full_population_coverage.svg)

## 1. Original 3D search sufficiency audit

### Frozen P0 formulation

The canonical implementation was re-read rather than inferred. At every physical step:

```text
B_goal,i = radial_bound_0.5(goal_i - position_i)
B_rel,i  = (1/(N-1)) sum_(j != i) radial_bound_0.5(position_i - position_j)
g_i      = alpha B_goal,i + beta u_safe,i + gamma B_rel,i
u_exec   = Pi_U(x)(u_safe + g)
```

Each pair vector is bounded before averaging. There is no clipping of raw `g`. The second projection enforces the unchanged wall constraints, all six unordered agent pairs, and four per-agent `0.5 m/s` speed balls. At `N=2`, the relational mean contains exactly one term and reduces to the Toy two-agent definition.

The global P0 domain remained unchanged:

```text
alpha in [ 0.50, 1.25]
beta  in [-0.50, 0.50]
gamma in [ 0.00, 0.75]
```

### Dense pilot audit

The pre-registered pilot contained 4 previous P0-positive timeouts, 8 uniformly selected previous P0-negative timeouts, and 6 baseline-success controls. Every episode received the same 255-point scrambled Sobol design plus A004; eta zero was an additional baseline sentinel.

| Pilot stratum | Episodes with a nonzero P0 success |
|---|---:|
| Previous P0-positive | 4/4 |
| Previous P0-negative | 2/8 |
| Baseline-success controls | 4/6 |

The `2/8` newly positive result was below the originally registered `3/8` early-stop threshold, so the structural pilot was allowed to run. There were 4,626 P0 rollouts, zero collisions, and no eta-zero reproduction error.

### Full-population cross-state inheritance

The dense pilot exposed seven unique successful P0 points: A004 plus six new points. Since P1/P2/P3 must contain P0 exactly, omitting newly discovered P0 points would confound representation capacity with the probability that a high-dimensional Sobol design lands on a narrow P0 subspace. `H0_INHERITANCE_AMENDMENT.json` therefore registered, before structural full results were available, one global rule: embed all six new P0 points exactly and apply them to every episode.

| New P0 point | Timeout targets rescued | Success controls preserved |
|---|---:|---:|
| A001 | 7/61 | 4/24 |
| A037 | 6/61 | 4/24 |
| A073 | 10/61 | 3/24 |
| A113 | 9/61 | 8/24 |
| A145 | 0/61 | 1/24 |
| A217 | 10/61 | 5/24 |
| Six-point union | 32/61 | 15/24 |
| A004 + six-point union | **37/61** | **16/24** |

The six points were selected from pilot success, so `37/61` is an observed-existence lower bound, not an unbiased basin-volume estimate. It nevertheless directly falsifies the practical premise that only 15 of the 61 frozen cases have an observed P0 basin in the registered domain.

## 2. Executable-action expert-fit audit

Exactly three action-state indices were sampled uniformly from each of the 18 frozen pilot trajectories. No waiting, wall, bottleneck, goal, or failure-region filter was used. The unchanged centralized expert was queried under all eight hypotheses; only a validated collision-free continuation reaching all goals was retained.

The expert succeeded from only `7/54 = 12.96%` sampled states. The other 47 states were not replaced or locally resampled. This is a material limitation: the fit statistics are a diagnostic slice, not a population estimate.

For each valid state, the diagnostic oracle solved

```text
min_theta || Pi_U(x)(u_safe + g_theta(x)) - Pi_U(x)(u_expert) ||
```

within the globally fixed coefficient ranges. Per-state theta was never executed as a controller.

| Representation | Median best executable residual | Relative improvement over P0 |
|---|---:|---:|
| P0-3D | 0.4733 m/s | — |
| P1-Agent6 | 0.4149 m/s | 12.34% |
| P2-Pair8 | **0.3255 m/s** | **30.34%** |
| P3-Temporal6 | 0.4755 m/s | approximately 0% |

Pair-specific relation coefficients improve local oracle fit on this small valid subset. They did not create new pilot episode basins beyond dense P0. Therefore local first-action fit alone is insufficient evidence for H2 as the episode-level bottleneck.

![Offline expert fit](figures/offline_expert_fit.svg)

## 3. Projection-coupling / effective-rank analysis

The audited map was the executable correction

```text
F_x(theta) = Pi_U(x)(u_safe + g_theta(x)) - u_safe,
```

using normalized finite-difference steps of `1/64` and numerical rank threshold `max(1e-3, 0.01 sigma_max)`.

| Representation | Median raw rank | Median executable rank | Lost directions | Median retention at oracle fit |
|---|---:|---:|---:|---:|
| P0-3D | 3 | 3 | 0 | 0.8765 |
| P1-Agent6 | 6 | 3 | **3** | 0.7696 |
| P2-Pair8 | 6 | 5 | 1 | 0.9066 |
| P3-Temporal6 | 3 | 3 | 0 | 0.8764 |

P3 has six episode parameters but only one interpolated 3-vector is active at a fixed state, so local rank three is expected and is not itself evidence against temporal freedom.

Across successful pilot rollouts, the median raw/executed joint correction norms were:

| Representation | Raw correction | Executed correction | Median retention | Median mostly-rewritten fraction |
|---|---:|---:|---:|---:|
| P0-3D | 0.5068 | 0.1599 | 0.3139 | 81.56% |
| P1-Agent6 | 0.5087 | 0.1597 | 0.3030 | 78.16% |
| P2-Pair8 | 0.4937 | 0.1514 | 0.3156 | 78.46% |
| P3-Temporal6 | 0.4928 | 0.1514 | 0.3154 | 78.18% |

Thus H4 is real: raw coefficient freedom is strongly compressed during actual successful rollouts. Agent6 is especially affected locally, losing a median three sensitivity directions. This does not prove that projection is the primary failure cause because P0 search insufficiency already invalidates the earlier capacity conclusion.

![Projection effective rank](figures/projection_effective_rank.svg)

![Pilot projection coupling](figures/pilot_projection_coupling.svg)

## 4. P0 — current 3D results

On the 12 timeout pilot cases, dense P0 basin existence was `6/12`: all four old positives plus two old negatives. The dense design discovered six new successful points in addition to A004.

On all 61 frozen timeouts, the observed seven-point P0 union rescued `37/61`, compared with `15/61` for A004 alone. This full-population reuse result is the governing finding of this phase.

No claim is made that 37 is the final P0 capacity. A complete 256-point P0 map was not run across all 61 episodes; more P0 basins may remain undiscovered.

## 5. P1 — Agent6 pilot results

P1 used

```text
g_i = alpha_i B_goal,i + beta u_safe,i + gamma B_rel,i
theta = [alpha_1, alpha_2, alpha_3, alpha_4, beta, gamma].
```

All scalar domains match their P0 counterparts, and setting all four `alpha_i` equal exactly reproduces P0. The 129-point Stage-A design, exact P0 inheritance, 128-point same-domain negative confirmation, and fixed `1/32` normalized local checks produced:

```text
9/12 pilot timeout episodes with an observed basin
gain over dense P0: +3 episodes
median local success fraction: 42.86%
```

The three genuinely new episodes, for which dense P0 had no success, had local success fractions `19/28`, `19/28`, and `6/28` (67.86%, 67.86%, and 21.43%). This passed the pilot promotion rule. Agent6 was the only promoted structure.

This is credible pilot evidence for H1, but it is not a full-population Agent6 capacity result. The full sweep was stopped and excluded after H0 changed the P0 conclusion.

## 6. P2 — Pair8 pilot results

P2 used one global goal coefficient, one global safe-action coefficient, and six symmetric unordered-pair relation coefficients:

```text
g_i = alpha B_goal,i + beta u_safe,i
      + (1/3) sum_(j != i) gamma_ij b_ij,
gamma_ij = gamma_ji.
```

All six equal `gamma` exactly reproduces P0. P2 achieved `6/12` pilot basin existence, exactly the dense-P0 set, with median local success fraction 42.19%. Pair-specific perturbations remained successful around the two newly discovered P0 basins, but no episode that failed dense P0 was newly rescued. P2 was not promoted.

The contrast—30.34% better local expert-fit residual but no new episode basin—is evidence that improved local action span did not resolve the tested long-horizon failures.

## 7. P3 — Temporal6 pilot results

P3 used two 3-vectors and the globally fixed interpolation

```text
s_k = k / 849
eta_k = (1-s_k) eta_start + s_k eta_end.
```

It exactly reduces to P0 when `eta_start = eta_end`. P3 achieved `6/12` pilot basin existence, exactly the dense-P0 set, with median local success fraction 33.93%. It rescued no dense-P0-negative episode and was not promoted. The pilot provides no positive evidence that generic linear time variation is the primary missing freedom.

## 8. Optional P4

P4-Agent12 was not run. The registered trigger required P1 to beat P0 by at least four targets, beat P2 and P3 by at least two, remain below 10/12, and retain a local-fit residual above 0.03. P1 improved by three, so the trigger was false. This avoided dimensional expansion for its own sake.

## 9. Full-rollout basin-existence comparison

![Pilot basin existence](figures/pilot_basin_existence.svg)

| Representation | Completed pilot target existence | Promoted? | Completed full-61 structural result |
|---|---:|---:|---|
| P0-3D | 6/12 dense pilot; 37/61 observed seven-point union | reference | H0 result valid |
| P1-Agent6 | 9/12 | yes | **stopped at 2,839/10,965 and excluded** |
| P2-Pair8 | 6/12 | no | not run |
| P3-Temporal6 | 6/12 | no | not run |
| P4-Agent12 | — | trigger false | not run |

The partial P1 full data are retained only as an audit trail. They are not used to estimate existence, robustness, correction energy, preservation, or attribution.

## 10. Basin and stochastic-seed robustness

Local geometric robustness was completed on the pilot as described above. The pre-registered multi-seed MACFlow reruns were not launched because the mandatory H0 stop occurred first. Running them after the stop would have continued an invalid structural-comparison branch. No stochastic-robustness claim is made.

## 11. Baseline-success preservation

On the six pilot controls, at least one nonzero successful point existed for 4/6 P0, 5/6 Agent6, 4/6 Pair8, and 4/6 Temporal6 controls. This is existence, not one global deployment parameter.

On all 24 controls, A004 preserved 6, the six new P0 points preserved a union of 15, and their combined observed union preserved `16/24`. The same eta need not rescue all targets and preserve all controls. No deployment eta was selected.

## 12. Cross-state parameter reuse

Cross-state reuse is precisely what triggered the H0 stop. Individual new P0 points rescued 0–10 targets; their complementary union rescued 32, and union with A004 rescued 37. No per-case range expansion or parameter tuning was performed.

Because the experiment stopped at H0, cross-state reuse of Agent6/P2/P3 discovery points was not expanded to all 85 episodes. Pilot Stage-A samples were shared across all 18 pilot episodes, so the completed pilot comparison remains internally matched.

## 13. Capacity attribution

### Governing attribution

**H0 — SEARCH INSUFFICIENCY.** This is the only full-population conclusion supported by the completed protocol.

The requested structural label is therefore not scientifically identifiable yet. For bookkeeping only:

```text
CAPACITY_ATTRIBUTION = MIXED (PROVISIONAL; H0 STOP PREVENTS A FINAL STRUCTURAL LABEL)
```

The completed pilot evidence behind that provisional label is:

- H1/agent sharing: positive pilot signal (`+3/12` new cases, locally robust in two of three);
- H4/projection coupling: strong, independently measured compression, particularly Agent6 raw rank `6 -> 3`;
- H2/relation aggregation: improved local expert fit but no new pilot basin;
- H3/temporal rigidity: no pilot basin gain;
- missing basis directions: not established, because only 7 expert-recoverable offline states were available and P0 search was demonstrably incomplete.

It would be incorrect to return a definitive `AGENT-LIMITED`, `RELATION-LIMITED`, `TEMPORAL-LIMITED`, `PROJECTION-LIMITED`, or `BASIS-LIMITED` label from this phase.

## 14. Recommended minimal representation

```text
RECOMMENDED_MINIMAL_REPRESENTATION = P0-3D (unchanged, pending adequate search)
```

Agent6 is the smallest structural candidate with a positive pilot signal, but it should not replace P0 yet. The current data first require a sufficiently dense P0 map on the full frozen population. No `G_phi` target space should be trained from the old sparse 3D map.

## 15. Next scientific step

Run one denser, globally shared P0-3D search over the same frozen domain on all 61 safe-timeout targets and 24 controls, using the existing 256-point Sobol design and the same fixed local-robustness rule. Confirm representative successes across MACFlow seeds. Do not expand the domain, change the bases, or add representation dimensions during that step.

Only after the full P0 map is frozen should the Agent6 pilot signal be revisited against the genuinely residual P0-negative population.

## Reproducibility and hygiene

Primary artifacts:

- `PREREGISTRATION.json` and `H0_INHERITANCE_AMENDMENT.json`;
- `p0_dense_search_audit.json` and `H0_STOP_AUDIT.json`;
- `offline_expert_fit_and_projection_audit.json` and `offline_state_fit.csv`;
- `pilot_capacity_summary.json`;
- immutable job manifests, Sobol designs, raw JSONL outcomes, and logs;
- figures under `figures/`;
- excluded partial structural outputs under `incomplete_after_h0_stop/`.

The one-state smoke artifact and Python bytecode caches were removed. Diagnostic representations remain isolated under this study directory. Canonical P0 code was not modified.

Post-experiment frozen SHA-256 values match pre-registration:

| Object | SHA-256 |
|---|---|
| S-XL-128 checkpoint | `6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd` |
| dataset manifest | `771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56` |
| MACFlow source | `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8` |
| environment | `3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc` |
| hard projection | `847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79` |
| canonical 3D corrector | `48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40` |

Regression command:

```text
C1_PYTHON=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python bash scripts/test.sh
```

Result: `58 + 2 + 19 = 79` tests passed. JAX printed the workstation CUDA warning and fell back to CPU; no test failed. Toy Give-Way, S-XL-128 MACFlow, the environment, hard-safety projection, and canonical P0 remained unchanged.
