# Double-Bottleneck P0-3D common-256 basin audit — initial robustness protocol

Date: 2026-09-26

This report closes the first common-256 protocol. It uses the frozen S-XL-128 MACFlow, four-agent environment, two hard-safety projections, canonical fixed episode-level P0-3D correction, 850-step horizon, 61 frozen safe-timeout targets, and 24 frozen success controls. No model was trained and no representation, eta domain, safety constraint, environment rule, horizon, or success definition changed.

## 1. Frozen protocol

The executed correction was

```text
B_goal,i = radial_bound_0.5(goal_i - position_i)
B_rel,i  = (1/(N-1)) sum_(j != i) radial_bound_0.5(position_i-position_j)
g_i      = eta1 B_goal,i + eta2 u_safe,i + eta3 B_rel,i
u_exec   = Pi_U(x)(u_safe + g)
```

with one eta fixed for the episode. The common domain was `eta1 in [0.5,1.25]`, `eta2 in [-0.5,0.5]`, and `eta3 in [0,0.75]`. Every one of the 85 episodes received the same 255-point Sobol design plus the embedded old A004 point, for 256 total points.

The diagnostic P0 wrapper and canonical `DiagnosticCorrector` were exactly equal on 2,048 scalar comparisons; the Agent6 equal-alpha embedding used only as an exact cache source was also exactly equal. Maximum absolute difference was zero.

## 2. Sparse search versus the full 256 points

| Search evidence | Timeout episodes covered |
|---|---:|
| Old 65-point study / old A004 | 15/61 |
| Six subsequently discovered fixed P0 points | 32/61 |
| Seven-point observed union | 37/61 |
| Full common 256-point design | **38/61** |

Thus sparse search caused a large undercount: the observed existence result moved from 15 to 38 episodes. The 256-point completion added only one episode beyond the already observed seven-point union, which also shows that the successful sampled volume remains concentrated.

All `21,760/21,760` global episode/eta pairs are present with unique IDs. There were zero wall or agent collisions.

## 3. Full timeout basin existence and fractions

A non-empty sampled P0 basin was observed for

```text
38 / 61 = 62.30%
```

There were 23 empty episodes. Among positive episodes, the successful-point count had median 1, mean 1.526, and range 1–4. The positive basin fraction `rho_B=|B|/256` had:

- median `1/256 = 0.390625%`;
- mean `0.5962%`;
- range `0.390625%–1.5625%`.

Counts by number of successful points were: 23 episodes with 0, 23 with 1, 11 with 2, 3 with 3, and 1 with 4. At normalized eta distance threshold 0.20, positive episodes had a median one sampled component; this is only a sampling-resolution geometry proxy.

Existence split was 22/32 on the existing test and 16/29 on the fresh test. It appeared in both baseline directions: 16/29 LTR-first and 22/32 RTL-first.

![Basin fractions](figures/basin_fraction_distribution.svg)

![Representative eta projections](figures/representative_eta_projections.svg)

![Representative 3D scatter](figures/representative_eta3_scatter.svg)

## 4. Local eta robustness

The first protocol selected up to three deduplicated representatives per positive episode: minimum eta norm, maximum common-design support within normalized radius 0.20, and maximum cross-state coverage. Each received the same 38-point normalized-radius-`1/32` neighborhood.

Across 48 selected centers:

- broad (`Q>=0.75`): 6;
- narrow (`0.10<=Q<0.75`): 40;
- isolated (`Q<0.10`): 2;
- center-level median `Q_local=0.4211`.

Taking the best registered representative per episode, median `Q_local=0.4342`; 18/38 episodes had a representative with `Q_local>=0.50`. Local geometry is real but predominantly narrow.

## 5. MACFlow-seed robustness

The same 48 centers were each rerun under eight pre-registered independent MACFlow seeds.

- seed-robust centers (`Q>=0.75`): 0;
- seed-sensitive (`0.25<=Q<0.75`): 17;
- lucky (`Q<0.25`): 31;
- center-level median `Q_seed=0.125`;
- best-per-episode median `Q_seed=0.125`, maximum 0.50.

No episode had a seed-robust representative under this selection protocol. This is the principal unresolved weakness.

![Robustness categories](figures/robustness_classifications.svg)

## 6. Cross-state overlap and reuse

Among the 38 positive episodes, the 703 off-diagonal Jaccard values had mean 0.1752 and median 0. The sampled-overlap graph at Jaccard 0.25 contained one component of 37 episodes and one singleton. Same-order-signature pairs had a higher mean overlap (0.2325) than cross-signature pairs (0.1589), but the median was zero for both.

The most reusable point remained the embedded old A004:

```text
eta = [0.5364547465, 0.1891057156, 0.0249288732]
rescues 15/61 timeout targets
preserves 6/24 success controls
```

The point preserving the most controls was A113: 8/24 controls while rescuing 9/61 targets. No deployment eta was selected.

![Rescue/preservation tradeoff](figures/eta_rescue_preservation_tradeoff.svg)

![Cross-state overlap](figures/cross_state_overlap.svg)

## 7. Baseline-success preservation

Across each control’s 256 eta values, the median preservation fraction was `1/256`; the maximum for any control was `4/256`, and some controls had no nonzero preserving point. Across all global runs there were 89 successes, 1,110 runtime strict-deadlock terminations, 20,561 timeouts, and zero collisions. These eta-induced strict-deadlock outcomes do not relabel the original target population.

## 8. Projection coupling

Step-weighted statistics were:

| Group | Raw correction | Executed correction | Removed by second projection | Mostly rewritten |
|---|---:|---:|---:|---:|
| Successful eta rollouts | 0.5685 m/s | 0.1694 m/s | 0.4897 m/s | 83.09% |
| Failed eta rollouts | 0.7040 m/s | 0.3700 m/s | 0.5203 m/s | 50.72% |
| eta=0 | 0 | 0 | 0 | n/a |

The second-projection/raw ratio of means was 86.15% on successful rollouts and 73.90% on failed rollouts. Projection remains integral to the executable behavior, but this descriptive activity alone is not treated as evidence that safety is harmful.

## 9. Behavioral mechanism

The deterministic high-, median-, and low-`rho_B` trace pairs all converted eta-zero timeout into full success. Relative to eta zero:

- last second-bottleneck clearance advanced by 4.1–5.0 s;
- mean waiting fell by 3.68–6.15 agent-seconds per agent;
- completion occurred in 36.05–39.90 s rather than timing out at 42.5 s;
- chamber transit changed only modestly.

The observed mechanism remains timeout recovery through reduced waiting and completion acceleration, not validated deadlock escape. Paired NPZ traces and SVG trajectories are under `representatives/`.

## 10. P0-3D viability assessment

The full search overturns the old capacity rejection: observed basin existence is now a majority, spans both directions and all regimes, and local neighborhoods are usually non-isolated. However, common-design basin fractions are tiny, median cross-state overlap is zero, and representative success is highly sensitive to MACFlow sampling seeds.

**REVISE — P0-3D is viable, but stochastic robustness and sampled-basin sparsity remain limiting.**

This does not justify representation expansion and does not authorize `G_phi` training. A stricter fixed representative-selection and seed protocol is required before judging whether a state-to-eta learning target is well-posed.

## 11. Reproducibility and hygiene

Primary machine-readable artifacts include `eta_points.json`, `timeout_basin_membership.json`, `basin_matrices.npz`, `control_preservation_matrix.json`, `per_eta_coverage.json/csv`, `local_robustness_results.json`, `stochastic_seed_robustness_results.json`, `cross_state_overlap.json`, `projection_coupling_statistics.json`, raw resumable JSONL shards, and their SHA-256 manifests.

All six frozen hashes match pre-registration. The full regression suite passed 79/79 tests (58 shared/single-integrator, 2 Toy, 19 Double-Bottleneck). Toy Give-Way and all canonical frozen components remain unchanged.

