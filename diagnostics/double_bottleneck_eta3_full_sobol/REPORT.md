# Double-Bottleneck P0-3D full Sobol evaluation

Date: 2026-09-26  
Protocol: frozen unattended v2

This is the final report for the deterministic, non-adaptive evaluation of the original fixed episode-level P0-3D eta representation. It supersedes the scientific summary in `REPORT_INITIAL_PROTOCOL.md`, which is retained unchanged as an audit of the preceding robustness protocol.

No training, adaptive search, representation expansion, eta-domain change, safety change, horizon change, environment change, or success-definition change occurred.

## Frozen controller and identity

Every rollout used:

```text
u_flow = MACFlow(x)
u_safe = Pi_U(x)(u_flow)
g_i    = eta1 B_goal,i + eta2 u_safe,i + eta3 B_rel,i
u_exec = Pi_U(x)(u_safe + g)
```

where:

```text
B_goal,i = radial_bound_0.5(goal_i - position_i)
B_rel,i  = (1/(N-1)) sum_(j != i) radial_bound_0.5(position_i-position_j)
```

Eta was fixed for the full episode. The unchanged domain was:

```text
eta1 in [ 0.50, 1.25]
eta2 in [-0.50, 0.50]
eta3 in [ 0.00, 0.75]
```

The common design contained the existing 255 Sobol points plus the embedded old A004 point, for exactly 256 eta values.

Pre- and post-run SHA-256 checks matched for the S-XL-128 checkpoint, dataset manifest, MACFlow source, environment, hard-safety projection, canonical P0 implementation, eta design, and a deterministic Toy Give-Way source-tree hash. The diagnostic P0 wrapper and canonical corrector were also exactly equal in a separate 2,048-scalar numerical identity audit.

## A. Search completeness

The global design completed:

```text
85 episodes × 256 eta = 21,760 / 21,760 global rollouts
missing IDs:    0
duplicate IDs:  0
collisions:     0
```

The detailed rerun reproduced the previously frozen common-256 timeout and control membership matrices exactly.

Follow-up evaluation completed without early stopping:

| Evaluation | Rollouts |
|---|---:|
| Global common-256 | 21,760 |
| Fixed local robustness | 1,216 |
| Fixed eight-seed robustness | 304 |
| Eta-zero timing controls | 38 |
| Scientific total | **23,318** |

All output was written incrementally to deterministic shards. Raw shard paths, row counts, and SHA-256 values are indexed by `long_run_v2/global_rollouts_manifest.json`; `long_run_v2/run_manifest.json` records the completed state.

Global terminal outcomes were 89 successes, 1,110 runtime strict-deadlock terminations induced by tested eta values, and 20,561 timeouts. These strict-deadlock outcomes do not relabel the original 61 eta-zero targets, which remain frozen safe timeouts.

## B. Basin existence

A sampled P0 success basin was observed for:

```text
38 / 61 = 62.30% of safe-timeout targets
```

The basin-count distribution was fixed descriptively:

| Successful eta points | Episodes |
|---|---:|
| 0 | 23 |
| 1 | 23 |
| 2–4 | 15 |
| 5–15 | 0 |
| >15 | 0 |

Among the 38 positive episodes, `rho_i=|B_i|/256` had:

- median: `1/256 = 0.00390625` (0.390625%);
- mean: `0.00596217` (0.5962%);
- range: `1/256` to `4/256`.

Thus existence is a majority result, but sampled basin volume is small: 23 of 38 positive episodes have exactly one successful global point.

The complete episode-by-eta membership is in `long_run_v2/timeout_basin_matrix.npz` and `timeout_basin_matrix.json`.

## C. Comparison with the old search

| Search evidence | Basin observed |
|---|---:|
| Old 65-point search | 15/61 |
| Old A004 alone | 15/61 |
| Old A004 plus six newly discovered fixed points | 37/61 |
| Full common 256-point design | **38/61** |

The old 15/61 conclusion was primarily a search-resolution artifact. The previously identified seven-point union already captured 37 of the final 38 positive cases; the complete design added one more case. Therefore P0-3D cannot be rejected on basin existence alone, while the concentration in only seven rescuing global points remains important.

## D. Local robustness

For each positive episode, one representative eta was selected without trajectory inspection:

1. count successes among its eight nearest common-design neighbors in normalized eta coordinates;
2. maximize that count;
3. tie-break by minimum normalized distance to eta zero;
4. tie-break by smallest eta index.

Exactly 32 scrambled-Sobol perturbations were evaluated around each representative with fixed per-axis normalized radius 0.05 and clipping to the original domain. The center remained separately verified by the global matrix.

| Local category | Rule | Episodes |
|---|---|---:|
| Low | `Q_local < 0.25` | 18 |
| Medium | `0.25 <= Q_local < 0.75` | 19 |
| High | `Q_local >= 0.75` | 1 |

`Q_local` had median **0.25**, mean 0.2837, and range 0–0.75. Its interquartile range was 0.1016–0.4375. Local success is therefore not purely isolated, but robustness varies substantially and half of positive cases sit at or below the PASS boundary.

All episode-level values are in `long_run_v2/local_robustness.json`.

## E. MACFlow stochastic-seed robustness

Each fixed representative eta was rerun without reselection under exactly:

```text
[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]
```

| Seed category | Rule | Episodes |
|---|---|---:|
| Low | `Q_seed < 0.25` | 20 |
| Medium | `0.25 <= Q_seed < 0.75` | 17 |
| High | `Q_seed >= 0.75` | 1 |

`Q_seed` had median **0.125**, mean 0.1612, and range 0–0.75. Twenty-eight of 38 episodes were at or below 0.25; only one reached the High category. No alternative eta was selected after these results.

This is the decisive limitation under the fixed status rule. Episode-level terminal reasons and values are in `long_run_v2/seed_robustness.json`.

## F. Cross-state overlap

Pairwise Jaccard overlap was computed on the common 256-bit membership vectors for all 703 pairs of basin-positive episodes:

| Statistic | Value |
|---|---:|
| Mean Jaccard | 0.1752 |
| Median Jaccard | **0.0** |
| Zero-overlap fraction | 65.58% |
| Fraction above 0.1 | 34.42% |
| Fraction above 0.25 | 28.59% |

There is meaningful reuse for a subset, but most episode pairs share no successful sampled eta. The full matrix is in `long_run_v2/jaccard_overlap.json`. No semantic-mode interpretation is assigned to numerical overlap.

## G. Global eta reuse

The complete 256-row rescue/preservation table is saved as `long_run_v2/rescue_preserve_table.json` and `.csv`.

Only seven eta points rescued at least one timeout episode; seven preserved at least one control, and six did both. The most reusable point was the embedded old A004:

```text
eta index: 255
eta: [0.5364547465, 0.1891057156, 0.0249288732]
timeout rescue:  15/61
control preserve: 6/24
```

The largest control-preservation count was 8/24, achieved by A113 while rescuing 9/61. No scalar rescue/preservation score was formed and no global deployment eta was selected.

## H. Baseline control preservation

Every one of the 24 frozen success controls was evaluated at all 256 eta points. The full Boolean matrix is in `long_run_v2/control_preservation_matrix.npz`.

The best single eta preserved only 8/24 controls; the most rescuing eta preserved 6/24. Nonzero eta therefore often turns an eta-zero success into a safe liveness failure. This is a task-success tradeoff, not a safety failure: all global control rollouts remained collision-free.

## I. Projection coupling

Step-weighted correction statistics were:

| Group | Raw norm | Executed norm | Removal magnitude | Removal ratio | Steps with >50% removed |
|---|---:|---:|---:|---:|---:|
| Successful eta rollouts | 0.5685 m/s | 0.1694 m/s | 0.4897 m/s | 0.8541 | 97.42% |
| Failed eta rollouts | 0.7040 m/s | 0.3700 m/s | 0.5203 m/s | 0.7169 | 87.57% |
| Eta zero | 0 | approximately (1.36	imes10^{-11}) | approximately (1.36	imes10^{-11}) | undefined | 0% |

The eta-zero residual is numerical solver tolerance. High projection activity is descriptive: successful P0 behavior uses a small feasible residual of the raw correction, but these data alone do not establish that projection is harmful. Full statistics are in `long_run_v2/projection_statistics.json`.

## J. Behavioral timing mechanism

For each of the 38 rescued episodes, the representative eta was compared with a separately rerun eta-zero hard-safety trajectory under the original episode seed. Bottleneck clearance is the latest directed crossing among all four agents at the same fixed resource planes used in prior diagnostics. Deltas below are `eta_rep - eta0`; negative values mean earlier or less.

| Metric | Median delta | 10th–90th percentile |
|---|---:|---:|
| First-bottleneck clearance | -2.025 s | -4.180 to +0.515 s |
| Second-bottleneck clearance | -2.475 s | -4.580 to +0.115 s |
| Total waiting | -22.55 agent-s | -25.94 to -11.94 agent-s |
| Final goal entry | -3.85 s | -6.05 to -0.34 s |
| Completion vs 42.5 s horizon | -3.85 s | -6.42 to -0.34 s |

Eta-zero final goal entry was absent in 37/38 cases and was explicitly right-censored at 42.5 s for the aggregate final-entry delta. Raw null/censored flags are retained per episode.

The general mechanism is reduced waiting, earlier resource clearance, and completion acceleration. These are timeout/liveness recoveries, not evidence that the original cases were validated deadlocks. Full data are in `long_run_v2/timing_analysis.json`.

## K. Deterministic representative plots

The fixed rules selected:

- R1, maximum `rho`: `fresh_untouched_test|051`, 4/256;
- R2, closest to median positive `rho`: `existing_untouched_test|010`, 1/256;
- R3, minimum positive `rho`: `existing_untouched_test|010`, 1/256;
- R4, lowest-ID basin-negative case: `existing_untouched_test|002`.

R2 and R3 coincide because the median and minimum positive basin fractions are both 1/256 and the specified episode-ID tie-break selects the same case. This duplication is protocol-determined, not manual selection.

Figures:

- [R1–R4 3D eta scatter](long_run_v2/figures/R1_R4_eta_3d_scatter.svg)
- [R1–R4 coordinate projections](long_run_v2/figures/R1_R4_eta_2d_projections.svg)
- [R1–R3 local robustness](long_run_v2/figures/R1_R3_local_robustness.svg)

Eta-zero and representative trajectory SVG/NPZ files are under `long_run_v2/representatives/`; R4 correctly has no representative-eta trajectory.

## Reproducibility and regression

The run is fully resumable through immutable job manifests and append-only per-shard JSONL files. Required artifacts are under `long_run_v2/`:

- `PREREGISTRATION.json`, `run_manifest.json`;
- `global_rollouts_manifest.json` and raw global shards;
- `timeout_basin_matrix.{json,npz}`;
- `control_preservation_matrix.npz`;
- `local_robustness.json`, `seed_robustness.json`;
- `jaccard_overlap.json`;
- `projection_statistics.json`;
- `timing_analysis.json`;
- `rescue_preserve_table.{json,csv}`;
- representative traces and figures.

The frozen regression suite passed **79/79** tests. Toy Give-Way’s deterministic source-tree hash and all canonical hashes match pre-registration.

## Fixed factual conclusion

1. `P0 basin observed in 38/61 timeout cases.`
2. `Median positive basin fraction = 0.00390625 (1/256).`
3. `Median local robustness = 0.25.`
4. `Median seed robustness = 0.125.`
5. `Median pairwise basin overlap = 0.0.`
6. `Best shared eta rescues 15/61 and preserves 6/24 controls.`
7. `All tested rollouts collision-free: YES.`

## Status

The fixed PASS test requires all three:

- `N_exist >= 31`: met (38);
- median `Q_local >= 0.25`: met exactly (0.25);
- median `Q_seed >= 0.50`: **not met** (0.125).

**REVISE**

P0-3D has meaningful sampled existence and reaches the local-robustness boundary, but the deterministic representative selection is not robust to MACFlow sampling stochasticity. This status is a protocol summary only. It does not authorize `G_phi` training or any representation expansion.

