# ORTHOFLOW3_FIXED_ETA_STATE_TO_Q_IDENTIFIABILITY_V2

## Decision

- **Toy: `STATE_TO_Q_STRONGLY_LEARNABLE`**
- **Double-Bottleneck: `DATA_NOT_IDENTIFIABLE`**
- **NEW ROLLOUT = 0**

The experiment used exact fixed eta heads and source-group-isolated unseen states. Eta coordinates were never supplied to the model, so none of the results depend on eta interpolation or extrapolation.

## Data and leakage audit

The current authoritative global rollout database was queried directly. Compatible exact seed records were aggregated once per canonical state×eta pair; aggregate-only records would only be used when seed-exact evidence was absent. No aggregate-only record was needed for the selected inventories.

| Scenario | State split | Exact eta | Eta on >=10 states | Strict repeated coverage* | Failure↔B15 state-dependent eta |
|---|---:|---:|---:|---:|---:|
| Toy | 128 / 32 / 32 | 4,312 | 131 | 55 | 27 |
| DB | 64 / 16 / 16 | 350 | 25 | 12 | 0 |

\* At least 10 states overall, with at least 6 TRAIN, 2 VAL, and 2 TEST states. TRAIN/VAL/TEST source-group overlap was exactly zero in both scenarios.

Probe selection used TRAIN outcomes only. VAL/TEST metadata was used only to require that a probe had records in those splits. The Toy panel contained all 27 strict state-dependent probes plus four high/low-prevalence controls. The DB panel contained all 12 dense probes.

## Toy fixed-eta result

The table below is restricted to the 27 state-dependent/intermediate eta probes (504 unseen-state TEST pairs).

| Predictor | NLL | MAE | Q Spearman | B15 accuracy | B15 AUROC |
|---|---:|---:|---:|---:|---:|
| Per-eta constant | 0.5852 | 0.3575 | 0.2458 | 50.0% | 0.6373 |
| 3-NN in h | 0.3769 | 0.1289 | 0.8084 | 85.3% | 0.9527 |
| Fixed-head MLP | **0.2671** | **0.0801** | **0.8636** | **90.5%** | **0.9753** |
| State-shuffled MLP | 0.6582 | 0.3826 | 0.0945 | 49.4% | 0.5483 |

Relative to the per-eta constant, the normal MLP improved NLL by **0.3182** (54.4%) and MAE by **0.2774** (77.6%). It improved MAE on all 27 probes and NLL on 24/27. The median per-probe improvements were 0.2826 MAE and 0.3322 NLL.

The 3-NN result independently establishes that local information is present in the frozen h representation: it improved NLL by 0.2083 and MAE by 0.2286. Destroying the h↔outcome correspondence removed the MLP advantage.

For 2,216 TEST failure-versus-robust state pairs evaluated at the same exact eta, ordering accuracy was:

- constant: 50.0%;
- 3-NN: 96.7%;
- MLP: **97.9%**;
- shuffled MLP: 35.9%.

Thus, for the same eta, Toy h reliably distinguishes states where that eta fails from states where it is robust.

## Fixed-panel ranking

Ranking was evaluated on the state-dependent probes only, excluding universal/high-prevalence control eta.

| Predictor | Selected true Q | Oracle Q | Regret | B15 selection | Top-3 B15 hit |
|---|---:|---:|---:|---:|---:|
| Constant | 0.9365 | 1.0000 | 0.0635 | 90.6% | 93.8% |
| 3-NN | 0.9688 | 1.0000 | 0.0312 | 96.9% | 100% |
| Fixed-head MLP | **1.0000** | 1.0000 | **0.0000** | **100%** | **100%** |
| Shuffled MLP | 0.9351 | 1.0000 | 0.0649 | 90.6% | 96.9% |

On the exact old 12-mode Toy codebook, the MLP also selected a B15 mode on 32/32 TEST states with zero empirical regret. This sanity result is not used to establish state dependence because that codebook includes broadly successful modes.

## Double-Bottleneck qualification

DB has 12 dense exact eta probes across all 64/16/16 states, but **none** contains both clear failures and B15 robust states in TRAIN. Ten probes are essentially robust controls. Two probes show graded Q variation below the robust threshold, but neither provides a failure↔B15 transition.

On those two graded-Q probes, the MLP does show a real regression signal:

| Predictor | NLL | MAE | Q Spearman |
|---|---:|---:|---:|
| Constant | 0.6530 | 0.1588 | 0.4370 |
| 3-NN | 0.6127 | 0.0889 | 0.7336 |
| MLP | **0.6086** | **0.0864** | **0.8134** |
| Shuffled MLP | 0.7009 | 0.1707 | 0.3207 |

This is evidence that DB h contains some graded-Q information. It is **not** enough to decide whether h predicts robust feasibility, because there are zero valid failure-versus-B15 state pairs and no B15 candidate-selection test on those probes. The correct result is therefore:

`DB_FIXED_ETA_NOT_IDENTIFIABLE_FROM_CURRENT_DATA`

The missing evidence is a repeated exact-eta cross-matrix in which the same eta is evaluated on source-diverse DB states and exhibits both clear failure and B15 robust outcomes. No rollout was added to obtain it.

## Explanation of the earlier continuous-critic failure

For Toy, the conclusion is sharp. When eta identity is fixed and repeated across states, the frozen h representation predicts Q extremely well; when h is shuffled, performance collapses. This rules out “h contains no feasibility information” as the main explanation for the earlier continuous critic's poor Toy unseen-state/unseen-eta result.

The dominant remaining explanation is the earlier dataset's sparse state×eta cross-structure and the resulting unseen-eta compositional-generalization problem: most eta were observed on only one state, so the continuous critic had to infer both a state effect and an eta interaction without a repeated cross-matrix. The current experiment removes that confound and succeeds.

For DB, that attribution remains underresolved because the database does not contain state-dependent robust-transition probes.

## Reproducibility

All detailed per-eta metrics, panel definitions, source splits, controls, ordering pairs, and rankings are stored alongside this report. The canonical pair snapshot is a read-only aggregation of the current global rollout database. No controller execution, GPU rollout, eta search, generator training, or unseen-eta test occurred.

