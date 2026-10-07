# Four Way Intersection — formal hard safety and OrthoFlow3

**Final classification: `3D_ROBUST_SHARED`.** No MACFlow retraining, `G_phi`, eta-dimension expansion, basis redesign, eta-domain tuning, or scenario-specific basis was used.

## 1. Frozen Stage-I reference

The frozen checkpoint is `/home/zhihan/research/Basin_C1/diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_source_balanced_macflow/best.pkl` (SHA-256 `643270b8376bcd2a77b31bf849c5f259814f1ea015b0d5fd131c11ca581cce1e`). The environment fingerprint is `73910125d272d2a45793581f11bbd503cc4262f1d6651aaa44bbbb854db78849`, horizon is 1100 steps at dt=0.05, and the formal comparison uses the same 60 frozen test states and evaluation seed 8123 as the accepted Stage-I result.

## 2. Formal hard-safety result

| Controller | Success | Collision | Timeout / safe liveness failure | Mean steps |
|---|---:|---:|---:|---:|
| No safety | 29/60 | 6/60 | 25/60 | 905.20 |
| Hard safety | 32/60 | 0/60 | 28/60 | 970.12 |

Hard safety minimum recorded wall/obstacle and inter-agent clearances were 0.184548 and 0.152567. Projection was active on a mean 3.20% of steps; mean correction norm was 0.001796, with mean correction/Flow norm ratio 0.002248. This strongly suppresses collision without rewriting the policy on essentially every timestep.

## 3. No-safety → safety conversion matrix

| Matched conversion | Episodes |
|---|---:|
| `collision → success` | 2 |
| `collision → timeout` | 4 |
| `success → success` | 21 |
| `success → timeout` | 8 |
| `timeout → success` | 9 |
| `timeout → timeout` | 16 |

## 4. Frozen safe-failure population

The target set was frozen before eta search and contains **28 `SAFE_LIVENESS_FAILURE` states**: collision-free, unsuccessful, and terminated by timeout. The environments expose no independent runtime strict-deadlock detector, so none is relabelled as deadlock. State IDs are preserved in `formal_safety_summary.json`.

## 5. OrthoFlow3 definition and frozen eta domain

The controller is exactly `u_flow → Pi_U → u_safe + eta_g B_goal + eta_perp B_flow_perp + eta_rel B_rel → Pi_U`, with `B_flow_perp` the goal-orthogonal residual scaled by the frozen factor 3.303687238760696. The common domain is `[0.5,1.25] × [-0.5,0.5] × [0,0.75]`; both scenarios share the same 256-point scrambled Sobol designs. The second hard projection is mandatory.

## 6. Global 256-point search

The primary screen evaluated the common 256 points with logical seeds 0–3. It produced 1425 state/eta preliminary candidates at Q4≥3/4. Every 4/4 candidate and deterministically ranked 3/4 candidates were promoted according to the frozen rule; Q16 uses the fixed ≥15/16 robust threshold. Cache manifests and preflights are under `plans/` and `cache/`.

## 7. Secondary fixed search for negative states

After primary Q16 validation, 0 negative states received the independent second 256-point Sobol design over the unchanged domain. Secondary search added 0 robust-positive states. No range or representation change was made.

## 8. Robust eta existence

Robust eta exists for **28/28 = 100.00%** states. Primary search certified 28; secondary search certified 0. The remaining 0 states are classified `NO_ROBUST_ETA_OBSERVED`, not mathematical impossibility.

Exact seed-level stopping evaluated 24733 canonical seed outcomes versus 25248 without logical stopping, saving 2.04%. Of 1578 robust-decision tuples, 1114 were evaluated to all 16 seeds, 0 were rejected after the first four seeds, 7 were rejected later at the second observed failure, and 457 were accepted at 15 observed successes. Unrun seeds were neither fabricated nor stored as failures. These classifications are mathematically identical to full 15/16 membership evaluation; the saving is implementation efficiency, not a scientific result.

An additional 68 unique robust-membership tuples were not certified because one or more deterministic seed executions remained numerical-invalid. They are excluded from the early-stop saving denominator and are not imputed as failures.

## 9. Local robust-basin width

Each robust center received the common 32-point radius-0.05 local Sobol design, eight-seed screening, and fixed-index Q16 promotion for exactly 12 points. Median robust promoted fraction is **1.0000** (range 0.7500–1.0000).

## 10. Robust eta cross-state coverage

There are 5 numerically unique center values. Best robust state coverage is **1.0000** and median is **0.9286**. Counts at ≥10/25/50/75/100% are `{'0.1': 5, '0.25': 5, '0.5': 5, '0.75': 5, '1.0': 1}`. Every membership decision uses Q16≥15/16, never a single seed.

## 11. Basin overlap / state dependence

Median pairwise robust-membership Jaccard is 1.0000; 0.00% of state pairs have zero overlap over the tested center library. These sampled memberships support the stated classification but do not prove continuous basin topology.

## 12. Success-control preservation

Up to 24 uniformly selected hard-safety-success controls were evaluated with eight fixed seeds for every unique center. Median center preservation probability is 1.0000; the full center distribution is in `success_control_preservation.json`. There were 19/960 numerical-incomplete control evaluations; they are excluded from probability denominators, never imputed as failures. No center was selected or optimized using controls.

## 13. Behavioral mechanism

On diagnostic seed 16, 27/27 numerically valid selected-center pairs rescued their state (1/28 planned pairs were numerical-incomplete); 24/27 changed the realized crossing signature. Among rescued pairs, median completion-time change was -32.700s, median per-agent waiting change was -30.738s, and the median change in the explicitly labelled low-speed/outside-centre cyclic-yielding proxy was -5.250s. These are descriptive behavioral proxies, not a runtime deadlock declaration.

## 14. Final classification

`3D_ROBUST_SHARED`: robust existence is 100.00%, and best shared-center coverage is 100.00%. The interpretation follows the predeclared existence-versus-shared-coverage distinction. Numerical execution audit: 54949 exact controller records, 723 retained numerical-invalid records excluded from scientific outcomes, 0 eta collisions, and 0 database foreign-key violations.
