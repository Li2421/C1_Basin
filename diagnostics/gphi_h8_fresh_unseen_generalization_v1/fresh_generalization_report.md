# Frozen H=8 fresh-unseen WIDE generalization

**Classification: H8_FRESH_GENERALIZATION_CONFIRMED.** This is the single pre-frozen 200-episode test; no alternative cadence or checkpoint was evaluated.

## Frozen test integrity

- Generator: `short_scene_wide_initial_state_robustness_v1` with seed `2026092501`; exact authoritative replay was bitwise verified.
- Manifest SHA256: `68eecae6d04b00b0e4db7447a8e05d2443d0c886e67d14526b2eae8952841d45`; frozen at `2026-09-25T02:32:58.300903+00:00` with zero rollout records present.
- Overlap: 0 exact initial-state matches and 0 seed/stream collisions against the audited prior assets.
- Checkpoint: `/home/zhihan/research/Basin_C1/diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz` (`c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e`).
- Fixed endpoint: 850 steps at 0.05 s; matched fresh Flow root 2026092502; Safety and one-step-only H=8 only.

## Primary result

| Controller | Success | Strict deadlock | Timeout | Collision | Q (95% Wilson CI) |
|---|---:|---:|---:|---:|---:|
| Safety | 135/200 | 6 | 59 | 0 | 0.6750 [0.6073, 0.7361] |
| H=8 | 188/200 | 0 | 12 | 0 | 0.9400 [0.8981, 0.9653] |

Paired cells: BOTH_SUCCESS=129, RESCUE=59, BREAK=6, BOTH_FAIL=6. 
Delta Q=+0.2650 (paired bootstrap 95% CI [+0.1950, +0.3350]); exact one-sided McNemar improvement p=2.48233e-12.

## Failure-type rescue

- Safety timeout: 59/59 rescued (1.0).
- Safety strict deadlock: 0/6 rescued (0.0).
- Timeout-vs-deadlock rescue asymmetry reproduced descriptively: **yes**.

## Deformation and safety

H=8 J_def mean=0.024892, median=0.019369, P95=0.052970, max=0.061000.
Hard safety: **INTACT**.
