# Strict-deadlock supervision coverage retraining audit

## Result

**DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED**

The original dataset contained no exact match to any of the historical 11
earliest robust-onset states.  Adding exactly one standard 64-variant cohort
per historical state reduced mean target L2 from
0.268149 to
0.002952; on the untouched Fresh-6 target
diagnostic it fell from 0.269337 to
0.032305.

## Frozen checkpoint

- Seed 41, epoch 1177
- `/home/zhihan/research/Basin_C1/diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz`
- SHA256 `340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700`
- Selection used startup/warm validation only.

## Offline retention

| Test cohort | Old mean L2 | New mean L2 |
|---|---:|---:|
| Startup | 0.088858 | 0.080094 |
| Warm V3 | 0.012075 | 0.015854 |

## Closed loop, frozen H=8

- Historical strict-deadlock rescue: old 0/11, new 0/11.
- Fresh held-out strict-deadlock rescue: old 0/6, new 1/6.
- Development fresh-WIDE regression: old Q=0.940, new Q=0.880.
- Safety timeout rescue: 50/59.
- Breaks of Safety successes: 10.
- Collisions: 0.
- Mean J_def: 0.027750.

Hard safety passed across all
new rollouts.  The single smallest justified next experiment is: Without training or eta search, replay each frozen minimum-J_def robust eta on the 17 onset states using the deployment H=8 one-step cadence; compare with its known dense fixed-eta continuation.
