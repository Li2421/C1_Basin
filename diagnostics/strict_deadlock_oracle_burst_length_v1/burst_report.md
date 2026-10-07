# Strict-deadlock fixed-eta burst-length audit

## Result

**LONG_BURST_REQUIRED**

This is an oracle-only temporal-persistence ablation. Every state uses its frozen minimum-J_def eta, frozen H=8 global trigger phase, matched Flow streams, and unchanged controller stack. No eta search, learned G_phi, gate, or model training occurred.

## Integrity and semantics

- States: 17 (11 historical, 6 fresh unseen); starting state and eta hashes match the capacity/cadence audits.
- H1 robust rows reused: 1088/1088 successful.
- L1 rows reused: 1088, exactly the prior one-step H8 condition.
- New conditions: L2, L4, L6, L8; 64 matched seeds per state plus exact Flow.
- A burst is created only by a trigger observed at an absolute global timestep divisible by 8. Query start never creates an artificial trigger.
- Special old_r106 phase: query step 279 is inactive; first eligible trigger is global step 280. L8 is therefore not silently treated as H1 at its first continuation step.

## Aggregate robust result

| condition | success/1088 | rate | deadlock | timeout | B63 states | mean J_def | active fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 | 0.406249 | 1.0000 |
| L1 | 82/1088 | 0.0754 | 659 | 347 | 0/17 | 0.031104 | 0.1260 |
| L2 | 34/1088 | 0.0312 | 699 | 355 | 0/17 | 0.051694 | 0.2520 |
| L4 | 165/1088 | 0.1517 | 189 | 734 | 1/17 | 0.111263 | 0.5014 |
| L6 | 291/1088 | 0.2675 | 0 | 797 | 0/17 | 0.223548 | 0.7507 |
| L8 | 1088/1088 | 1.0000 | 0 | 0 | 17/17 | 0.406257 | 0.9998 |

The shortest tested condition restoring near-dense capacity (at least 16/17 B63 and aggregate success at least 0.99) is **L8**. Its mean-J_def saving relative to dense H1 is -0.00%.

## Per-state robust persistence

| case | cohort | H1 | L1 | L2 | L4 | L6 | L8 | minimum B63 L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| old_r002 | historical | 64 | 1 | 0 | 0 | 29 | 64 | 8 |
| old_r018 | historical | 64 | 0 | 0 | 64 | 0 | 64 | 4 |
| old_r029 | historical | 64 | 2 | 0 | 0 | 22 | 64 | 8 |
| old_r052 | historical | 64 | 1 | 0 | 0 | 0 | 64 | 8 |
| old_r096 | historical | 64 | 2 | 0 | 0 | 25 | 64 | 8 |
| old_r106 | historical | 64 | 0 | 0 | 0 | 13 | 64 | 8 |
| old_r132 | historical | 64 | 2 | 0 | 0 | 0 | 64 | 8 |
| old_r156 | historical | 64 | 4 | 0 | 1 | 0 | 64 | 8 |
| old_r175 | historical | 64 | 2 | 0 | 62 | 6 | 64 | 8 |
| old_r182 | historical | 64 | 2 | 0 | 0 | 27 | 64 | 8 |
| old_r191 | historical | 64 | 0 | 0 | 0 | 13 | 64 | 8 |
| fresh_r011 | fresh_unseen | 64 | 1 | 0 | 0 | 15 | 64 | 8 |
| fresh_r058 | fresh_unseen | 64 | 28 | 29 | 34 | 58 | 64 | 8 |
| fresh_r062 | fresh_unseen | 64 | 0 | 0 | 0 | 16 | 64 | 8 |
| fresh_r114 | fresh_unseen | 64 | 0 | 0 | 0 | 14 | 64 | 8 |
| fresh_r133 | fresh_unseen | 64 | 34 | 2 | 0 | 32 | 64 | 8 |
| fresh_r139 | fresh_unseen | 64 | 3 | 3 | 4 | 21 | 64 | 8 |

Minimum-B63 distribution: {'8': 16, '4': 1}.

## Monotonicity

Exact nondecreasing state curves: 6/17. Nonmonotonic states: old_r002, old_r018, old_r029, old_r052, old_r096, old_r132, old_r156, old_r175, old_r182, fresh_r011, fresh_r133. Raw state counts and any downward transitions are recorded in `monotonicity_audit.csv`.

## Historical versus fresh-unseen cohorts

| cohort | condition | success | rate | B63 states |
|---|---|---:|---:|---:|
| historical | H1 | 704/704 | 1.0000 | 11/11 |
| historical | L1 | 16/704 | 0.0227 | 0/11 |
| historical | L2 | 0/704 | 0.0000 | 0/11 |
| historical | L4 | 127/704 | 0.1804 | 1/11 |
| historical | L6 | 135/704 | 0.1918 | 0/11 |
| historical | L8 | 704/704 | 1.0000 | 11/11 |
| fresh_unseen | H1 | 384/384 | 1.0000 | 6/6 |
| fresh_unseen | L1 | 66/384 | 0.1719 | 0/6 |
| fresh_unseen | L2 | 34/384 | 0.0885 | 0/6 |
| fresh_unseen | L4 | 38/384 | 0.0990 | 0/6 |
| fresh_unseen | L6 | 156/384 | 0.4062 | 0/6 |
| fresh_unseen | L8 | 384/384 | 1.0000 | 6/6 |

## Special phase and temporal evidence

old_r106 begins at global step 279 (phase 7), waits until the trigger at step 280, and has minimum B63 burst length **8**. Representative exact-Flow transitions from a shorter failing burst to a longer successful burst are in `divergence_cases.csv`, including the first shorter-off/longer-active step, correction after projection, goal-error change, and monitor state.

## Interpretation

Evidence supports: **persistent recovery mode requiring an explicit exit condition**.

Smallest justified next experiment: Test one frozen persistent recovery-mode exit rule, with the existing eta and trigger unchanged, on these 17 states.

Success remains the hard constraint; J_def savings are interpreted only for conditions that restore robust recovery.

## Runtime and resources

- New rollouts: 4420; new physical steps: 2549795.
- Reused robust rollouts: 1088 H1 + 1088 L1.
- Maximum shard wall time: 1232.2 s.
- Allocation: 2 GPU shards, 6 CPU cores requested, 48 GB memory requested.
