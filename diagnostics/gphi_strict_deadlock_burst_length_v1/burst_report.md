# Frozen G_phi strict-deadlock burst-length audit

## Result

**GPHI_BURST_NONMONOTONIC**

The exact original checkpoint `c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e` was used. No training, eta search, online oracle, or gate was used. G_phi was recomputed from the current 214-D feature on every active burst step; no prediction was held. The frozen source eta was evaluated only as a counterfactual teacher diagnostic on each G_phi-visited state.

## Aggregate result

| condition | success | rate | deadlock | timeout | B63 states | exact success | mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 144/1088 | 0.1324 | 1 | 943 | 1/17 | 0/17 | 0.032146 |
| L2 | 111/1088 | 0.1020 | 0 | 977 | 0/17 | 0/17 | 0.059431 |
| L4 | 65/1088 | 0.0597 | 0 | 1023 | 0/17 | 0/17 | 0.141970 |
| L6 | 31/1088 | 0.0285 | 0 | 1057 | 0/17 | 0/17 | 0.113631 |
| L8 | 45/1088 | 0.0414 | 66 | 977 | 0/17 | 0/17 | 0.280722 |
| H1 | 45/1088 | 0.0414 | 66 | 977 | 0/17 | 0/17 | 0.280417 |

## Per-state result and oracle alignment

| case | L1 | L2 | L4 | L6 | L8 | H1 | G_phi min B63 | oracle min B63 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old_r002 | 1 | 1 | 0 | 2 | 0 | 0 | NONE | 8 |
| old_r018 | 0 | 0 | 0 | 1 | 0 | 0 | NONE | 4 |
| old_r029 | 7 | 1 | 0 | 2 | 0 | 0 | NONE | 8 |
| old_r052 | 1 | 0 | 0 | 0 | 0 | 0 | NONE | 8 |
| old_r096 | 4 | 1 | 0 | 0 | 0 | 0 | NONE | 8 |
| old_r106 | 0 | 0 | 0 | 2 | 13 | 13 | NONE | 8 |
| old_r132 | 8 | 2 | 0 | 0 | 0 | 0 | NONE | 8 |
| old_r156 | 19 | 8 | 1 | 1 | 0 | 0 | NONE | 8 |
| old_r175 | 3 | 3 | 3 | 1 | 0 | 0 | NONE | 8 |
| old_r182 | 3 | 1 | 0 | 0 | 0 | 0 | NONE | 8 |
| old_r191 | 0 | 0 | 0 | 1 | 0 | 0 | NONE | 8 |
| fresh_r011 | 1 | 0 | 0 | 1 | 0 | 0 | NONE | 8 |
| fresh_r058 | 29 | 29 | 27 | 8 | 2 | 2 | NONE | 8 |
| fresh_r062 | 2 | 1 | 0 | 0 | 0 | 0 | NONE | 8 |
| fresh_r114 | 0 | 0 | 0 | 1 | 0 | 0 | NONE | 8 |
| fresh_r133 | 64 | 61 | 32 | 11 | 30 | 30 | 1 | 8 |
| fresh_r139 | 2 | 3 | 2 | 0 | 0 | 0 | NONE | 8 |

G_phi minimum-B63 distribution: {'NONE': 16, '1': 1}. Oracle minimum-B63 distribution: {'8': 16, '4': 1}.

## Monotonicity and teacher drift

Nondecreasing learned state curves: 1/17. Nonmonotonic states: old_r002, old_r018, old_r029, old_r052, old_r096, old_r132, old_r156, old_r175, old_r182, old_r191, fresh_r011, fresh_r058, fresh_r062, fresh_r114, fresh_r133, fresh_r139.

Across per-state/condition first-burst summaries, mean executed correction error at burst step 1 is 0.268412 m/s; for steps 2+ it is 0.216776 m/s. Detailed state/condition/step metrics are in `first_burst_prediction_error.csv`, and every robust active-timestep prediction is preserved in compressed `active_logs/` with decile summaries in `prediction_drift_over_time.csv`.

Where the matched oracle exact Flow succeeds but G_phi fails, trajectory and correction divergence is localized in `state_divergence_cases.csv`. The 0.05 m/s "large mismatch" marker is diagnostic reporting only and never affects control.

## Interpretation

Best learned burst by aggregate strict-deadlock success: **L1** (0.1324). Supported next architecture: **G_phi itself still requires closed-loop supervision before choosing a temporal architecture**.

Smallest justified next experiment: Collect oracle labels only along frozen G_phi burst trajectories at the first correction-mismatch states, then retrain once with no cadence change.

## Runtime/resources

- New retained rollouts: 5720; retained physical steps: 4595919.
- Logged active prediction timesteps: 2484544.
- Maximum shard wall time: 2136.6 s.
- Allocation: 2 GPU shards, 6 CPU cores requested, 48 GB memory requested.
