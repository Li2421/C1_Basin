# G_phi training dataset v2

## Result

**READY_FOR_PILOT_TRAINING**

- Selected exact augmented states: **246**; usable labeled states: **246**.
- Supervised samples: **15744** (64 matched Flow samples per usable state).
- Selected categories: {'NORMAL': 120, 'PRE_DEADLOCK': 60, 'RECOVERY': 66}; usable: {'PRE_DEADLOCK': 60, 'NORMAL': 120, 'RECOVERY': 66}.
- Zero/nonzero oracle states: **131 / 115**.
- B_63 empty states: **0**; multivalued quarantines: **0**.
- Label classes: {'LABEL_STABLE': 214, 'LABEL_MILDLY_AMBIGUOUS': 32}.
- Diversity: 198 source trajectories and 132 leakage groups.

## Exact stored example

Input is a 214-D unnormalized deployment-available vector. Its exact ordered segment schema and units are in `feature_schema.json`; structured arrays are also stored in `samples.npz`. The 4-D target is

`g*_exec(z,xi) = mean_{eta in E_near}[Pi_safe(u_safe + g_raw_eta) - u_safe]`

at the identical restored state and identical current Flow sample. Eta, J_def, and future continuation outcomes occur only in metadata, never in the feature vector.

## Integrity

- Restoration audit: PASS.
- Frozen hashes unchanged: True.
- Maximum same-seed u_safe discrepancy across eta: 0.000e+00.
- First/second projection replay errors: 0.000e+00 / 0.000e+00.
- Averaged target minimum CBF residual: 2.498e-16; maximum speed excess: -1.183e-01.
- Split leakage check: True (split by source/leakage group, never Flow seed).

## Runtime

- New continuation attempts: 41364; unique state/eta/seed tuples: 41364; duplicate interrupted-resume attempts: 0.
- Attempted physical steps: 10184076; unique-tuple physical steps: 10184076.
- Compatible old continuation tuples reused: 20732.
- End-to-end measured wall time: 5126.4 s.
- Resource use was batched and shared: eight allocated CPU cores plus two GPU shards, each process capped at 10% JAX memory; no full-card preallocation.

This task generated data only. **G_phi was not trained.**
