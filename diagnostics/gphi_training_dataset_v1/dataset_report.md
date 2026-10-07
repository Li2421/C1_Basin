# G_phi training dataset v1

## Result

**READY_FOR_PILOT_TRAINING**

- Selected exact augmented states: **67**; usable labeled states: **67**.
- Supervised samples: **4288** (64 matched Flow samples per usable state).
- Selected categories: NORMAL 24, PRE_DEADLOCK 23, RECOVERY 20; usable: {'PRE_DEADLOCK': 23, 'NORMAL': 24, 'RECOVERY': 20}.
- Zero/nonzero oracle states: **24 / 43**.
- B_63 empty states: **0**; multivalued quarantines: **0**.
- Label classes: {'LABEL_STABLE': 51, 'LABEL_MILDLY_AMBIGUOUS': 16}.

## Exact stored example

Input is a 214-D unnormalized deployment-available vector. Its exact ordered segment schema and units are in `feature_schema.json`; structured arrays are also stored in `samples.npz`. The 4-D target is

`g*_exec(z,xi) = mean_{eta in E_near}[Pi_safe(u_safe + g_raw_eta) - u_safe]`

at the identical restored state and identical current Flow sample. Eta, J_def, and future continuation outcomes occur only in metadata, never in the feature vector.

## Integrity

- Restoration audit: PASS.
- Frozen hashes unchanged: True.
- Maximum same-seed u_safe discrepancy across eta: 0.000e+00.
- First/second projection replay errors: 0.000e+00 / 0.000e+00.
- Averaged target minimum CBF residual: 2.498e-16; maximum speed excess: -2.806e-01.
- Split leakage check: True (split by source/leakage group, never Flow seed).

## Runtime

- New continuation attempts: 20549; unique state/eta/seed tuples: 20396; duplicate interrupted-resume attempts: 153.
- Attempted physical steps: 4558422; unique-tuple physical steps: 4519399.
- Compatible old continuation tuples reused: 336.
- End-to-end measured wall time: 3385.2 s.
- Resource use was batched and shared: four CPU workers during the largest phase plus one GPU shard capped at 10% JAX memory; no full-card preallocation.

This task generated data only. **G_phi was not trained.**
