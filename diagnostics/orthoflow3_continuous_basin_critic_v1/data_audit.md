# Data audit

- Source: `/home/zhihan/research/Basin_C1/shared_rollout_db/rollout.sqlite` only; database unchanged during the experiment: **False**.
- Authoritative basis: `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.
- Canonical available pairs: Toy 13,307; DB 1,854.
- Sampled TRAIN pairs: Toy 3,146; DB 512.
- Evidence unit: unique compatible state×eta aggregate; duplicate experiment provenance and seed records were deduplicated.
- Excluded: quarantined conflicts, numerical failures, ambiguous rollouts, and non-authoritative controller profiles.
- Training weight: `min(n_trials,16)`; the empirical success ratio remains `n_success/n_trials`.

Success-rate coverage: `{"DB": {"available": {"clear_failure": 367, "intermediate": 116, "near_boundary": 4, "pairs": 1854, "robust": 1367}, "train": {"clear_failure": 55, "intermediate": 18, "near_boundary": 0, "pairs": 512, "robust": 439}}, "Toy": {"available": {"clear_failure": 2750, "intermediate": 639, "near_boundary": 37, "pairs": 13307, "robust": 9881}, "train": {"clear_failure": 787, "intermediate": 137, "near_boundary": 2, "pairs": 3146, "robust": 2220}}}`.

No rollout was requested or executed.
