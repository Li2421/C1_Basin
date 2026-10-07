# ORTHOFLOW3_SPARSE_MODE_RELEARNING_V1

- Inputs are frozen from `orthoflow3_shared_eta_codebook_v1`.
- Sparse-codebook construction uses TRAIN outcomes only. Strong feasibility is
  `successes >= 15/16`.
- Desired per-mode prevalence is 10--30% and desired union coverage is
  70--85%. Modes above 35% are preferentially excluded.
- If no subset satisfies both bands, selection is frozen by: union coverage in
  70--85%, then minimum cardinality (at least two modes, because a singleton
  cannot test state-conditioned selection), minimum maximum prevalence, minimum mean
  prevalence, minimum mean Jaccard overlap, maximum minimum normalized eta
  separation, and lexicographic mode IDs.
- The MLP is reinitialized for seeds 17, 23, and 41. Model, temperature, and
  abstention threshold are selected using VAL only.
- Fresh replication is triggered if original TEST coverable-state B63
  selection is at least 80%, mean regret is at most 0.10, and mean selected Q64
  is within 0.10 of oracle.
