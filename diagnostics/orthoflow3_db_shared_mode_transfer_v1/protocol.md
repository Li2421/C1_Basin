# ORTHOFLOW3_DOUBLE_BOTTLENECK_SHARED_MODE_TRANSFER_V1

- Frozen Toy anchors: the 12 mode IDs and coordinates from
  `orthoflow3_shared_eta_codebook_v1/codebook_eta.csv`.
- Shared object: mode identity in normalized persistent OrthoFlow3 eta space;
  the DB state encoder is scenario-local.
- DB feature: exact true-t0 4x18 observation plus the exact frozen current
  raw Flow action (80 scalars). This follows the existing DB conditioning
  identity and does not force the Toy 214-D schema.
- Outcome-blind source split: 64/16/16 source groups, hash ordered before any
  new mode outcome. Four previously geometry-inspected DB source groups are
  quarantined as transform-development states and cannot enter VAL/TEST.
- Transform hierarchy: anisotropic similarity, condition-controlled affine,
  then affine plus a bounded scenario-level per-mode residual (normalized
  norm <=0.20). All transforms act on all 12 anchors; mode IDs never change.
- Selection and fitting use transform-development/TRAIN evidence only. TEST
  is quarantined until the transform, selector, calibration, and threshold
  are frozen.
- Exact matrix: TRAIN 16, VAL 32, TEST 64 matched future continuations per
  state-mode pair. Raw Toy anchors are a frozen diagnostic baseline.
- Controller semantics and basis implementation are unchanged.

