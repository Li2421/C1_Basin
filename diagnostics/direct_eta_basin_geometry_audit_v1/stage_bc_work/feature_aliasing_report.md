# Feature aliasing and zero/active diagnostic audit

This is a CPU-only read-only audit of the frozen 27,136-sample / 424-state
dataset. It trained only the explicitly permitted diagnostic probes and ran no
continuations or simulator calls.

## Provenance and rules

- Authoritative samples SHA256: `79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9`
- Authoritative metadata SHA256: `03714a831f9f98a091ae6e7be8aa67c9b8243e3c81f0ff8098f07b835f691213`
- Frozen G_eta checkpoint SHA256: `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`
- Geometry normalization: exact frozen training-split mean/scale reproduced to
  max absolute mean error 0 and scale error 0.
- State cloud rule: centroid of all 64 normalized Flow variants.
- Source grouping: authoritative `leakage_group`; groups are indivisible across
  the five diagnostic cross-validation folds.

## Exact aliasing

- Bitwise-identical cross-state feature pairs: 0.
- Bitwise-identical pairs with different canonical eta: 0.
- Additional cross-state pairs within 1e-12 L-infinity in normalized feature
  space: 0; incompatible eta among them: 0.

See `feature_aliasing_summary.json` and the full nearest-neighbor CSVs for the
near-collision distributions. Ordinary overlap is not classified as
representation impossibility.

At state-centroid level, 22.2%
of directed nearest-neighbor pairs change canonical eta, 10.6%
have coordinate-normalized eta jump >=0.5, and 9.2%
cross ZERO/ACTIVE. The closest >=0.5 eta jump has normalized-feature L2
0.375;
the closest ZERO/ACTIVE crossing is farther away at
0.852.

## Zero/active probes

Primary metrics are state-level pooled out-of-fold results. Random-sample
metrics are explicitly a leaky shortcut diagnostic because Flow variants of
the same state occur in train and test. Full metrics, fold composition,
optimizer diagnostics, sensitivity, and specificity are in
`zero_active_probe_results.json`.

- Linear grouped: AUROC 0.934,
  BAcc 0.857.
- MLP grouped: AUROC 0.940,
  BAcc 0.920.
- Random-sample shortcut: linear AUROC
  0.987,
  MLP AUROC 1.000.

Grouped performance remains strong, so ZERO/ACTIVE information is present in
the representation. The random/group gap shows measurable shortcut optimism,
but not a grouped-evaluation collapse.

## Canonical smoothness

Preliminary smoothness uses the five directed nearest state-centroid neighbors
per state. It is target geometry only; it does not establish successful-set
discontinuity. Cross-eta continuation tests are intentionally outside this
subtask.
