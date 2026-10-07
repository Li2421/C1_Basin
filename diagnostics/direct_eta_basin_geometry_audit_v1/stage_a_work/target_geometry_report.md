# Stage A — 424-state canonical eta target geometry

## Counting and provenance

Every unique augmented `state_id` counts exactly once. The 27,136 deployment-feature rows were used only to verify the invariant 64 Flow variants per state and one shared canonical eta; they do not reweight any statistic. Canonical eta is read directly from `eta_best_metadata_only` and cross-checked against the frozen `eta_targets.npz`; eta is never inferred from `g`.

The exact checkpoint SHA256 is `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`. The exact samples SHA256 is `79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9`.

## Main result

- ZERO: **242 / 424 (57.08%)**.
- ACTIVE: **182 / 424 (42.92%)**.
- Distinct canonical eta vectors at absolute tolerance 1e-12: **14**.
- Exactly repeated vectors: **10**; singleton vectors: **4**; **420 / 424** states belong to a repeated vector.
- Geometry description: **large exact zero spike plus a small set of repeatedly selected active lattice/refinement modes**. This is not one smooth continuous cloud. This statement is about canonical target B, not success-set topology A.

## Exact canonical eta frequencies

| Rank | eta | States | Fraction |
|---:|---|---:|---:|
| 1 | (0, 0, 0) | 242 | 57.08% |
| 2 | (0.375, -0.4375, -0.0625) | 103 | 24.29% |
| 3 | (0.40625, -0.5, 0) | 22 | 5.19% |
| 4 | (0.40625, -0.4375, -0.0625) | 22 | 5.19% |
| 5 | (0.4375, -0.53125, 0) | 8 | 1.89% |
| 6 | (0.5703125, -0.375, -0.125) | 7 | 1.65% |
| 7 | (0.578125, -0.40625, -0.125) | 7 | 1.65% |
| 8 | (0.5, -0.5, 0) | 4 | 0.94% |
| 9 | (1.25, -0.5, 0.375) | 3 | 0.71% |
| 10 | (1, 0, 0.25) | 2 | 0.47% |
| 11 | (0.5, -0.4375, 0) | 1 | 0.24% |
| 12 | (0.5, -0.375, 0) | 1 | 0.24% |
| 13 | (1.25, -0.5, 0) | 1 | 0.24% |
| 14 | (1.25, -0.5, 0.0625) | 1 | 0.24% |

## Coordinate distributions

Columns after population are min, q05, q25, median, mean, q75, q95, max, population SD.

| Population | min | q05 | q25 | median | mean | q75 | q95 | max | SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eta1 all | 0 | 0 | 0 | 0 | 0.187002506 | 0.375 | 0.5 | 1.25 | 0.239786582 |
| eta1 active | 0.375 | 0.375 | 0.375 | 0.375 | 0.43565419 | 0.40625 | 0.578125 | 1.25 | 0.16007597 |
| eta2 all | -0.53125 | -0.5 | -0.4375 | 0 | -0.19037441 | 0 | 0 | 0 | 0.222753401 |
| eta2 active | -0.53125 | -0.5 | -0.4375 | -0.4375 | -0.443509615 | -0.4375 | -0.3765625 | 0 | 0.0576905191 |
| eta3 all | -0.125 | -0.0625 | -0.0625 | 0 | -0.0185731132 | 0 | 0 | 0.375 | 0.0512013514 |
| eta3 active | -0.125 | -0.125 | -0.0625 | -0.0625 | -0.0432692308 | -0.0625 | 0 | 0.375 | 0.0709847263 |

## Eta norm distributions

| Quantity | min | q05 | q25 | median | mean | q75 | q95 | max | SD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| physical_eta_l2_all_states | 0 | 0 | 0 | 0 | 0.272477614 | 0.579601156 | 0.693906584 | 1.39754249 | 0.327187293 |
| physical_eta_l2_active_states | 0.579601156 | 0.579601156 | 0.579601156 | 0.579601156 | 0.634783013 | 0.644235254 | 0.717560156 | 1.39754249 | 0.139317796 |
| coordinate_normalized_eta_l2_all_states | 0.321506615 | 0.321506615 | 0.344951451 | 0.534592599 | 0.466548368 | 0.534592599 | 0.534592599 | 1.15214968 | 0.125092303 |
| coordinate_normalized_eta_l2_active_states | 0.321506615 | 0.321506615 | 0.321506615 | 0.321506615 | 0.376071972 | 0.356302451 | 0.480750355 | 1.15214968 | 0.148702733 |

## Frozen union-envelope boundary saturation

The frozen normalization envelope is low `[0.0, -0.53125, -0.125]`, high `[1.25, 0.5, 0.75]`. It is the union envelope documented by the fixed-D target preparation, enlarged only to include eta=0. Exact uses absolute tolerance 1e-12; near/on means coordinate-normalized distance <=1% from any face.

- Any exact face: **269 / 424 (63.44%)**.
- Among ACTIVE only: **27 / 182 (14.84%)**.
- Near/on any face at 1%: **269 / 424 (63.44%)**; ACTIVE only **27 / 182 (14.84%)**.
- Exact coordinate-face counts: `{"eta1_high": 5, "eta1_low": 242, "eta2_high": 0, "eta2_low": 8, "eta3_high": 0, "eta3_low": 14}`.

The overall boundary fraction is dominated by the 242 exact-zero targets lying on the eta1 lower envelope. The ACTIVE-only fraction is the relevant saturation diagnostic.

## Stored robust candidates and near-equivalent selection

- Multiple stored robust B63 eta values: **176 / 424 (41.51%)**.
- Multiple statistically near-equivalent `E_near` candidates confirmed: **73 states**.
- `E_near` metadata is present for **413** V3/startup states and unavailable for the **11** strict-deadlock augmentations. Thus the conservative all-state frequency is **17.22%**, or **17.68%** among states with that field.

Accordingly, frozen canonical selection chose one target from a stored multi-`E_near` near-equivalent set in **73 confirmed states**. These are not necessarily exact floating-point J_def ties: `E_near` denotes paired-statistical indistinguishability under the frozen oracle procedure.

Multiple stored robust candidates do not establish disconnected basins. Likewise, repeated canonical values reflect a finite frozen candidate/refinement lattice and frozen minimum-J_def selection; they are not by themselves evidence for a multimodal conditional success set.

## Existing category slices (post-hoc only)

| Existing category | States | ZERO | ACTIVE | Distinct eta | Multiple stored robust | Multiple near-equivalent confirmed |
|---|---:|---:|---:|---:|---:|---:|
| NORMAL | 120 | 104 | 16 | 5 | 16 | 7 |
| PRE_DEADLOCK | 60 | 0 | 60 | 7 | 58 | 32 |
| RECOVERY | 114 | 75 | 39 | 6 | 39 | 20 |
| STARTUP | 119 | 63 | 56 | 5 | 55 | 14 |
| STRICT_DEADLOCK | 11 | 0 | 11 | 6 | 8 | 0 |

No semantic phase was resampled or reweighted. No clustering was used to declare multimodality.

## Files

- `target_geometry_424.csv`: one row per unique state, with source/split/category, canonical eta, norm, candidate metadata, and boundary flags.
- `canonical_eta_frequency.csv`: one row per distinct canonical eta vector.
- `stage_a_summary.json`: machine-readable statistics and frozen rules.
- `asset_inventory.json`: exact authoritative paths, hashes, and schemas.
- `frozen_asset_hashes.json`: compact path/hash table for parent-manifest assembly.
- `dataset_424_manifest.json`: one authoritative manifest entry per unique state.
- `stage_audit.py`: deterministic reproduction script.
