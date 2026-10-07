# Startup-complete deterministic G_phi training

This directory contains an isolated, fail-closed pipeline for merging the
immutable V3 corpus with a future startup-state corpus and retraining the
frozen `214 -> 128 -> 128 -> 4` SiLU regressor from scratch.  The scripts do
not write into either input dataset.

## Required startup dataset contract

The default input is
`diagnostics/gphi_training_dataset_startup_complete_v1/`.  It must contain:

- `samples.npz`
- `state_manifest.jsonl`
- `sample_metadata.jsonl`
- `feature_schema.json`
- `protocol.json`
- `manifest.json`

The dataset is expected to be the authoritative merged output produced by
`finalize_startup_dataset.py`: the complete V3 corpus as an exact prefix,
followed by the startup-state suffix.  `samples.npz` must have every array key
present in V3 with identical trailing shapes and compatible dtypes.  In
particular it must contain 214-D
`features`, 4-D `targets`, `target_actions`, `u_safe`, `state_id`,
`sample_id`, `category`, and `state_index`.  Every state-manifest row must
contain a nonempty `source_episode`; this is the indivisible split unit.
Every sample-metadata row must identify its `state_id` and `sample_id`.  The
suffix state/sample IDs must be new, must use category `STARTUP`, and must
provide `source_episode`.  Source-episode splits are frozen by the dataset
generator at approximately 70/15/15; this pipeline verifies them and rejects
any episode/group leakage rather than re-splitting the data.  The feature
schema and frozen environment/projection protocol must match V3 exactly.  The
startup manifest must provide the SHA-256 of `samples.npz` under
`files_sha256`.

V3 memberships are unchanged.  V3 arrays must be the exact prefix of the
merged arrays; numeric prefix bytes and all string values are verified before
training.

The cohort definitions are fixed:

- `STARTUP`: rows originating in the startup dataset.
- `WARM_V3`: the unchanged V3 corpus (states with the established full-history
  representation).
- category metrics use the literal `category` values supplied by the inputs.

## Usage

Audit and merge only (no ML imports or training):

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/gphi_training_startup_complete_v1/pipeline.py --prepare-only
```

Train all registered seeds and evaluate offline:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  diagnostics/gphi_training_startup_complete_v1/pipeline.py
```

Generated files go only to `artifacts/` below this directory.  The selected
checkpoint is chosen solely by validation state-grouped MSE.  The projection
replay integrity gate requires finite outputs, zero validation/test solver
failures, and exact oracle-target reconstruction.  Projection rewrite size is
reported but is not used to select a model.

The requested output path must not already exist.  A run writes to an isolated
same-filesystem `.artifacts.inprogress.<pid>` directory and atomically renames
it to `artifacts/` only after every integrity gate passes.  Failed/interrupted
runs remain explicitly non-complete staging directories and cannot be mixed
with a later run.

Tests (no dataset writes and no training):

```bash
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m unittest \
  diagnostics.gphi_training_startup_complete_v1.test_pipeline
```
