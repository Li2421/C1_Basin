# Reproduction and recovery

Read `HANDOFF.md` and `REPORT.md` first. The strong-state **selection** gate
failed. `broad_main_study.py` is a guarded, unexecuted design; do not remove its
guard or describe it as completed. All primary predictions and raw evidence are
immutable reference artifacts. Recalculation may rewrite derived reports;
copy the directory first if preserving byte-identical derived files matters.

## Environment and read-only checks

```bash
cd /home/zhihan/research/Basin_C1
C1PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Package versions: `environment_requirements.txt`. Source hashes and frozen Toy
Flow checkpoint: `source_implementation_audit.json`. Git HEAD and dirty baseline
are in the first preregistration receipt and `git_status_baseline.txt`. Existing
PoC code and legacy data are external dependencies at the recorded absolute
paths; this directory is not a full backup of the shared SQLite or old projects.

```bash
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.final_audit
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.test_contracts
```

`FINAL_AUDIT.json` verifies all 7,172 attempted records, their cache entries,
protocol hashes and all 3,072 factor trajectory hashes/lengths; independently
rechecks swept safety on stratified examples. It does **not** declare the
scientific strong-selection gate passed. One numerical failure is retained in
SQLite but intentionally excluded by exact-label lookup: postflight reports
7,171 reusable binary labels and one missing valid label, not an unrun episode.

## Regenerate metrics and figures from existing evidence

Run the following sequentially on CPU; no training or rollout is performed.

```bash
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.pilot
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.pilot_revision_analysis
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.broad_pilot_analysis
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.horizon_pilot
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.main_analysis evaluate --legacy
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.calibration
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.factor_analysis evaluate
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.normalization_diagnostic
JAX_PLATFORMS=cpu "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.replay_analysis --recorded
/home/zhihan/venvs/gpu/bin/python -m diagnostics.controller_state_geometry_transfer_v1.figures
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.report
```

The plot interpreter provides matplotlib3.10.9; training Python does not.
`plot_environment_requirements.txt` records installed package metadata; no
packages or shared environments were modified.

The normalization command reuses all nine existing `.npz` predictions; if those
are absent it fits models, so use its Slurm script instead. Main and factor
evaluation expect the preserved `.npz` predictions. `replay_analysis` uses the
separately saved GPU checkpoint replays. Original empirical-Q, unknown masks,
features and training indices are in `data/` and `models/`.

## Checkpoint inference and fresh fitting

`checkpoint_tools.py replay` reconstructs predictions from `.msgpack` without
training. GPU execution must use Slurm; this checks the saved checkpoint's
meaning, not a guarantee of bitwise CPU/GPU or XLA recompilation reproducibility.
The corrected replay explicitly matches original precision: Toy x64=false,
factor/normalization x64=true. `poc.field_controller` sets the latter during
original factor imports, and the entity encoder's implicit-dtype identity matrix
promotes pooling. All138 matching-precision GPU argmax choices reproduce.
See `CHECKPOINT_REPLAY_{cpu,gpu}_recorded.json` and
`REPLAY_SENSITIVITY_recorded.json`. Files without `_recorded` preserve the first
audit's omitted-x64 error as a precision-mismatch sensitivity, not proof of
same-setting numerical instability. Strict Q tolerances are separately reported.

```bash
sbatch --job-name=csg_replay --partition=gpu --gres=shard:1 \
  --cpus-per-task=2 --mem=10G --time=00:15:00 \
  --output=diagnostics/controller_state_geometry_transfer_v1/logs/replay_%j.log \
  --wrap='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_state_geometry_transfer_v1.checkpoint_tools replay --all'
```

To reproduce a fit, use a **new nonexistent** output directory. The helper reads
the saved training indices/labels, fixed seed, architecture, step count and
source-only normalization from the preserved dataset. It refuses overwriting
an existing destination. Run via a Slurm GPU allocation, for example:

```bash
srun --partition=gpu --gres=shard:1 --cpus-per-task=2 --mem=10G --time=00:20:00 \
  "$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.checkpoint_tools refit \
  --checkpoint diagnostics/controller_state_geometry_transfer_v1/models/legacy_diagnostic/MANY/ENT_17.msgpack \
  --outdir diagnostics/controller_state_geometry_transfer_v1_refit_example
```

No such additional fit is required to use these results. The original batch
launchers `train_legacy.slurm`, `factor_train.slurm`, `normalization.slurm`
reuse completed fits; cached returns verify training indices/labels but do not
check every feature hash. Use `MANIFEST.json` before replay or a new destination
after implementation changes.

## Rollout recovery and shared cache

All originally planned attempts are complete; do not rerun the numerical
failure. Map each stage to its module:

| Directory | Python module | Worker launcher |
| --- | --- | --- |
| pilot_revision | revision_study | revision_worker.slurm |
| broad_pilot | broad_study | broad_worker.slurm |
| legacy_supplement | supplement | supplement_worker.slurm |
| prefix_audit | prefix_study | prefix_worker.slurm |
| factors | factor_study | factor_worker.slurm |

Each module is under `diagnostics.controller_state_geometry_transfer_v1`.
Frozen `protocol.json`, `cases.json` and `planned_rollouts.json` are already
registered. A `design` command checks immutability and registers identities;
some Toy designs also require the preserved stage-local prereg/reproduction
files. Do not regenerate identifiers by changing implementation paths.

For an actual interrupted/restored stage, first compare code hashes, then:

```bash
"$C1PY" -m shared_rollout_db.plan \
  --manifest diagnostics/controller_state_geometry_transfer_v1/factors/planned_rollouts.json \
  --output diagnostics/controller_state_geometry_transfer_v1/factors/cache_preflight.json
# Inspect missing seeds against existing raw attempts before any sbatch.
# Workers skip existing raw records, including unknowns.
sbatch diagnostics/controller_state_geometry_transfer_v1/factor_worker.slurm
# Only after workers finish, ONE merger:
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.factor_study merge
```

Workers never write SQLite; they append journals. Do not run simultaneous
mergers. A cached record with a lost local file should be restored from its
indexed source, not regenerated. The pure-Q video exemption applies.

## Completed execution evidence

Revision19071; broad19088; supplement19104; native-prefix19121;
legacy fits19120; calibration19126 (later metric-only additions);
factor probe19125/full19127; factor fits19153/evaluation19145;
normalization19155; checkpoint audit19162/19163, corrected precision19165.
Failed assertions and subsequent
corrections are documented in `EXECUTION_NOTES.md`; logs are retained. Slurm
accounting is disabled, so per-record elapsed time is an approximation, not
an accounting-system CPU/GPU usage claim.

Storage integrity: `MANIFEST.json`; new raw index: `DATA_INDEX.json`; archived
input hashes: `data/*index.json`. `bundle_receipt.json`, if present, identifies
a local compressed backup and explicitly lists its scope.

```bash
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.artifact_manifest --verify
```

After intentional recalculation, `artifact_manifest` without `--verify` checks
all referenced raw/prediction hashes and writes a fresh manifest. Preserve the
previous manifest if comparing revisions. `DELIVERY_CHECK.json` distinguishes
artifact completeness from the still-failed scientific strong-selection gate.
