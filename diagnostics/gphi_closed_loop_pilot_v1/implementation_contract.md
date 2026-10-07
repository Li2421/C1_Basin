# Startup-complete G_phi pilot implementation contract

This directory contains the deployment evaluator only. It does not alter the
startup dataset or training directories and it never imports or calls an eta
search, basin search, continuation oracle, or learned gate.

## Required inputs

- `../gphi_training_startup_complete_v1/artifacts/best_checkpoint.npz`
  - fields: `architecture_json`, checkpoint normalization, binary mask, and
    three dense layers with shapes `214x128`, `128x128`, `128x4`;
  - the evaluator fails if the model is not exactly `214 -> 128 -> 128 -> 4`.
- `../gphi_training_startup_complete_v1/artifacts/normalization.json`
  - must exactly match the normalization embedded in the checkpoint.
- `../gphi_training_dataset_startup_complete_v1/samples.npz`
  - must contain `features`, `state_id`, and `split`; it defines the frozen
    train-state-centroid OOD reference.
- the audited startup feature wrapper in
  `../gphi_training_dataset_startup_complete_v1/startup_feature_builder.py`.

The original projection and environment paths/hashes are asserted before
rollout. The locked projection is
`/home/zhihan/research/02_C1_Toy_GiveWay/single_integrator/cbf.py`, SHA-256
`841a2dbb74676599d8c4187de9cf29920a6eda02c4372e29060ce6ca451ade48`.

## Freeze the fresh matched evaluation list

```bash
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/prepare_evaluation.py \
  --episodes 128
```

This audits numeric seed/namespace overlap and exact initial-position overlap
against the V3 and startup source manifests. The generated production manifest
is immutable: a different request refuses to overwrite it.

## Smoke mode

Use a separate seed file and namespace so smoke tuples can never enter the
production analysis:

```bash
$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/prepare_evaluation.py \
  --episodes 2 --purpose smoke \
  --ic-seed-base 3407000000 --flow-seed-base 3417000000 \
  --output diagnostics/gphi_closed_loop_pilot_startup_complete_v1/runs/smoke/seed_manifest.json

$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/run_evaluation.py \
  --seed-manifest diagnostics/gphi_closed_loop_pilot_startup_complete_v1/runs/smoke/seed_manifest.json \
  --controller both --namespace smoke --limit 2 --device cpu
```

## Production execution and sharding

One process can run both controllers. Alternatively, controller and episode
ranges can be sharded safely; the common scientific controller config is
independent of invocation/shard identity.

```bash
# Example disjoint workers (scheduler resource requests intentionally omitted).
$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/run_evaluation.py \
  --controller safety --start-index 0 --stop-index 128 --device gpu
$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/run_evaluation.py \
  --controller learned --start-index 0 --stop-index 64 --device gpu
$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/run_evaluation.py \
  --controller learned --start-index 64 --stop-index 128 --device gpu
```

Every controller/episode tuple is committed independently. A repeated command
validates and skips complete tuples rather than rerunning them.

After all 128 matched pairs exist:

```bash
$PY diagnostics/gphi_closed_loop_pilot_startup_complete_v1/analyze_results.py \
  --episodes 128
```

The analyzer fails closed on missing pairs and writes the requested success,
paired comparison, deformation, startup deformation, projection, OOD, timing,
failure, sanity, runtime, report, and manifest artifacts.

## Implemented diagnostics

- mutually exclusive success/deadlock/timeout/collision/other outcomes;
- `J_def = dt sum ||u_exec-u_safe||^2`, split into steps 0--40 and >=41;
- raw and executed correction norms, second-projection rewrite, retries,
  numerical/feasibility checks;
- normalized 214-D distance to the nearest training-state centroid, plus saved
  high-OOD and failure-preceding features;
- corrected-step fraction and consecutive correction run lengths;
- paired success delta with deterministic paired bootstrap CI and marginal
  Wilson intervals.
