#!/usr/bin/env bash
# Planning/orchestration template.  It intentionally does not submit itself;
# the parent must inspect shared-server load and choose SHARDS before launch.
set -euo pipefail

ROOT=/home/zhihan/research/Basin_C1
OUT="$ROOT/diagnostics/gphi_training_dataset_startup_complete_v1"
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
SHARDS=${SHARDS:-1}
DEVICE=${DEVICE:-gpu}
export PYTHONPATH="$ROOT:$OUT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

"$PY" "$OUT/freeze_startup_states.py"
"$PY" "$OUT/audit_startup_states.py"
"$PY" "$OUT/audit_startup_feature_builder.py"
"$PY" "$OUT/plan_oracle.py" --phase zero
"$PY" "$OUT/split_arms.py" --arms eta_zero_arms.json --shards "$SHARDS" --prefix eta_zero

cat <<EOF
Preparation complete.  Do not advance automatically: launch each frozen shard
only after the shared-resource check, e.g. for shard i:
  $PY $OUT/run_oracle.py --arms eta_zero_shard\${i}_arms.json --stage eta_zero_shard\${i} --device $DEVICE --batch 32
Then run:
  $PY $OUT/plan_oracle.py --phase candidates
  $PY $OUT/split_arms.py --arms candidate_arms.json --shards $SHARDS --prefix candidate
  # launch candidate shards as above
  $PY $OUT/plan_oracle.py --phase assessment
  $PY $OUT/finalize_startup_dataset.py
EOF

