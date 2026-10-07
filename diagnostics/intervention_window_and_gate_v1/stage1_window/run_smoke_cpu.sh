#!/usr/bin/env bash
set -euo pipefail
AUDIT_ROOT=/home/zhihan/research/Basin_C1
AUDIT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
STAGE_DIR="$AUDIT_ROOT/diagnostics/intervention_window_and_gate_v1/stage1_window"
export PYTHONPATH="$AUDIT_ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$AUDIT_ROOT"
"$AUDIT_PY" "$STAGE_DIR/check_stage1_integrity.py"
"$AUDIT_PY" "$STAGE_DIR/run_delay_rollouts.py" \
  --delays 0,4 --stage smoke_reuse --shard 0 --shards 1 --device cpu --batch 16 \
  --selection-json "$STAGE_DIR/smoke_selection.json"
