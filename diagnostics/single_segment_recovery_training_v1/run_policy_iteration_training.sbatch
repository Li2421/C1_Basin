#!/bin/bash
#SBATCH --job-name=ssrec_pi_train
#SBATCH --partition=gpu
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/pi-train-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/pi-train-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$D
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.08
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=0 "$PY" "$D/train_decision_head_grid.py" \
  --dataset-manifest "$D/labels/entry_policy_iteration_merged/decision_dataset_manifest.json" \
  --output-directory "$D/entry_exit_training_history/entry_policy_iteration" --execute \
  >"$D/logs/pi-entry-train.out" 2>"$D/logs/pi-entry-train.err" &
entry_pid=$!
CUDA_VISIBLE_DEVICES=0 "$PY" "$D/train_decision_head_grid.py" \
  --dataset-manifest "$D/labels/exit_policy_iteration_merged/decision_dataset_manifest.json" \
  --output-directory "$D/entry_exit_training_history/exit_policy_iteration" --execute \
  >"$D/logs/pi-exit-train.out" 2>"$D/logs/pi-exit-train.err" &
exit_pid=$!
status=0
wait "$entry_pid" || status=$?
wait "$exit_pid" || status=$?
exit "$status"
