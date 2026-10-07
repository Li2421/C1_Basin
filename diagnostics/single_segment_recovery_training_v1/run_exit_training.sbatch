#!/bin/bash
#SBATCH --job-name=ssrec_exittrain
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/exittrain-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/exittrain-%j.err
set -euo pipefail
TASK_DIR=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$TASK_DIR
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.08
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
"$PY" "$TASK_DIR/train_decision_head_grid.py" \
  --dataset-manifest "$TASK_DIR/labels/exit_bootstrap_never_exit/decision_dataset_manifest.json" \
  --output-directory "$TASK_DIR/entry_exit_training_history/exit_bootstrap" --execute
