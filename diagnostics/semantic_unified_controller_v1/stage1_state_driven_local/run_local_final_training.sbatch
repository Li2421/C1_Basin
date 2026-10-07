#!/bin/bash
#SBATCH --job-name=semantic_local_train
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage1_state_driven_local/logs/local-train-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage1_state_driven_local/logs/local-train-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/semantic_unified_controller_v1/stage1_state_driven_local
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1:$D
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.08
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
"$PY" "$D/train_local_head_grid.py" \
  --dataset-manifest "$D/labels/local_merged/decision_dataset_manifest.json" \
  --output-directory "$D/training/local_final" --execute
