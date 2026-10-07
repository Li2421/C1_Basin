#!/bin/bash
#SBATCH --job-name=ssrec_exitfinal
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/exitfinal-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/logs/exitfinal-%j.err
set -euo pipefail
D=/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1; PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/02_C1_Toy_GiveWay:/home/zhihan/research/Basin_C1:$D XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.08 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
"$PY" "$D/train_decision_head_grid.py" --dataset-manifest "$D/labels/exit_merged/decision_dataset_manifest.json" --output-directory "$D/entry_exit_training_history/exit_final" --execute
