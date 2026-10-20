#!/bin/bash
#SBATCH --job-name=crosstype1_old_evidence
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-9%10
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/old_evidence_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/worker.py old_probe "$SLURM_ARRAY_TASK_ID"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/worker.py old_evaluation "$SLURM_ARRAY_TASK_ID"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B /home/zhihan/research/Basin_C1/diagnostics/crosstype1_learned_zoo_selector_v1/worker.py old_tuning_v2 "$SLURM_ARRAY_TASK_ID"
