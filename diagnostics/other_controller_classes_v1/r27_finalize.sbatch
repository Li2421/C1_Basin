#!/bin/bash
#SBATCH --job-name=r27_finalize
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10000M
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/other_controller_classes_v1/r27_finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/other_controller_classes_v1/merge.py
"$PY" -B diagnostics/other_controller_classes_v1/o4.py
exec "$PY" -B diagnostics/other_controller_classes_v1/analyze.py
