#!/bin/bash
#SBATCH --job-name=r29_finish
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/nominal_safety_baselines_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/nominal_safety_baselines_v1/merge.py
"$PY" -B diagnostics/nominal_safety_baselines_v1/analyze.py
# Pure-Q evaluation: AGENTS.md explicitly exempts the video acceptance entry.
test -s diagnostics/nominal_safety_baselines_v1/FINAL_AUDIT.json
