#!/bin/bash
#SBATCH --job-name=csg_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/controller_state_geometry_transfer_v1/logs/%x_%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
C1PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.main_analysis evaluate --legacy
"$C1PY" -m diagnostics.controller_state_geometry_transfer_v1.calibration
