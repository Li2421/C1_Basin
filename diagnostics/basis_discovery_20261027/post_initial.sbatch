#!/bin/bash
#SBATCH --job-name=c1_dict_combos
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/basis_discovery_20261027/post_initial_slurm.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" diagnostics/basis_discovery_20261027/static_audit.py
"$PY" diagnostics/basis_discovery_20261027/screen.py expand --workers "$SLURM_CPUS_PER_TASK"
"$PY" diagnostics/basis_discovery_20261027/matched_controls.py
"$PY" diagnostics/basis_discovery_20261027/screen.py delete --workers "$SLURM_CPUS_PER_TASK"
"$PY" diagnostics/basis_discovery_20261027/screen.py minimal
