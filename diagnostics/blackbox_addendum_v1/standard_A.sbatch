#!/bin/bash
#SBATCH --job-name=r37_A_stats
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/blackbox_addendum_v1/A_stats_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/blackbox_addendum_v1/standard_A.py
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/blackbox_addendum_v1/geometry_curves.py
