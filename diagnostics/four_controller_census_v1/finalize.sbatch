#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14b_finish
#SBATCH --cpus-per-task=1
#SBATCH --mem=3000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/four_controller_census_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/four_controller_census_v1/merge.py
exec "$PY" -B diagnostics/four_controller_census_v1/finalize.py
