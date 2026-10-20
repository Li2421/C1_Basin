#!/bin/bash
#SBATCH --job-name=r27_freeze_main
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000M
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/other_controller_classes_v1/r27_freeze_main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/other_controller_classes_v1/probes.py freeze
"$PY" -B diagnostics/other_controller_classes_v1/freeze_choices.py
exec "$PY" -B diagnostics/other_controller_classes_v1/design.py main
