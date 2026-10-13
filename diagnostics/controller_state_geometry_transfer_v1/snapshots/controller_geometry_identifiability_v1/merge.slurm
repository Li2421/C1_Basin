#!/bin/bash
#SBATCH --job-name=geom_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=diagnostics/controller_geometry_identifiability_v1/logs/%x_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_geometry_identifiability_v1.study \
  merge --stage "${STAGE:?set STAGE}"
