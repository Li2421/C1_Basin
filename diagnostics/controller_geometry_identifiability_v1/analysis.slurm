#!/bin/bash
#SBATCH --job-name=geom_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/controller_geometry_identifiability_v1/logs/%x_%j.log
set -euo pipefail
export JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_geometry_identifiability_v1.analysis
