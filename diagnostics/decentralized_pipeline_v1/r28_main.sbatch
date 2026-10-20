#!/bin/bash
#SBATCH --job-name=r28_main
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=06:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/decentralized_pipeline_v1/r28_main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
exec "$PY" -B diagnostics/decentralized_pipeline_v1/run.py main "$SLURM_ARRAY_TASK_ID" 18
