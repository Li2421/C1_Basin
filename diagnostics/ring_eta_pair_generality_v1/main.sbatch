#!/bin/bash
#SBATCH --job-name=r22_main
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500M
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta_pair_generality_v1/main_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_eta_pair_generality_v1/run.py main "$SLURM_ARRAY_TASK_ID" 18
