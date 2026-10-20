#!/bin/bash
#SBATCH --job-name=r22_gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta_pair_generality_v1/gate_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_eta_pair_generality_v1/run.py gate "$SLURM_ARRAY_TASK_ID" 8
