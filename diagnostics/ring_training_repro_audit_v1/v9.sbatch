#!/bin/bash
#SBATCH --job-name=r24_v9
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_training_repro_audit_v1/v9_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_training_repro_audit_v1/step1_v9.py base_u_v9_local_macflow
