#!/bin/bash
#SBATCH --job-name=r21_complete
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/schedule_phenomenon_consolidated_v1/complete_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/schedule_phenomenon_consolidated_v1/complete.py
