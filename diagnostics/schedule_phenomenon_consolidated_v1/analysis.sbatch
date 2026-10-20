#!/bin/bash
#SBATCH --job-name=r21_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/schedule_phenomenon_consolidated_v1/analysis_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/schedule_phenomenon_consolidated_v1/analysis.py
