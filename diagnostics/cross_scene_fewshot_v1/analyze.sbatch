#!/bin/bash
#SBATCH --job-name=r23_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_fewshot_v1/analysis_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_fewshot_v1/analyze.py
