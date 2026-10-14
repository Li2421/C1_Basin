#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_guard
#SBATCH --cpus-per-task=1
#SBATCH --mem=2500M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/wave_stage_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B "diagnostics/ring_controller_population_v1/wave_$1.py" "$2"
