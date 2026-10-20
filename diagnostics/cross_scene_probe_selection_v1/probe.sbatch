#!/bin/bash
#SBATCH --job-name=r20_probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_probe_selection_v1/probe_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_probe_selection_v1/probe.py "$SLURM_ARRAY_TASK_ID" 12
