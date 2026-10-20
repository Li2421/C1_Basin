#!/bin/bash
#SBATCH --job-name=r20_finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_probe_selection_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH=/home/zhihan/research/Basin_C1/diagnostics/controller_loco_real_flow_v1/plot_vendor
$PY -B diagnostics/cross_scene_probe_selection_v1/merge.py
$PY -B diagnostics/cross_scene_probe_selection_v1/postflight.py
$PY -B diagnostics/cross_scene_probe_selection_v1/analyze.py
exec $PY -B diagnostics/cross_scene_probe_selection_v1/videos.py
