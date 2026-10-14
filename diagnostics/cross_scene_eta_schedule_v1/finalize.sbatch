#!/bin/bash
#SBATCH --job-name=r13_delivery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/cross_scene_eta_schedule_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
R13_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
R13_DIR=diagnostics/cross_scene_eta_schedule_v1
$R13_PY -B "$R13_DIR/merge.py"
$R13_PY -B "$R13_DIR/analyze.py"
R13_VIDEO_RC=0
$R13_PY -B "$R13_DIR/videos.py" || R13_VIDEO_RC=$?
$R13_PY -B "$R13_DIR/report.py"
exit "$R13_VIDEO_RC"
