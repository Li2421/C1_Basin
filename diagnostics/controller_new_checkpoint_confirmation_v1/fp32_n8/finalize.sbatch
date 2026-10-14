#!/bin/bash
#SBATCH --job-name=r12f_delivery
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
R12_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
R12_DIR=diagnostics/controller_new_checkpoint_confirmation_v1/fp32_n8
$R12_PY -B "$R12_DIR/merge.py"
$R12_PY -B "$R12_DIR/analyze.py"
$R12_PY -B "$R12_DIR/video_history.py"
R12_VIDEO_RC=0
$R12_PY -B "$R12_DIR/videos.py" || R12_VIDEO_RC=$?
# Report the real gate failure while preserving numerical results.
$R12_PY -B "$R12_DIR/report.py"
exit "$R12_VIDEO_RC"
