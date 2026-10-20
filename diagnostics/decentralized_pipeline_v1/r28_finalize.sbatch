#!/bin/bash
#SBATCH --job-name=r28_finish
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/decentralized_pipeline_v1/r28_finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/decentralized_pipeline_v1/merge.py
"$PY" -B diagnostics/decentralized_pipeline_v1/analysis.py
status=0
/home/zhihan/research/Basin_C1/.venv-video/bin/python -B diagnostics/decentralized_pipeline_v1/videos.py || status=$?
"$PY" -B diagnostics/decentralized_pipeline_v1/report.py
exit "$status"
