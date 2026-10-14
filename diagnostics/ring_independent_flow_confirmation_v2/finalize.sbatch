#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14a_finalize
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/merge.py
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/analyze.py
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/failure_details.py
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/plots.py
set +e
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/videos.py
video_rc=$?
set -e
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/report.py
exit "$video_rc"
