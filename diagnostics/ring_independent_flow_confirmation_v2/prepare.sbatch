#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14a_prepare
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/prepare_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/prepare_targets.py
exec "$PY" -B diagnostics/ring_independent_flow_confirmation_v2/probe.py create
