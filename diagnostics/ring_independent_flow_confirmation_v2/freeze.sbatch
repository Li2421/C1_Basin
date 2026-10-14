#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r14a_freeze
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/freeze_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/probe.py merge
"$PY" -B diagnostics/ring_independent_flow_confirmation_v2/models.py
exec "$PY" -B diagnostics/ring_independent_flow_confirmation_v2/design.py create
