#!/bin/bash
#SBATCH --job-name=r22_finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_eta_pair_generality_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/ring_eta_pair_generality_v1/merge.py
exec $PY -B diagnostics/ring_eta_pair_generality_v1/analyze.py
