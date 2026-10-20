#!/bin/bash
#SBATCH --job-name=r24_followup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_training_repro_audit_v1/followup_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/ring_training_repro_audit_v1/initial_grid.py
diagnostics/ring_training_repro_audit_v1/venv_numpy222/bin/python -B diagnostics/ring_training_repro_audit_v1/step1_numpy222.py base_u_v2_macflow
$PY -B diagnostics/ring_training_repro_audit_v1/step1.py base_u_v9_local_macflow
