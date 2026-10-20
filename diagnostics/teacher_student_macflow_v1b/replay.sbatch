#!/bin/bash
#SBATCH --job-name=r30b_replay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/teacher_student_macflow_v1b/replay_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/teacher_student_macflow_v1b/replay_gate.py run
$PY -B diagnostics/teacher_student_macflow_v1b/components.py
