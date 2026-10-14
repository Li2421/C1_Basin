#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_design_gate
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/design_gate_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/ring_controller_population_v1/design.py create
$PY -B diagnostics/ring_controller_population_v1/waves.py
$PY -B diagnostics/ring_controller_population_v1/run.py gate 0 1
$PY -B diagnostics/ring_controller_population_v1/gate.py
