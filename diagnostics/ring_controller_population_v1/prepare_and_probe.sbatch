#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=05:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/probe_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
$PY -B diagnostics/ring_controller_population_v1/prepare_targets.py
$PY -B diagnostics/ring_controller_population_v1/probe_r5.py create
$PY -B diagnostics/ring_controller_population_v1/probe_r5.py worker 0 1
$PY -B diagnostics/ring_controller_population_v1/probe_r5.py merge
$PY -B diagnostics/ring_controller_population_v1/probe_r6.py create
$PY -B diagnostics/ring_controller_population_v1/probe_r6.py worker 0 1
$PY -B diagnostics/ring_controller_population_v1/probe_r6.py merge
