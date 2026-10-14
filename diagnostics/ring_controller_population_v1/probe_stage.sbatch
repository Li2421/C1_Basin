#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_probe_stage
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/probe_stage_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
case "$1" in
  prepare) $PY -B diagnostics/ring_controller_population_v1/prepare_targets.py; $PY -B diagnostics/ring_controller_population_v1/probe_r5.py create ;;
  r5merge) $PY -B diagnostics/ring_controller_population_v1/probe_r5.py merge; $PY -B diagnostics/ring_controller_population_v1/probe_r6.py create ;;
  r6merge) $PY -B diagnostics/ring_controller_population_v1/probe_r6.py merge ;;
  *) exit 2 ;;
esac
