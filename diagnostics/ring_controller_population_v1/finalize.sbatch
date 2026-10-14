#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=r15_finalize
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000M
#SBATCH --time=00:45:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_controller_population_v1/finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
deadline_epoch=$(date -d '2026-10-14 10:30:00 +08' +%s)
remaining=$((deadline_epoch-$(date +%s)))
if (( remaining <= 0 )); then exit 1; fi
exec timeout --signal=TERM "${remaining}s" /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_controller_population_v1/finalize.py
