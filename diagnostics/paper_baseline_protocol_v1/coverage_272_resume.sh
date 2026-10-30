#!/bin/bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
mountpoint -q /mnt/ssd
if squeue -h -u "$USER" -o '%j' | rg -q '^pbp1_cov272_coordinator$'; then
  echo 'Historical-cohort coordinator already active.'
  exit 0
fi
active_cpus=$(squeue -h -u "$USER" -t RUNNING,CONFIGURING,COMPLETING -o '%C' | awk '{s+=$1} END{print s+0}')
if [ "$active_cpus" -ge 18 ]; then
  echo 'CPU budget currently fully allocated; no job submitted.'
  exit 1
fi
srun -p cpu --job-name=pbp1_cov272_coordinator --cpus-per-task=1 --mem=6000M --time=02:00:00 \
  env PYTHONDONTWRITEBYTECODE=1 STB_COORDINATOR=1 \
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B \
  diagnostics/paper_baseline_protocol_v1/coverage_272.py coordinator \
  >> diagnostics/paper_baseline_protocol_v1/coverage_272_artifacts/coordinator.log 2>&1
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B \
  diagnostics/paper_baseline_protocol_v1/coverage_272_analysis.py
