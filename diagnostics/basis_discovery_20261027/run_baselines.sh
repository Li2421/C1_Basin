#!/usr/bin/env bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" diagnostics/basis_discovery_20261027/baseline_checks.py
"$PY" diagnostics/basis_discovery_20261027/baselines.py
