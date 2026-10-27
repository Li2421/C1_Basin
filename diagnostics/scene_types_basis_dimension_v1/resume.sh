#!/bin/bash
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
stb_watch_pid=''
if [[ -r diagnostics/scene_types_basis_dimension_v1/resource_watch.pid ]]; then
  read -r stb_watch_pid < diagnostics/scene_types_basis_dimension_v1/resource_watch.pid
fi
stb_watch_args=''
if [[ "$stb_watch_pid" =~ ^[0-9]+$ ]]; then
  stb_watch_args=$(ps -p "$stb_watch_pid" -o args= || true)
fi
if [[ "$stb_watch_args" != *scene_types_basis_dimension_v1/resource_watch.py* ]] && [[ ! -e diagnostics/scene_types_basis_dimension_v1/RESOURCE_WATCH_STOP ]]; then
  nohup /usr/bin/python3 diagnostics/scene_types_basis_dimension_v1/resource_watch.py > diagnostics/scene_types_basis_dimension_v1/resource_watch.log 2>&1 &
fi
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/scene_types_basis_dimension_v1/main.py
