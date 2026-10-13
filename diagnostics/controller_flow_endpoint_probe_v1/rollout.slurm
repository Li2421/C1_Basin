#!/bin/bash
#SBATCH --job-name=q_flow_endpoint_probe
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --array=0-63%24
#SBATCH --output=diagnostics/controller_flow_endpoint_probe_v1/rollout_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.controller_flow_endpoint_probe_v1.study worker --index "$SLURM_ARRAY_TASK_ID" --workers 64
