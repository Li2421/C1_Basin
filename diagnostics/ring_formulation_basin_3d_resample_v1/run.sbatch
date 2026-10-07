#!/bin/bash
#SBATCH --job-name=ring-basin-illustrative
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-63%24
#SBATCH --chdir=/home/zhihan/research/Basin_C1
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_formulation_basin_3d_resample_v1/logs/%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/ring_formulation_basin_3d_resample_v1/logs/%A_%a.err
set -euo pipefail
export JAX_PLATFORMS=cpu JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1
export XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
srun --cpu-bind=cores /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -u diagnostics/ring_formulation_basin_3d_resample_v1/worker.py --index "$SLURM_ARRAY_TASK_ID"
