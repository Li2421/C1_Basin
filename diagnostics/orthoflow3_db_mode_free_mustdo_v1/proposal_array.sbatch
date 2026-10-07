#!/bin/bash
#SBATCH --job-name=ortho3-db-proposal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --array=0-4
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_db_mode_free_mustdo_v1/logs/proposal_%A_%a.out
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
/home/zhihan/venvs/gpu/bin/python diagnostics/orthoflow3_db_mode_free_mustdo_v1/run_proposals.py --shard "${SLURM_ARRAY_TASK_ID}" --shards 5
