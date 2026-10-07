#!/bin/bash
#SBATCH --job-name=interaction-probe
#SBATCH --partition=cpu
#SBATCH --array=0-15
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/orthoflow3_state_eta_interaction_v1/%A_%a_probe.out
#SBATCH --error=diagnostics/orthoflow3_state_eta_interaction_v1/%A_%a_probe.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_state_eta_interaction_v1.probe run --shard "$SLURM_ARRAY_TASK_ID" --shards 16
