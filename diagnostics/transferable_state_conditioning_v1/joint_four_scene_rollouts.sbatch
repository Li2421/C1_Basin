#!/bin/bash
#SBATCH --job-name=c1-q-joint-test
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --output=diagnostics/transferable_state_conditioning_v1/joint_four_scene_v1/rollout_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRANSFER_Q_FINAL_ROOT=/home/zhihan/research/Basin_C1/diagnostics/transferable_state_conditioning_v1/joint_four_scene_v1/fresh_confirmation
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.transferable_state_conditioning_v1.joint_four_scene_frozen_worker run --index "$SLURM_ARRAY_TASK_ID"
