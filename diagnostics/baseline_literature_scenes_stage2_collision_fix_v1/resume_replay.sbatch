#!/bin/bash
#SBATCH --job-name=lit_resume_replay
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_stage2_collision_fix_v1/resume_replay_%A_%a.log
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
for ((task_index=SLURM_ARRAY_TASK_ID;task_index<32;task_index+=RESUME_REPLAY_WORKERS)); do
 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/baseline_literature_scenes_stage2_collision_fix_v1/resume_replay.py "$task_index"
done
