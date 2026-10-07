#!/bin/bash
#SBATCH --job-name=c1-goal-phase-plan
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_phase_prepare_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_phase_prepare_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.goal_phase_intervention prepare
task_out=diagnostics/orthoflow3_critic_rootcause_resolution_v1/goal_phase_intervention
"$task_python" -m shared_rollout_db.plan --manifest "$task_out/planned_rollouts.json" --output "$task_out/cache_preflight.json"
"$task_python" -m shared_rollout_db.plan --manifest "$task_out/planned_parent_reuse.json" --output "$task_out/parent_cache_preflight.json"
