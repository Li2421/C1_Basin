#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=r14a_backup
#SBATCH --array=0-6%3
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/ring_independent_flow_confirmation_v2/backup_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cuda
case "$SLURM_ARRAY_TASK_ID" in
0) run=base_u_v9_local_macflow; seed=3126;;
1|2|3) run=base_u_v2_macflow; seed=$((3104+20000+SLURM_ARRAY_TASK_ID));;
4|5|6) run=base_u_v9_local_macflow; seed=$((3126+20000+SLURM_ARRAY_TASK_ID-3));;
esac
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/ring_independent_flow_confirmation_v2/train_one.py "$run" "$seed" "backup/${run}_${seed}"
