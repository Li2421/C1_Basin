#!/bin/bash
#SBATCH --job-name=gap1_n50_v10
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_train_gate_terminal_v10_slurm_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n50_gate_terminal_dataset_v10/dataset \
  --output diagnostics/gap_flow_scale_20261007/n50_train_gate_terminal_v10 \
  --init-checkpoint diagnostics/gap_flow_scale_20261007/n50_train_goal_v8_retry40g/best.pkl \
  --steps 30000 --snapshot-interval 5000 --batch-size 64 --seed 9070 \
  --near-goal-fraction 0.30 --early-transition-fraction 0.35 \
  --early-steps 2500 --early-nominal-only --source-balanced-sampling \
  --motion-loss-weight 20 --permutation-augmentation \
  --shared-agent-normalization --active-action-normalization \
  --architecture set_attention --set-width 64 --set-layers 1
