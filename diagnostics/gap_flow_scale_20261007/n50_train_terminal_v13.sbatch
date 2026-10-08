#!/bin/bash
#SBATCH --job-name=gap1_n50_v13
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_train_terminal_v13_slurm_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v13/dataset \
  --output diagnostics/gap_flow_scale_20261007/n50_train_terminal_v13 \
  --init-checkpoint diagnostics/gap_flow_scale_20261007/n50_train_onset_v12/snapshot_0003000.pkl \
  --steps 8000 --snapshot-interval 1000 --batch-size 64 --seed 9073 \
  --near-goal-fraction 0.20 --early-transition-fraction 0.25 \
  --early-steps 2500 --early-nominal-only \
  --onset-recovery-fraction 0.20 --onset-recovery-steps 50 \
  --terminal-recovery-fraction 0.20 --terminal-recovery-steps 150 \
  --source-balanced-sampling --motion-loss-weight 20 \
  --permutation-augmentation --shared-agent-normalization \
  --active-action-normalization --architecture set_attention \
  --set-width 64 --set-layers 1
