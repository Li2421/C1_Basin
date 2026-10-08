#!/bin/bash
#SBATCH --job-name=gap1_n20_flow_postgate
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_train_postgate_v7_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
: "${GAP1_INIT_CHECKPOINT:?choose from complete non-opposing v5 screen}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_merged_v7/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_train_postgate_v7 \
  --init-checkpoint "$GAP1_INIT_CHECKPOINT" \
  --steps 8000 --snapshot-interval 1000 --batch-size 128 --seed 20261024 \
  --early-transition-fraction 0.05 --early-steps 2000 --early-nominal-only \
  --near-goal-fraction 0.10 --near-goal-distance 1.0 \
  --onset-recovery-fraction 0.10 --onset-recovery-steps 150 \
  --terminal-recovery-fraction 0.10 --terminal-recovery-steps 150 \
  --gate-recovery-fraction 0.05 --gate-recovery-steps 300 \
  --late-postgate-recovery-fraction 0.20 --late-postgate-recovery-steps 300 \
  --postgate-fraction 0.10 --postgate-min-goal-distance 1.0 \
  --source-balanced-sampling --motion-loss-weight 20 \
  --permutation-augmentation --shared-agent-normalization \
  --active-action-normalization --architecture set_attention \
  --set-width 64 --set-layers 1
