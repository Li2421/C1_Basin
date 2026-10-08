#!/bin/bash
#SBATCH --job-name=gap1_n20_flow_recovery
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_train_recovery_v5_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
: "${GAP1_INIT_CHECKPOINT:?select from complete non-opposing DEV screening first}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_merged_v5/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_train_recovery_v5 \
  --init-checkpoint "$GAP1_INIT_CHECKPOINT" \
  --steps 8000 --snapshot-interval 1000 --batch-size 128 --seed 20261022 \
  --early-transition-fraction 0.20 --early-steps 2000 --early-nominal-only \
  --near-goal-fraction 0.20 --near-goal-distance 1.0 \
  --onset-recovery-fraction 0.20 --onset-recovery-steps 150 \
  --terminal-recovery-fraction 0.15 --terminal-recovery-steps 150 \
  --gate-recovery-fraction 0.15 --gate-recovery-steps 300 \
  --source-balanced-sampling --motion-loss-weight 20 \
  --permutation-augmentation --shared-agent-normalization \
  --active-action-normalization --architecture set_attention \
  --set-width 64 --set-layers 1
