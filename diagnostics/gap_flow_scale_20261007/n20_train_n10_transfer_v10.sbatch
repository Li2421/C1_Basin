#!/bin/bash
#SBATCH --job-name=gap1_n20_n10_transfer
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_train_n10_transfer_v10_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# The accepted N=10 set-attention actor has count-independent parameter
# shapes and the same eight input/two action features. Copy its actor weights
# into a newly fitted N=20 model, leaving the frozen N=10 file untouched.
# Use the N=10 action-normalization convention for this controlled transfer.
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_merged_v7/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_train_n10_transfer_v10 \
  --transfer-set-checkpoint diagnostics/gap_flow_scale_20261007/frozen/gap1_n10_flow_navigation_v1.pkl \
  --steps 20000 --snapshot-interval 2000 --batch-size 128 --seed 20261027 \
  --early-transition-fraction 0.05 --early-steps 2000 --early-nominal-only \
  --near-goal-fraction 0.10 --near-goal-distance 1.0 \
  --onset-recovery-fraction 0.10 --onset-recovery-steps 150 \
  --terminal-recovery-fraction 0.10 --terminal-recovery-steps 150 \
  --gate-recovery-fraction 0.05 --gate-recovery-steps 300 \
  --late-postgate-recovery-fraction 0.20 --late-postgate-recovery-steps 300 \
  --postgate-fraction 0.10 --postgate-min-goal-distance 1.0 \
  --source-balanced-sampling --motion-loss-weight 10 \
  --permutation-augmentation --shared-agent-normalization \
  --architecture set_attention --set-width 64 --set-layers 1
