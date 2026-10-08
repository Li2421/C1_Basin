#!/bin/bash
#SBATCH --job-name=gap1_n20_flow_endpoint
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_train_endpoint_v6_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# The initial checkpoint was selected from the complete non-opposing raw
# screen.  This candidate corrects observed post-gate starvation and
# terminal-recovery errors; no opposing state enters training or selection.
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_merged_v5/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_train_endpoint_v6 \
  --init-checkpoint diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1/snapshot_0018000.pkl \
  --steps 8000 --snapshot-interval 1000 --batch-size 128 --seed 20261023 \
  --early-transition-fraction 0.10 --early-steps 2000 --early-nominal-only \
  --near-goal-fraction 0.15 --near-goal-distance 1.0 \
  --onset-recovery-fraction 0.10 --onset-recovery-steps 150 \
  --terminal-recovery-fraction 0.10 --terminal-recovery-steps 150 \
  --gate-recovery-fraction 0.10 --gate-recovery-steps 300 \
  --postgate-fraction 0.25 --postgate-min-goal-distance 1.0 \
  --source-balanced-sampling --motion-loss-weight 20 \
  --endpoint-action-loss-weight 1 --endpoint-motion-weight 4 \
  --permutation-augmentation --shared-agent-normalization \
  --active-action-normalization --architecture set_attention \
  --set-width 64 --set-layers 1
