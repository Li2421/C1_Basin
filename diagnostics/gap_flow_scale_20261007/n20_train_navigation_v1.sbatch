#!/bin/bash
#SBATCH --job-name=gap1_n20_flow_train
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.train_flow \
  --dataset diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --output diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1 \
  --steps 20000 --snapshot-interval 2000 --batch-size 128 \
  --seed 20261020 --near-goal-fraction 0.25 --near-goal-distance 1.0 \
  --early-transition-fraction 0.25 --early-steps 100 \
  --motion-loss-weight 20 --permutation-augmentation \
  --shared-agent-normalization --active-action-normalization \
  --architecture set_attention --set-width 64 --set-layers 1
