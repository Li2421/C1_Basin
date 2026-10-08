#!/bin/bash
#SBATCH --job-name=gap1_n20_postgate_traces
#SBATCH --partition=debug
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_postgate_trace_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
categories=(solo_LR solo_RL five_LR five_RL full_LR full_RL)
category=${categories[$SLURM_ARRAY_TASK_ID]}
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.collect_scale_dagger \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --checkpoint diagnostics/gap_flow_scale_20261007/n20_train_recovery_v5/snapshot_0001000.pkl \
  --output "diagnostics/gap_flow_scale_20261007/n20_postgate_model_${category}" \
  --cases-per-category 4 --case-start 16 --categories "$category" \
  --horizon 10000 --anchors 3000 5000 7000 9000 --seed 20261029 \
  --goal-stop --hold-initially-passive --model-only
