#!/bin/bash
#SBATCH --job-name=gap1_n20_onset_traces
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n20_onset_traces_%A_%a.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_PLATFORMS=cpu
case_start=$((SLURM_ARRAY_TASK_ID * 4))
output="diagnostics/gap_flow_scale_20261007/n20_onset_model_shard_${SLURM_ARRAY_TASK_ID}"
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
  -m bottleneck_family.collect_scale_dagger \
  --source diagnostics/gap_flow_scale_20261007/n20_competence_data_v2/dataset \
  --checkpoint diagnostics/gap_flow_scale_20261007/n20_train_navigation_v1/snapshot_0006000.pkl \
  --output "$output" --categories full_LR full_RL \
  --source-id-substring perm0 --case-start "$case_start" \
  --cases-per-category 4 --horizon 150 --anchors 10 30 100 \
  --seed 20261011 --goal-stop --model-only
