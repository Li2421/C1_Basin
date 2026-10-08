#!/bin/bash
#SBATCH --job-name=gap1_n50_v19_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_scale_20261007/n50_merge_v19_slurm_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
shards=()
for index in $(seq -w 0 11); do
  shards+=(--shard "diagnostics/gap_flow_scale_20261007/n50_v16_train_recovery/shard_${index}/dataset")
done
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.merge_n50_recovery \
  --source diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v13/dataset \
  "${shards[@]}" \
  --output diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v19/dataset
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m bottleneck_family.audit_n50_dataset \
  --dataset diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v19/dataset \
  --output diagnostics/gap_flow_scale_20261007/n50_recovery_dataset_v19/phase_audit.json
