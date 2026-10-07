#!/usr/bin/env bash
#SBATCH --job-name=gap-scaled-wide
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=diagnostics/gap_flow_v1/logs/%A_%a_scaled.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%A_%a_scaled.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
sizes=(10 50)
trains=(48 24)
devs=(12 8)
tests=(12 8)
anchors=(4 3)
steps=(10000 20000)
slot="$SLURM_ARRAY_TASK_ID"
count="${sizes[$slot]}"
root="diagnostics/gap_flow_v1/n${count}_wide_recovery"
mkdir -p "$root"
"$flow_python" -m bottleneck_family.flow_dataset --agents "$count" --door-width 0.62 --train "${trains[$slot]}" --dev "${devs[$slot]}" --test "${tests[$slot]}" --recovery-anchors "${anchors[$slot]}" --seed 20261007 --output "$root/dataset"
"$flow_python" -m bottleneck_family.train_flow --dataset "$root/dataset" --output "$root/train" --steps "${steps[$slot]}" --batch-size 256 --seed 17
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_raw" --seed 17 --samples-per-step 16
if [[ "$count" == 10 ]]; then
  "$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_safety" --seed 17 --samples-per-step 16 --safety
fi
