#!/usr/bin/env bash
#SBATCH --job-name=gap-n10-wide
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/gap_flow_v1/logs/%j_n10_wide.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%j_n10_wide.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
root=diagnostics/gap_flow_v1/n10_wide_recovery
mkdir -p "$root"
"$flow_python" -m bottleneck_family.flow_dataset --agents 10 --door-width 0.62 --train 48 --dev 12 --test 12 --recovery-anchors 4 --seed 20261007 --output "$root/dataset"
"$flow_python" -m bottleneck_family.train_flow --dataset "$root/dataset" --output "$root/train" --steps 10000 --batch-size 256 --seed 17
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_raw" --seed 17 --samples-per-step 16
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_safety" --seed 17 --samples-per-step 16 --safety
