#!/usr/bin/env bash
#SBATCH --job-name=gap-n50-train
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=diagnostics/gap_flow_v1/logs/%j_n50_train.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%j_n50_train.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
root=diagnostics/gap_flow_v1/n50_wide_recovery
"$flow_python" -m bottleneck_family.train_flow --dataset "$root/dataset" --output "$root/train" --steps 20000 --batch-size 256 --seed 17
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_raw" --seed 17 --samples-per-step 16
