#!/usr/bin/env bash
#SBATCH --job-name=gap-n2-waypoint
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_v1/logs/%j_n2_center.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%j_n2_center.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
root=diagnostics/gap_flow_v1/n2_center
mkdir -p "$root"
"$flow_python" -m bottleneck_family.flow_dataset --agents 2 --train 48 --dev 12 --test 12 --seed 20261007 --output "$root/dataset"
"$flow_python" -m bottleneck_family.train_flow --dataset "$root/dataset" --output "$root/train" --steps 6000 --batch-size 256 --seed 17
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_raw" --seed 17
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_safety" --seed 17 --safety
