#!/usr/bin/env bash
#SBATCH --job-name=gap-n2-mc
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=diagnostics/gap_flow_v1/logs/%j_n2_mc.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%j_n2_mc.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
root=diagnostics/gap_flow_v1/n2_center
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_raw" --seed 17 --samples-per-step 16
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev_mc16_safety" --seed 17 --samples-per-step 16 --safety
