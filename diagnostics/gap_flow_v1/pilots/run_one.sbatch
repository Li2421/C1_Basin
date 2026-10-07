#!/usr/bin/env bash
#SBATCH --job-name=gap-flow
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=diagnostics/gap_flow_v1/logs/%A_%a.out
#SBATCH --error=diagnostics/gap_flow_v1/logs/%A_%a.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
flow_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
sizes=(2 10 50)
trains=(48 32 12)
devs=(12 8 4)
tests=(12 8 4)
steps=(6000 6000 8000)
slot="${SLURM_ARRAY_TASK_ID}"
count="${sizes[$slot]}"
root="diagnostics/gap_flow_v1/n${count}"
mkdir -p "$root"
"$flow_python" -m bottleneck_family.flow_dataset --agents "$count" --train "${trains[$slot]}" --dev "${devs[$slot]}" --test "${tests[$slot]}" --seed 20261007 --output "$root/dataset"
"$flow_python" -m bottleneck_family.train_flow --dataset "$root/dataset" --output "$root/train" --steps "${steps[$slot]}" --batch-size 256 --seed 17
"$flow_python" -m bottleneck_family.evaluate_flow --dataset "$root/dataset" --checkpoint "$root/train/best.pkl" --output "$root/eval_dev" --seed 17
