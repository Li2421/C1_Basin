#!/usr/bin/env bash
#SBATCH --job-name=gphi-cadence
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_cadence_ablation_v1/slurm-%j.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_cadence_ablation_v1/slurm-%j.err

set -euo pipefail

root=/home/zhihan/research/Basin_C1
out="$root/diagnostics/gphi_fixed_cadence_ablation_v1"
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export PYTHONPATH="$root:$out"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.14
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$root"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
for cadence in 4 8 16; do
  "$py" "$out/run_evaluation.py" \
    --cadence "$cadence" --namespace production --device gpu \
    --start-index 0 --stop-index 128
done
