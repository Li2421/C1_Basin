#!/bin/bash
#SBATCH --job-name=c1-ring-held-flow
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_v11_flow_%j.out
#SBATCH --error=diagnostics/orthoflow3_controller_information_probe_v1/held_controller_v11_flow_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python - <<'PY'
from new_benchmark_common.training import train_stage1
from pathlib import Path
source = Path('diagnostics/ring_exchange_stage1/base_u_v10_local_dataset')
output = Path('diagnostics/orthoflow3_controller_information_probe_v1/held_controller_v11_flow')
result = train_stage1(source, output, seed=88123, steps=100000, batch_size=256,
                      log_interval=250, validation_batches=8,
                      early_transition_fraction=.5, early_steps=15,
                      early_nominal_only=False, source_balanced_sampling=False)
print(result, flush=True)
PY
