#!/bin/bash
#SBATCH --job-name=c1-q-fresh-toy-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/physical_state_critic_minimal_v1/logs/fresh_toy_finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.physical_state_critic_minimal_v1.fresh_toy evaluate
printf 'Prospective 48-state selected-only Toy Q16 evaluation complete.\n' > diagnostics/physical_state_critic_minimal_v1/fresh_toy_48/COMPLETE.txt
