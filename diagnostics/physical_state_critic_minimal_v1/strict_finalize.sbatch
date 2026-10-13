#!/bin/bash
#SBATCH --job-name=c1-q-cache-strict-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/physical_state_critic_minimal_v1/logs/strict_finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export C1_Q_PILOT_VARIANT=strict
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
for panel in within_toy within_db within_four within_ring loso_toy loso_db loso_four loso_ring; do
  $py -m diagnostics.physical_state_critic_minimal_v1.pilot predict --panel "$panel"
done
for panel in within_toy within_db within_four within_ring loso_toy loso_db loso_four loso_ring; do
  $py -m diagnostics.physical_state_critic_minimal_v1.pilot evaluate --panel "$panel"
done
$py -m diagnostics.physical_state_critic_minimal_v1.pilot report
printf 'All eight strict-source cache-only panels complete.\n' > diagnostics/physical_state_critic_minimal_v1/strict/COMPLETE.txt
