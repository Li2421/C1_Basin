#!/bin/bash
#SBATCH --job-name=c1-q-capacity-final
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=diagnostics/physical_state_critic_minimal_v1/logs/capacity_finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
py=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
for variant in strict enriched; do
  export C1_Q_PILOT_VARIANT="$variant"
  for panel in within_toy within_db within_four within_ring loso_toy loso_db loso_four loso_ring; do
    $py -m diagnostics.physical_state_critic_minimal_v1.pilot capacity --panel "$panel"
  done
  $py -m diagnostics.physical_state_critic_minimal_v1.pilot report
done
printf 'Wide eta-only capacity controls complete for strict and enriched source.\n' > diagnostics/physical_state_critic_minimal_v1/CAPACITY_COMPLETE.txt
