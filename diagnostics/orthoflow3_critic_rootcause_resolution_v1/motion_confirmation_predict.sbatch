#!/bin/bash
#SBATCH --job-name=c1-motion-predict
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_confirmation_predict_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/motion_confirmation_predict_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for replica in 0 1; do
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.motion_confirmation_eval predict --replicate "$replica"
done
