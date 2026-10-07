#!/bin/bash
#SBATCH --job-name=c1-db-stage
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_stage_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/db_stage_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
case "$1" in
  prepare)
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train materialize
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train unit
    ;;
  freeze)
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train freeze
    ;;
  evaluate)
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_evaluate predict
    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_evaluate evaluate
    ;;
  *) exit 2;;
esac
