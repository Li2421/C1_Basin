#!/bin/bash
#SBATCH --job-name=controller_q_round3_p1_resume
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-23%24
#SBATCH --output=diagnostics/controller_q_transfer_round3_v1/P1_RESUME_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
cd /home/zhihan/research/Basin_C1
if /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -c 'import json, pathlib, sys; p=pathlib.Path("diagnostics/controller_q_transfer_round3_v1")/f"P1_WORKER_{sys.argv[1]}.json"; sys.exit(0 if p.exists() and json.loads(p.read_text())["completed"] else 1)' "$SLURM_ARRAY_TASK_ID"; then
  exit 0
fi
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/controller_q_transfer_round3_v1/p1_run.py worker --index "$SLURM_ARRAY_TASK_ID" --workers 24
