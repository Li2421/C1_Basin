#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=xfals2-a
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_toy_db_transfer_falsification_v2/raw/slurm_%A_%a.out
#SBATCH --error=/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_toy_db_transfer_falsification_v2/raw/slurm_%A_%a.err
#SBATCH --array=0-5
set -euo pipefail
cd /home/zhihan/research/Basin_C1/diagnostics/orthoflow3_toy_db_transfer_falsification_v2
exec /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python run_stage_a.py --shard "$SLURM_ARRAY_TASK_ID"
