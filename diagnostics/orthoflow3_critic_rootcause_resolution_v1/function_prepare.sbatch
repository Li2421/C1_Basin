#!/bin/bash
#SBATCH --job-name=c1-function-preflight
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --output=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_prepare_%j.out
#SBATCH --error=diagnostics/orthoflow3_critic_rootcause_resolution_v1/function_prepare_%j.err
set -euo pipefail
cd /home/zhihan/research/Basin_C1
export PYTHONPATH=/home/zhihan/research/Basin_C1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
task_python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
task_out=diagnostics/orthoflow3_critic_rootcause_resolution_v1/controller_function_support
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_support prepare
"$task_python" -m shared_rollout_db.plan --manifest "$task_out/planned_rollouts.json" --output "$task_out/cache_preflight.json"
for controller_name in alt second extra_a extra_b function_0 function_1 function_2 function_3 function_4 function_5 function_6 function_7; do
  "$task_python" -m shared_rollout_db.plan --manifest "$task_out/planned_$controller_name.json" --output "$task_out/cache_preflight_$controller_name.json"
done
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_support validate
"$task_python" -m diagnostics.orthoflow3_critic_rootcause_resolution_v1.function_support filter
