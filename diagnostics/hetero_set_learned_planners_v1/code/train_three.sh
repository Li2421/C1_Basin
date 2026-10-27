#!/bin/bash
# usage: train_three.sh DATA_DIRS(comma) TAG STEPS_MLP STEPS_ATTENTION STEPS_DIFFUSION SEED
# Trains the three learned planners concurrently, 6 cores each (18 total).
cd "$(dirname "$0")"
P=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6"
mkdir -p logs
taskset -c 0-5   $P train_bc.py mlp       "$1" "ckpt/$2_mlp_s$6"       "$6" "$3" > "logs/$2_mlp_s$6.log" 2>&1 &
taskset -c 6-11  $P train_bc.py attention "$1" "ckpt/$2_attention_s$6" "$6" "$4" > "logs/$2_attention_s$6.log" 2>&1 &
taskset -c 12-17 $P train_bc.py diffusion "$1" "ckpt/$2_diffusion_s$6" "$6" "$5" > "logs/$2_diffusion_s$6.log" 2>&1 &
wait
grep -h -E "step|saved|Error|Traceback" "logs/$2_mlp_s$6.log" "logs/$2_attention_s$6.log" "logs/$2_diffusion_s$6.log" | tail -n 30
