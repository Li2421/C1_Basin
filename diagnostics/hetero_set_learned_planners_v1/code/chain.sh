#!/bin/bash
# DAgger chain for one learned planner on a fixed 6-core set:
#   wait for round-0 model -> DAgger round 1 -> train round 1 -> DAgger round 2 -> train round 2
# usage: chain.sh KIND CORES STEPS_PER_ROUND FINETUNE(1/0)
cd "$(dirname "$0")"
K=$1; C=$2; S=$3; FT=$4
P=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export JAX_PLATFORMS=cpu
TRAINFLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6"
while [ ! -f ckpt/r0_${K}_s0/meta.json ]; do sleep 20; done
DATA=data/r0
for R in 1 2; do
  PREV=ckpt/r$((R-1))_${K}_s0
  if [ ! -f data/r${R}_$K/learner_outcomes.json ]; then
    (cd .. && LWORKERS=6 HS_FAST=1 taskset -c $C $P learned/dagger.py learned/$PREV learned/data/r${R}_$K) > logs/dagger_r${R}_$K.log 2>&1
  fi
  DATA=$DATA,data/r${R}_$K
  if [ ! -f ckpt/r${R}_${K}_s0/meta.json ]; then
    INIT=""; [ "$FT" = 1 ] && INIT=$PREV
    XLA_FLAGS="$TRAINFLAGS" INIT_CKPT=$INIT taskset -c $C $P train_bc.py $K $DATA ckpt/r${R}_${K}_s0 0 $S > logs/r${R}_${K}_s0.log 2>&1
  fi
  echo "$K round $R: $(grep -o 'done.*' logs/dagger_r${R}_$K.log | tail -1) | $(grep saved logs/r${R}_${K}_s0.log)"
done
