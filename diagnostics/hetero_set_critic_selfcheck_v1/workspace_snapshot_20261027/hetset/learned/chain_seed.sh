#!/bin/bash
# Same DAgger chain as chain.sh for an extra seed (own round-0 training, own DAgger data):
#   train round 0 -> DAgger round 1 -> train round 1 -> DAgger round 2 -> train round 2
# usage: chain_seed.sh KIND CORES STEPS_PER_ROUND FINETUNE(1/0) SEED
cd "$(dirname "$0")"
K=$1; C=$2; S=$3; FT=$4; SD=$5
P=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
export JAX_PLATFORMS=cpu
TRAINFLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6"
[ -f ckpt/r0_${K}_s$SD/meta.json ] || XLA_FLAGS="$TRAINFLAGS" taskset -c $C $P train_bc.py $K data/r0 ckpt/r0_${K}_s$SD $SD $S > logs/r0_${K}_s$SD.log 2>&1
DATA=data/r0
for R in 1 2; do
  PREV=ckpt/r$((R-1))_${K}_s$SD
  if [ ! -f data/r${R}_${K}_s$SD/learner_outcomes.json ]; then
    (cd .. && LWORKERS=6 HS_FAST=1 taskset -c $C $P learned/dagger.py learned/$PREV learned/data/r${R}_${K}_s$SD) > logs/dagger_r${R}_${K}_s$SD.log 2>&1
  fi
  DATA=$DATA,data/r${R}_${K}_s$SD
  if [ ! -f ckpt/r${R}_${K}_s$SD/meta.json ]; then
    INIT=""; [ "$FT" = 1 ] && INIT=$PREV
    XLA_FLAGS="$TRAINFLAGS" INIT_CKPT=$INIT taskset -c $C $P train_bc.py $K $DATA ckpt/r${R}_${K}_s$SD $SD $S > logs/r${R}_${K}_s$SD.log 2>&1
  fi
  echo "$K seed $SD round $R: $(grep -o 'done.*' logs/dagger_r${R}_${K}_s$SD.log | tail -1) | $(grep saved logs/r${R}_${K}_s$SD.log)"
done
