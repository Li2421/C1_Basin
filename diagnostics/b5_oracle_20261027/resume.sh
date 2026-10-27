#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
PLOT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python
EXP_DIR=diagnostics/b5_oracle_20261027
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for stage in smoke A A_deep B C D range E; do
    if [[ ! -f "$EXP_DIR/${stage}_execution.json" ]]; then
        taskset -c 0-17 "$PY" "$EXP_DIR/oracle.py" "$stage" --workers 18
    fi
done
if [[ ! -f "$EXP_DIR/regression.json" ]]; then
    taskset -c 0-17 "$PY" "$EXP_DIR/checks.py"
fi
if [[ ! -f "$EXP_DIR/mechanisms_summary.json" ]]; then
    taskset -c 0-17 "$PY" "$EXP_DIR/mechanisms.py"
fi
taskset -c 0-17 "$PY" "$EXP_DIR/independent.py" --freeze-only
if [[ ! -f "$EXP_DIR/EXPANSION_COMPLETE.json" ]]; then
    taskset -c 0-17 "$PY" "$EXP_DIR/expand.py" --workers 18
fi
if [[ ! -f "$EXP_DIR/supplement_execution.json" ]]; then
    taskset -c 0-17 "$PY" "$EXP_DIR/supplement.py"
fi
if [[ ! -f "$EXP_DIR/independent_execution.json" ]]; then
    taskset -c 0-17 "$PY" "$EXP_DIR/independent.py" --workers 18
fi
taskset -c 0-17 "$PY" "$EXP_DIR/legacy.py"
taskset -c 0-17 "$PY" "$EXP_DIR/legacy_prior.py"
if [[ ! -f "$EXP_DIR/legacy_mechanisms_summary.json" ]]; then
    taskset -c 0-17 "$PY" "$EXP_DIR/legacy_mechanisms.py"
fi
taskset -c 0-17 "$PY" "$EXP_DIR/safety_audit.py"
for experiment_uid in exp_b5_oracle_20261027 exp_b5_legacy_20261027; do
    taskset -c 0-17 "$PY" -m shared_rollout_db.src.merger --experiment "$experiment_uid"
done
taskset -c 0-17 "$PY" "$EXP_DIR/analyze.py"
taskset -c 0-17 "$PLOT_PY" "$EXP_DIR/plot.py"
taskset -c 0-17 "$PY" "$EXP_DIR/verify.py"
taskset -c 0-17 "$PY" "$EXP_DIR/delivery_manifest.py"
"$PY" "$EXP_DIR/write_report.py"
