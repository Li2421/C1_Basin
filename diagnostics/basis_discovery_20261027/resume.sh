#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    C1_SCRIPT="$(realpath "$0")"
    cd "$(dirname "$C1_SCRIPT")/../.."
    exec sbatch -p cpu --cpus-per-task="${C1_CPUS:-24}" --mem=32G --time=02:00:00 --job-name=c1_dict_resume --output=diagnostics/basis_discovery_20261027/resume_slurm.log "$C1_SCRIPT"
fi
cd "${SLURM_SUBMIT_DIR:-/home/zhihan/research/Basin_C1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
PLOT_PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python
EXP_DIR=diagnostics/basis_discovery_20261027
"$PY" "$EXP_DIR/checks.py"
"$PY" "$EXP_DIR/screen.py" initial --workers "$SLURM_CPUS_PER_TASK"
if [[ ! -f "$EXP_DIR/static_states.json" ]]; then "$PY" "$EXP_DIR/static_audit.py"; fi
"$PY" "$EXP_DIR/screen.py" expand --workers "$SLURM_CPUS_PER_TASK"
"$PY" "$EXP_DIR/matched_controls.py"
"$PY" "$EXP_DIR/screen.py" delete --workers "$SLURM_CPUS_PER_TASK"
if [[ ! -f "$EXP_DIR/MINIMAL_FAMILIES.json" ]]; then "$PY" "$EXP_DIR/screen.py" minimal; fi
# Frozen first holdout must be loaded before later compression changes its support.
"$PY" "$EXP_DIR/holdout.py" all
"$PY" "$EXP_DIR/compression.py"
"$PY" "$EXP_DIR/restarts.py"
"$PY" "$EXP_DIR/confirmatory.py"
"$PY" "$EXP_DIR/pure_family.py"
"$PY" "$EXP_DIR/final_baselines.py"
"$PY" "$EXP_DIR/mechanism_validation.py"
"$PY" "$EXP_DIR/baseline_checks.py"
"$PY" "$EXP_DIR/baselines.py"
"$PY" "$EXP_DIR/baseline_analysis.py"
"$PY" "$EXP_DIR/analyze.py"
"$PLOT_PY" "$EXP_DIR/plot.py"
"$PY" "$EXP_DIR/verify.py"
"$PY" "$EXP_DIR/write_report.py"
