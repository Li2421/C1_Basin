#!/usr/bin/env bash
set -euo pipefail
audit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
audit_python="${AUDIT_PYTHON:-/tmp/c1-transfer-audit-env/bin/python}"
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Read-only reuse of rollout/result/probe caches. The final action-only probe
# invokes frozen controller forward passes on four preregistered initial cases.
# It performs no simulator step, rollout, training or GPU request.
"$audit_python" "$audit_dir/extract_cached.py"
"$audit_python" "$audit_dir/score_frozen.py"
"$audit_python" "$audit_dir/eta_prior_baseline.py"
"$audit_python" "$audit_dir/stratified_eta_check.py"
"$audit_python" "$audit_dir/event_attribution.py"
"$audit_python" "$audit_dir/trajectory_mechanism.py"
"$audit_python" "$audit_dir/action_decomposition.py"
"$audit_python" "$audit_dir/four_chain_interaction.py"
"$audit_python" "$audit_dir/make_figures.py"
"$audit_python" "$audit_dir/horizon_ablation.py"
"$audit_python" "$audit_dir/field_substep_probe.py"
"$audit_python" "$audit_dir/verify.py"
"$audit_python" "$audit_dir/make_cache_index.py"
