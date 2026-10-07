#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# The maintained implementation is the single-integrator baseline.  The
# legacy VMAS tests encode the former PID geometry and are intentionally not
# part of this project health check.
if [[ -n "${C1_PYTHON:-}" ]]; then
  candidates=("$C1_PYTHON")
else
  candidates=("$PWD/.venv-c1/bin/python")
  # A sibling Git worktree may carry the shared, ignored virtual environment.
  # Reuse it without copying it into this worktree or hard-coding a machine path.
  while IFS= read -r worktree; do
    candidates+=("$worktree/.venv-c1/bin/python")
  done < <(git worktree list --porcelain 2>/dev/null | sed -n 's/^worktree //p')
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
fi

test_python=""
for candidate in "${candidates[@]}"; do
  if [[ -x "$candidate" ]] && "$candidate" -c \
      'import importlib.util as u; assert all(u.find_spec(x) for x in ("numpy", "scipy", "jax"))' \
      >/dev/null 2>&1; then
    test_python="$candidate"
    break
  fi
done
if [[ -z "$test_python" ]]; then
  echo "No Python with numpy/scipy/jax found; set C1_PYTHON to the maintained environment." >&2
  exit 2
fi

"$test_python" -m unittest discover -s single_integrator/tests
"$test_python" -m unittest discover -s toy_giveway/tests
"$test_python" -m unittest discover -s tests -p 'test_double_bottleneck.py'
