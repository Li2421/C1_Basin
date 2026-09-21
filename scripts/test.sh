#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
# The maintained implementation is the single-integrator baseline.  The
# legacy VMAS tests encode the former PID geometry and are intentionally not
# part of this project health check.
test_python="${C1_PYTHON:-python}"
if [[ -z "${C1_PYTHON:-}" && -x "$PWD/.venv-c1/bin/python" ]]; then
  test_python="$PWD/.venv-c1/bin/python"
fi
"$test_python" -m unittest discover -s single_integrator/tests
