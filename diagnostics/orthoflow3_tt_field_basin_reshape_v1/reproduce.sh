#!/usr/bin/env bash
set -euo pipefail
analysis_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
analysis_python="${BASIN_ANALYSIS_PYTHON:-/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python}"
plot_python="${BASIN_PLOT_PYTHON:-/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python}"
if [[ "${1:-}" == "--refresh-from-cache" ]]; then
  "$analysis_python" "$analysis_dir/analyze.py" export
elif [[ $# -gt 0 ]]; then
  echo 'Usage: reproduce.sh [--refresh-from-cache]' >&2
  exit 2
fi
"$analysis_python" "$analysis_dir/test_analysis.py" >"$analysis_dir/test_results.txt" 2>&1
"$analysis_python" "$analysis_dir/analyze.py" analyze
"$plot_python" "$analysis_dir/plot.py"
"$analysis_python" "$analysis_dir/verify.py"
"$analysis_python" "$analysis_dir/report.py"
