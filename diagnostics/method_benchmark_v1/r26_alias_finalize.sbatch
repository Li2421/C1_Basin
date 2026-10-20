#!/bin/bash
#SBATCH --job-name=r26_alias_finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000M
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhihan/research/Basin_C1/diagnostics/method_benchmark_v1/r26_alias_finalize_%j.log
set -euo pipefail
cd /home/zhihan/research/Basin_C1
PY=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python
"$PY" -B - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path('diagnostics/method_benchmark_v1').resolve()))
from analyze import analyze_test
analyze_test('T2',1)
PY
"$PY" -B diagnostics/method_benchmark_v1/analyze.py
exec "$PY" -B diagnostics/method_benchmark_v1/summarize.py
