"""Run a bilateral canonical plan with the frozen continuity rollout engine."""
from pathlib import Path
import sys

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=ROOT/'diagnostics/orthoflow3_bilateral_canonical_audit_v1'
sys.path[:0]=[str(ROOT)]
from diagnostics.orthoflow3_local_basin_continuity_v1 import run_continuity_arms as engine

# The engine is the exact fixed-OrthoFlow3, two-projection continuation path
# already integrity-tested in the prior audit.  Only its isolated output root
# changes; no controller/control semantics are substituted.
engine.HERE=HERE
if __name__ == '__main__': engine.main()
