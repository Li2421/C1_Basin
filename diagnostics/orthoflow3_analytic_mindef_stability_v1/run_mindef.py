"""Use the frozen OrthoFlow3 continuation engine in this isolated audit."""
from pathlib import Path
import sys
ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_analytic_mindef_stability_v1'
sys.path.insert(0,str(ROOT))
from diagnostics.orthoflow3_local_basin_continuity_v1 import run_continuity_arms as engine
engine.HERE=HERE
if __name__=='__main__':engine.main()
