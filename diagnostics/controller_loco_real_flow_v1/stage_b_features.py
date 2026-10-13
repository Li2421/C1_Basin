"""Stage B extra-source static features via the unchanged Stage A query code."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diagnostics.controller_loco_real_flow_v1 import features
features.HERE=Path(__file__).resolve().parent/'stage_b'
if __name__=='__main__':features.run(int(sys.argv[1]))
