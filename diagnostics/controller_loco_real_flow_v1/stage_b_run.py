"""Stage B wrapper around the exact Stage A Flow rollout/merge implementation."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from diagnostics.controller_loco_real_flow_v1 import run

BASE=Path(__file__).resolve().parent
run.HERE=BASE/'stage_b'
run.EXP='exp_controller_loco_real_flow_v1_stage_b'

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['worker','merge','postflight']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=19)
    a=ap.parse_args()
    if a.stage=='worker':
        assert json.loads((run.HERE/'CACHE_INVENTORY.json').read_text())['missing']<=30000
        run.worker(a.index,a.workers)
    else:getattr(run,a.stage)()
