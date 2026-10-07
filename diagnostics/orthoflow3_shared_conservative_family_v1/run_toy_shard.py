#!/usr/bin/env python3
"""Redirect the frozen audited Toy Q64 runner to this task's plans/results."""
import argparse,importlib.util,json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;D=HERE.parent;G=D/'orthoflow3_general_basin_geometry_v1'
AFF=np.array([.875,0,.375]);SCALE=np.array([.75,1,.75]);HS=np.unique(np.array(json.load(open(D/'orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'))['halfspaces']),axis=0)
SHA='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();tasks=[json.loads(l) for l in open(HERE/f'plans/{a.plan}/shard{a.shard}.jsonl')]
 assert all(r['state_id'].startswith('T0_WIDE_') and np.all(((np.array(r['eta'])-AFF)/SCALE)@HS[:,:3].T+HS[:,3]<=1e-10) for r in tasks);assert sha(D/'double_bottleneck_eta_basis_redesign/tools/bases.py')==SHA
 spec=importlib.util.spec_from_file_location('frozen_geometry_rollout',G/'run_rollout_shard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=HERE;m.main()
