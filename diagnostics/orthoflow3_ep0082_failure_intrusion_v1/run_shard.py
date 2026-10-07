#!/usr/bin/env python3
"""Only redirects the frozen, previously audited rollout driver to this task's files."""
from core import *
import argparse,importlib.util
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
 tasks=[json.loads(l) for l in open(H/f'plans/{a.plan}/shard{a.shard}.jsonl')]
 assert all(r['state_id']==SID and inside((np.array(r['eta'])-AFF)/SCALE)[0] for r in tasks)
 assert sha(D/'double_bottleneck_eta_basis_redesign/tools/bases.py')==SHA
 spec=importlib.util.spec_from_file_location('frozen_geometry_rollout',G/'run_rollout_shard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.HERE=H;m.main()
