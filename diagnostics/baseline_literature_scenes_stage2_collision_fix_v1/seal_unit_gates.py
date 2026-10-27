from experiment import *
from unit_gates import project,minimum
from states import config
import numpy as np

def main():
    heads=read(HERE/'UNIT_HEAD_ON.json');out={}
    for scene in ('lit_narrow_passage','lit_crowded_random','lit_circle_five'):
        trials=[read(HERE/'unit_snapshots'/f'{scene}_{i}.json') for i in range(10)]
        counts={k:sum(r['counts'][k] for r in trials) for k in trials[0]['counts']}
        head=heads[scene];headpass=head['head_on']['passed'] if scene=='lit_circle_five' else head['gamma'] is not None
        out[scene]=dict(passed=bool(headpass and all(r['passed'] for r in trials)),gamma=head.get('gamma'),counts=counts,
                        head_on_sha256=sha(HERE/'UNIT_HEAD_ON.json'),snapshot_files={str(HERE/'unit_snapshots'/f'{scene}_{i}.json'):sha(HERE/'unit_snapshots'/f'{scene}_{i}.json') for i in range(10)})
    s=read(HERE/'FLOW_STATES_lit_narrow_passage.json')[0];c=config(s['scene']);p=np.asarray(s['positions']);v=np.zeros_like(p)
    a,d=project(p,v,np.asarray(s['goals'])-p,c,.3)
    paper=dict(minimum_clearance=minimum(p,v,a,c),feasible=all(x['maximum_violation']<=1e-7 for x in d))
    theta=np.arange(3)*2*np.pi/3;p=.2*np.column_stack([np.cos(theta),np.sin(theta)]);v=np.zeros_like(p)
    a,d=project(p,v,-p,c,.3)
    symmetric=dict(minimum_clearance=minimum(p,v,a,c),feasible=all(x['maximum_violation']<=1e-7 for x in d))
    write(HERE/'STAGE2_FIX.receipt.json',dict(utc=utc(),scenes=out,narrow_paper=paper,symmetric_three=symmetric,
          passed=all(r['passed'] for r in out.values()),gating_policy='Only individually passed scenes may proceed. Failed random snapshots remain failures; do not silently condition them away.',
          files={str(HERE/n):sha(HERE/n) for n in ('stage2_safety.py','unit_gates.py','snapshot_gate.py','PREREGISTRATION.md')}))
    print({s:(r['passed'],r['counts']) for s,r in out.items()})
if __name__=='__main__':main()
