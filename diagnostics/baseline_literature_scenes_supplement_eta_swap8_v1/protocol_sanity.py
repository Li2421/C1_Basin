"""Static request checks only, no actor execution or label generation."""
import bootstrap
from experiment import *
from registry import verify_receipt
from swap_geometry import moving_path
from collections import Counter
import numpy as np

def main():
    verify_receipt('IMPLEMENTATION_FROZEN.receipt.json');verify_receipt('PREREGISTRATION.receipt.json');verify_receipt('WRITE_PATH_READY.receipt.json')
    cases=read(HERE/'capability_CASES.json');counts=Counter((c['state']['scene'],c['controller_payload']['nominal']['controller']) for c in cases);assert len(counts)==144 and set(counts.values())=={512}
    assert len({c['rollout_uid'] for c in cases})==73728
    sample={}
    for c in cases:
        s=c['state'];key=(s['parent_physical_identity'],s['agent'],c['controller_payload']['nominal']['controller'],c['seed'])
        if s['scene']=='lit_swap8_free':sample[key]=c['rng_state_uid']
        elif s['scene']=='lit_swap8_moving':assert c['rng_state_uid']==sample[key]
    del cases
    sel=read(HERE/'HISTORY_SELECTION.json');assert sel['lit_circle_five']['missing_types']==['centralized_mppi']
    eta=read(ROOT/'diagnostics/controller_loco_real_flow_v1/protocol.json')['eta'];fixed=read(HERE/'FIXED_ETA.json')
    assert all(np.asarray(eta[k],np.float64).tobytes()==np.asarray(fixed[str(k)]['values'],np.float64).tobytes() for k in (10,13))
    write(HERE/'PROTOCOL_SANITY.json',dict(passed=True,capability_requests=73728,per_scene_controller_requests=512,unique_uids=True,free_moving_rng_matched=True,eta10_eta13_exact=True,new_rollouts=0))
    print('Protocol sanity passed',flush=True)
if __name__=='__main__':main()
