"""Recompute genuine h,C for the slot intervention; do not pretend C is equal."""
import os,json
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from . import permutation_outcomes as design
from .agent_response import instrument
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc

def main():
    import jax
    assert jax.default_backend()=='gpu'
    design.configure();design.engine.configure();native.physical()
    out=design.OUT;read=design.read;write=design.write
    if (out/'derived_inputs.npz').exists():raise FileExistsError('Permutation inputs frozen')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20}
    physical=read(out/'physical.json');pairs=read(out/'pairs.json');n=len(pairs)
    context=np.zeros((2,n,24),np.float32);response=np.zeros((2,n,4,16),np.float32)
    for ci,profile in enumerate(read(out/'protocol.json')['profiles']):
        rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
        for i,p in enumerate(pairs):
            trace['runs']=[];f=rt.features(physical[p['state_index']]['physical'],p['eta'])
            means=np.array([np.array(r).mean(0) for r in trace['runs']]);assert means.shape==(4,4,8)
            context[ci,i]=f['mean'];response[ci,i]=np.concatenate([means[[0,2]].mean(0),means[[1,3]].mean(0)],-1)
    np.savez_compressed(out/'derived_inputs.npz',context=context,agent_response=response)
    write(out/'input_audit.json',dict(full_h_and_C_recomputed=True,physical_multiset_fixed=True,
        controller_role_assignment_changed=True,task_labels_used_for_inputs=False,new_task_rollouts=0,
        warning='Agent permutation invariance of h alone is not an exact alias of h+C when recomputed C differs.'))

if __name__=='__main__':main()
