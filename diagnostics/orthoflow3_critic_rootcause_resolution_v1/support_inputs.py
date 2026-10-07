"""Source-side physical inputs and matched H20 aggregate/entity responses."""
import argparse,json,os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np
from . import state_support as design
from .agent_response import instrument
from diagnostics.orthoflow3_controller_training_repair_v1 import data as native
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc

OUT=design.OUT;read=design.read;write=design.write

def physical():
    design.configure();design.engine.configure();native.physical()
    assert len(read(OUT/'physical.json'))==256

def features(controller):
    target=OUT/f'inputs_{controller}.npz'
    if target.exists():raise FileExistsError('Source response inputs already frozen')
    import jax
    assert jax.default_backend()=='gpu'
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20}
    profile=next(p for p in read(OUT/'protocol.json')['profiles'] if p['name']==controller)
    rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
    physical=read(OUT/'physical.json');pairs=read(OUT/'pairs.json')
    contexts=np.zeros((len(pairs),24),np.float32);responses=np.zeros((len(pairs),4,16),np.float32)
    valid=np.zeros(len(pairs),bool);errors=[]
    for i,p in enumerate(pairs):
        assert p['split'] in ('train','validation'),'Confirmation is sealed'
        state=physical[p['state_index']];assert state['state_uid']==p['state_uid']
        trace['runs']=[]
        try:
            result=rt.features(state['physical'],p['eta'])
            assert len(trace['runs'])==4 and all(len(r)==20 for r in trace['runs'])
            means=np.array([np.array(r).mean(0) for r in trace['runs']])
            contexts[i]=result['mean'];responses[i]=np.concatenate([means[[0,2]].mean(0),means[[1,3]].mean(0)],axis=-1)
            assert np.isfinite(contexts[i]).all() and np.isfinite(responses[i]).all()
            valid[i]=True
        except Exception as exc:
            errors.append(dict(pair_index=i,state_uid=p['state_uid'],error=f'{type(exc).__name__}: {exc}'))
        if (i+1)%32==0:print(dict(controller=controller,pairs=i+1,invalid=len(errors)),flush=True)
    np.savez_compressed(target,context=contexts,agent_response=responses,valid=valid)
    write(OUT/f'input_audit_{controller}.json',dict(pairs=len(pairs),invalid=errors,
        confirmation_labels_or_inputs_used=False,new_task_rollouts=0,horizon=20,
        behavior='Invalid response inputs flagged, never converted to task failure/success; confirmation inference requires explicit fallback accounting.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('physical','features'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='physical':physical()
    else:features(('alt','second')[a.index])
