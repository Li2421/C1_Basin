"""Fresh held-controller confirmation for source-selected goal response."""
import argparse,time
import numpy as np
from . import function_confirmation as parent
from .goal_response import OUT as SOURCE,RULE,measurement,read,write,sha


def configure(replica):
    assert replica in (0,1)
    parent.SOURCE=SOURCE;parent.RULE=RULE
    parent.TARGET_SEED=88130+replica;parent.FAMILY_SEED=940071000+replica*10000
    parent.OUT=SOURCE.parent/('goal_response_confirmation' if replica==0 else 'goal_response_confirmation_b')
    parent.FLOW=SOURCE/f'target_flow_seed{parent.TARGET_SEED}'
    parent.EXP=f'exp_orthoflow3_goal_response_confirmation{parent.TARGET_SEED}_v1'
    parent.configure()
    return parent.OUT


def goal_inputs(index):
    import jax
    from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
    assert jax.default_backend()=='gpu'
    profiles=read(parent.OUT/'protocol.json')['profiles']+read(SOURCE/'protocol.json')['profiles'][:4]
    p=profiles[index];dest=parent.OUT/f'goal_{p["name"]}.npz'
    if dest.exists():raise FileExistsError('Targetderivedinput alreadyfrozen')
    assert sha(p['path'])==p['sha256'];rt=RichRuntime('ring_exchange',p['path'])
    physical=read(parent.OUT/'physical.json');pairs=read(parent.OUT/'pairs.json');values=[];errors=[];times=[]
    for i,pair in enumerate(pairs):
        record=physical[pair['state_index']];assert record['state_uid']==pair['state_uid'];start=time.perf_counter()
        try:values.append(measurement(rt,record['physical'],np.array(pair['eta'],float)))
        except Exception as exc:values.append(np.full(16,np.nan));errors.append(dict(pair=i,error=f'{type(exc).__name__}: {exc}'))
        times.append(time.perf_counter()-start)
    np.savez_compressed(dest,goal_response=np.array(values,np.float32),valid=np.isfinite(values).all(1),seconds=np.array(times))
    write(parent.OUT/f'goal_audit_{p["name"]}.json',dict(invalid=errors,task_labels_read=False,new_task_rollouts=0,no_environment_steps=True,
        controller_sha256=p['sha256'],measurement_code_sha256=sha(SOURCE.parent/'goal_response.py'),models_frozen_sha256=sha(SOURCE/'models_frozen.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('flow','prepare','physical','features','goal','filter','run','dataset','validate','postprocess'));p.add_argument('--index',type=int,default=0);p.add_argument('--replicate',type=int,default=0);a=p.parse_args();configure(a.replicate)
    if a.action=='goal':goal_inputs(a.index)
    elif a.action=='features':parent.features(a.index)
    elif a.action in ('flow','prepare','physical','dataset'):getattr(parent,a.action)()
    elif a.action=='validate':
        from . import state_support
        state_support.OUT=parent.OUT;state_support.validate_plan()
    elif a.action=='postprocess':
        from . import postprocess
        # Engine globals above identify this new experiment; common serial merger
        # andpostflight implementation are unchanged.
        postprocess.run('goal_confirmation')
    elif a.action=='filter':parent.engine.filter_numerical()
    else:parent.engine.run('held',a.index)
