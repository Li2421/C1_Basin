"""Retain which agent receives which controller response, without slot IDs.

Replays exactly the existing H20 input probes (not task continuations). Verify
the old24 aggregate features numerically before accepting new per-agent
signed responses. New features have no task outcomes or termination labels.
"""
import argparse,copy,os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np
from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import read,write,SRC,OUT as AUDIT

OUT=Path(__file__).resolve().parent/'agent_response'

def instrument(rt):
    reset=rt.core.reset;project=rt.core.project;trace=dict(runs=[],pending=[])
    def wrapped_project(env,action):
        safe=project(env,action);trace['pending'].append((np.asarray(action).copy(),np.asarray(safe).copy()));return safe
    def wrapped_reset(physical):
        env=reset(physical);step=env.step;run=[];trace['runs'].append(run);trace['pending']=[]
        delta=np.asarray(physical['goals'])-np.asarray(physical['positions']);ex=delta/np.linalg.norm(delta,axis=-1,keepdims=True)
        ey=np.stack([-ex[:,1],ex[:,0]],axis=-1);v=rt.core.cfg.max_speed
        def local(a):return np.stack([(a*ex).sum(-1),(a*ey).sum(-1)],axis=-1)/v
        def wrapped_step(action):
            assert 1<=len(trace['pending'])<=2
            flow,safe=trace['pending'][0];executed=np.asarray(action)
            feat=np.concatenate([local(flow),local(safe),local(executed),
                (np.linalg.norm(safe-flow,axis=-1)/v)[:,None],(np.linalg.norm(executed-safe,axis=-1)/v)[:,None]],axis=-1)
            run.append(feat);trace['pending']=[];return step(action)
        env.step=wrapped_step;return env
    rt.core.reset=wrapped_reset;rt.core.project=wrapped_project
    return trace

def build(controller):
    target=OUT/f'features_{controller}.npz'
    if target.exists():raise FileExistsError('Frozen per-agent response exists')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':20}
    profiles=read(SRC/'protocol.json')['profiles'];profile=next(p for p in profiles if p['name']==controller)
    rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
    states=read(AUDIT/'source_states.json');physical={r['state_uid']:r['physical'] for r in read(SRC/'physical.json')}
    data=np.load(AUDIT/'source_data_db.npz');ci=('alt','second').index(controller)
    exact_etas={(p['state_uid'],p['eta_index']):p['eta'] for p in read(SRC/'pairs.json')}
    values=[];indices=[];errors=[];max_error=0.
    for si,s in enumerate(states):
      for ei in (10,15):
        index=si*16+ei;trace['runs']=[]
        # Stored training eta is float32 while original task eta is exact64.
        # Replay the exact original eta from the manifest, never rounded eta.
        exact=exact_etas[(s['uid'],ei)]
        trace['runs']=[];result=rt.features(physical[s['uid']],exact)
        error=float(np.max(abs(np.asarray(result['mean'])-data['context'][ci,index])))
        # CPU/GPU float32 Flow accumulation is not bitwise identical. The
        # observed few-e-6 descriptor residual is logged; this tolerance is
        # for derived inputs only and never relaxes rollout/cache identity.
        np.testing.assert_allclose(result['mean'],data['context'][ci,index],rtol=0,atol=1e-5)
        assert len(trace['runs'])==4 and all(len(r)==20 for r in trace['runs'])
        summaries=np.array([np.array(r).mean(0) for r in trace['runs']])
        agent=np.concatenate([summaries[[0,2]].mean(0),summaries[[1,3]].mean(0)],axis=-1)
        values.append(agent);indices.append(index);errors.append(error);max_error=max(max_error,error)
      if len(values)%16==0:print(dict(controller=controller,pairs=len(values),aggregate_replay_max_error=max_error),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,indices=np.array(indices),agent_response=np.array(values,np.float32),aggregate_replay_error=np.array(errors))
    write(OUT/f'audit_{controller}.json',dict(pairs=len(indices),agent_channels=16,max_aggregate_replay_error=max_error,
        labels_read=False,new_task_rollouts=0,eta_exact_serialization_preserved=True,
        definition='For nominal and eta-conditioned H20: per-agent mean raw Flow(2), safe action(2), executed action(2) in initial goal frames; safety and correction magnitudes(1 each). Mean over the same two frozen probe RNG streams.',
        interpretation='Preserves response-to-entity association; no controller slot ID or scene ID'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('controller',choices=('alt','second'));a=p.parse_args();build(a.controller)
