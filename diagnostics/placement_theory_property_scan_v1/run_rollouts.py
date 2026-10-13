"""Paired state-margin and live/frozen-B closed-loop probes, fixed controller."""
import argparse
import copy
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'))
from closed_loop_probe import key_for,margin
from validation_stage2.run_rollouts import make_runtime

def stem(j):
    if j['kind']=='state_margin':return f"{j['uid']}_{j['chain']}_h{j['radius']:g}_s{j['sign']}_o{j['line_offset']:+g}"
    if j['kind']=='nominal_B':return f"{j['uid']}_{j['chain']}_nominal"
    return f"{j['uid']}_{j['chain']}_c{j['candidate_index']}_{j['mode']}"

def simulate(rt,st,j,p):
    state=copy.deepcopy(st)
    if j['kind']=='state_margin':
        d=np.asarray(p['state_position_directions'][st['uid']],float)
        pos=np.asarray(state['physical']['positions'],float)
        state['physical']['positions']=(pos+j['radius']*j['sign']*d).tolist()
    env=rt.make_env(state)
    ref=None
    if j['kind']=='basis_feedback':
        path=HERE/'rollout_results'/(f"{j['uid']}_{j['chain']}_nominal.npz")
        if not path.exists():raise RuntimeError(f'Nominal B reference missing: {path}')
        with np.load(path) as z:ref=z['basis'].copy()
    pp=[env.positions.copy()];vv=[env.velocities.copy()];aa=[];kk=[];gg=[];ss=[];bb=[];refgaps=[]
    term='truncated';error=None
    for t in range(rt.config.max_steps):
        if ref is not None and j['mode']=='frozen_B' and t>=len(ref):
            term='reference_exhausted';break
        key=key_for(st['uid'],p['seed'],t,p['future_root'])
        try:
            obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
            live=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,axis=-1).reshape(-1,3)
            use=ref[t] if ref is not None and j['mode']=='frozen_B' else live
            delta=(use@np.asarray(j['eta'],float)).reshape(safe.shape)
            if j['chain']=='TT':a,_=safety.project(safe+delta)
            else:a,_=rt.integrate(env,obs,key,delta,safety)
            safety.assert_safe(a)
            _,_,done,info=env.step(a)
        except Exception as exc:
            term='numerical_failure';error=dict(step=t,type=type(exc).__name__,message=str(exc)[:1000]);break
        goal,safety_margin=margin(st['scenario'],env,info)
        pp.append(env.positions.copy());vv.append(env.velocities.copy());aa.append(a.copy());kk.append(np.asarray(key))
        gg.append(goal);ss.append(safety_margin);bb.append(live.copy())
        if ref is not None and t<len(ref):
            refgaps.append(float(np.linalg.norm((live-ref[t])@np.asarray(j['eta'],float))))
        if done:term=info['termination'];break
    goal=float(np.max(gg)) if gg else None
    saf=float(np.min(ss)) if ss else None
    rho=min(goal,saf) if goal is not None and saf is not None else None
    if rho is None:active=None
    elif goal<=saf:active=dict(kind='goal',step=int(np.argmax(gg))+1)
    else:active=dict(kind='safety',step=int(np.argmin(ss))+1)
    success=(bool(env.summary()['collision_free_success'] and term=='success' and error is None)
             if term not in ('truncated','reference_exhausted','numerical_failure') else
             (False if term=='numerical_failure' else None))
    arrays=dict(positions=np.asarray(pp),velocities=np.asarray(vv),actions=np.asarray(aa),
        noise_keys=np.asarray(kk),goal_margins=np.asarray(gg),safety_margins=np.asarray(ss),
        basis=np.asarray(bb),basis_delta_gaps=np.asarray(refgaps))
    meta=dict(uid=st['uid'],scene=st['scenario'],kind=j['kind'],chain=j['chain'],
        request_state_uid=j['request_state_uid'],request_controller_uid=j['request_controller_uid'],
        eta=j['eta'],rho=rho,success=success,termination=term,error=error,steps=len(aa),
        active_margin=active,goal_ever_margin=goal,safety_always_margin=saf,
        basis_delta_gap_mean=float(np.mean(refgaps)) if refgaps else None,
        basis_delta_gap_max=float(max(refgaps)) if refgaps else None,
        nominal_reference_steps=len(ref) if ref is not None else None)
    return arrays,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=('state_margin','nominal_B','basis_feedback'),required=True)
    ap.add_argument('--protocol',default='rollout_protocol.json');ap.add_argument('--preflight',default='cache_preflight.json')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args();p=json.loads((HERE/'protocol.json').read_text())
    rp=json.loads((HERE/args.protocol).read_text())
    pre=json.loads((HERE/args.preflight).read_text())
    if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(rp['jobs']):raise RuntimeError('preflight mismatch')
    valid={(r['state_uid'],r['eta_uid'],r['controller_uid']) for r in pre['details'] if r['missing_seeds']}
    states={s['uid']:s for s in p['states']};jobs=[j for j in rp['jobs'] if j['kind']==args.kind]
    if args.shards==4:
        jobs=[j for j in jobs if (0 if j['scene']=='ring_exchange' else 2)+(int(j['uid'][-1])%2)==args.shard]
    else:jobs=[j for i,j in enumerate(jobs) if i%args.shards==args.shard]
    runtime={};dest=HERE/'rollout_results';dest.mkdir(exist_ok=True)
    for j in jobs:
        if (j['request_state_uid'],j['eta_uid'],j['request_controller_uid']) not in valid:
            raise RuntimeError('rollout not preflighted '+stem(j))
        meta=dest/(stem(j)+'.json');arr=dest/(stem(j)+'.npz')
        if meta.exists() and arr.exists():continue
        if j['scene'] not in runtime:runtime[j['scene']]=make_runtime(j['scene'])
        arrays,result=simulate(runtime[j['scene']],states[j['uid']],j,p)
        result.update({k:v for k,v in j.items() if k not in result})
        np.savez_compressed(arr,**arrays);meta.write_text(json.dumps(result,indent=2)+'\n')
        print(stem(j),result['termination'],result['rho'],result['steps'],flush=True)

if __name__=='__main__':main()
