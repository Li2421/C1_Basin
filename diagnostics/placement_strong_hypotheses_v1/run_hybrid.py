"""Full closed-loop action-family swap with a measured safety closure."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'))
from closed_loop_probe import key_for,margin
from validation_stage2.run_rollouts import make_runtime
import numpy as np


def hybrid_action(rt,env,key,eta,mode):
    obs,safety,flow,safe,delta,_=rt.prepare(env,key,np.asarray(eta,float))
    u_tt0,_=safety.project(safe)
    u_tt,_=safety.project(safe+delta)
    u_ff0,_=rt.integrate(env,obs,key,np.zeros_like(delta),safety)
    u_ff,_=rt.integrate(env,obs,key,delta,safety)
    if mode=='FF_base_TT_corr':raw=u_ff0+(u_tt-u_tt0)
    elif mode=='TT_base_FF_corr':raw=u_tt0+(u_ff-u_ff0)
    else:raise ValueError(mode)
    before=safety.certificate(raw)
    executed,_=safety.project(raw)
    after=safety.assert_safe(executed)
    return executed,dict(raw_max_violation=float(before['max_violation']),
                         closure_norm=float(np.linalg.norm(executed-raw)),
                         baseline_gap=float(np.linalg.norm(u_ff0-u_tt0)),
                         donor_correction_norm=float(np.linalg.norm((u_tt-u_tt0) if mode=='FF_base_TT_corr' else (u_ff-u_ff0))),
                         certified_max_violation=float(after['max_violation']))


def simulate_hybrid(rt,state,mode,eta):
    env=rt.make_env(state);p=[env.positions.copy()];v=[env.velocities.copy()];u=[];keys=[];gm=[];sm=[];audit=[]
    term='truncated';error=None
    for step in range(rt.config.max_steps):
        key=key_for(state['uid'],0,step,2026100601)
        try:
            a,diag=hybrid_action(rt,env,key,eta,mode)
            _,_,done,info=env.step(a)
        except Exception as exc:
            term='numerical_failure';error=dict(step=step,type=type(exc).__name__,message=str(exc)[:1000]);break
        g,s=margin(state['scenario'],env,info)
        p.append(env.positions.copy());v.append(env.velocities.copy());u.append(a.copy());keys.append(np.asarray(key));gm.append(g);sm.append(s);audit.append(diag)
        if done:term=info['termination'];break
    goal=float(np.max(gm)) if gm else None;safety=float(np.min(sm)) if sm else None
    rho=min(goal,safety) if goal is not None and safety is not None else None
    success=bool(env.summary()['collision_free_success'] and term=='success' and error is None)
    arrays=dict(positions=np.asarray(p),velocities=np.asarray(v),actions=np.asarray(u),noise_keys=np.asarray(keys),
                goal_margins=np.asarray(gm),safety_margins=np.asarray(sm))
    meta=dict(uid=state['uid'],mode=mode,eta=eta,seed=0,termination=term,error=error,steps=len(u),rho=rho,
              success=success,goal_ever_margin=goal,safety_always_margin=safety,
              goal_active_step=int(np.argmax(gm))+1 if gm else None,
              safety_active_step=int(np.argmin(sm))+1 if sm else None,
              closure_max=float(max((x['closure_norm'] for x in audit),default=0)),
              closure_mean=float(np.mean([x['closure_norm'] for x in audit])) if audit else 0.,
              closure_nonzero_steps=sum(x['closure_norm']>1e-8 for x in audit),
              raw_violation_max=float(max((x['raw_max_violation'] for x in audit),default=0)),
              baseline_gap_mean=float(np.mean([x['baseline_gap'] for x in audit])) if audit else 0.,
              donor_correction_mean=float(np.mean([x['donor_correction_norm'] for x in audit])) if audit else 0.)
    return arrays,meta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',default='hybrid_protocol.json')
    ap.add_argument('--preflight',default='hybrid_cache_preflight.json')
    ap.add_argument('--outdir',default='hybrid_results')
    ap.add_argument('--shard',type=int,default=0)
    ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args()
    protocol=json.loads((HERE/args.protocol).read_text())
    pre=json.loads((HERE/args.preflight).read_text())
    if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(protocol['jobs']):raise RuntimeError('preflight mismatch')
    valid={(r['state_uid'],r['eta_uid'],r['controller_uid']) for r in pre['details'] if r['missing_seeds']}
    states={s['uid']:s for s in json.loads((Path('/home/zhihan/research/Basin_C1_hard_field_20261006')/'confirmation_manifest.json').read_text())['states']}
    rt=make_runtime('ring_exchange');out=HERE/args.outdir;out.mkdir(exist_ok=True)
    for i,j in enumerate(protocol['jobs']):
        if i%args.shards!=args.shard:continue
        if (j['uid'],j['eta_uid'],j['controller_uid']) not in valid:raise RuntimeError('unplanned')
        stem=f"{j['uid']}_{j['mode']}_{j['index']:02d}"
        meta=out/(stem+'.json');arr=out/(stem+'.npz')
        if meta.exists() and arr.exists():continue
        arrays,r=simulate_hybrid(rt,states[j['uid']],j['mode'],j['eta'])
        r.update(fraction=j['fraction'],eta_uid=j['eta_uid'])
        np.savez_compressed(arr,**arrays);meta.write_text(json.dumps(r,indent=2)+'\n')
        print(i,j['uid'],j['mode'],j['fraction'],r['rho'],r['termination'],r['closure_max'],flush=True)


if __name__=='__main__':main()
