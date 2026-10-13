"""Fixed-key action regularity and physical-state continuity probes.

Only the frozen controller is called. No physical rollout occurs here.
"""
import argparse
import copy
import hashlib
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
from closed_loop_probe import key_for
from validation_stage2.run_rollouts import make_runtime

def digest(v):
    return hashlib.sha256(np.asarray(v,dtype=np.uint8).tobytes()).hexdigest()[:16]

def activity(safety,a,tol):
    z=np.asarray(a,float)
    linear=safety.A@z.ravel()-safety.lower
    speed=safety.max_speed-np.linalg.norm(z,axis=1)
    flags=np.r_[linear<tol,speed<tol]
    return dict(hash=digest(flags),active_linear=int(np.sum(linear<tol)),
                active_speed=int(np.sum(speed<tol)),minimum_linear=float(np.min(linear)),
                minimum_speed_slack=float(np.min(speed)))

def frozen(rt,state,key):
    env=rt.make_env(state)
    obs,safety,flow,safe,_,_=rt.prepare(env,key,np.zeros(3))
    b=np.stack(rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed).values,axis=-1)
    # Each column is a physical correction direction; shape [N,2,3].
    return env,obs,safety,flow,safe,b.reshape(-1,3)

def action(rt,ctx,key,eta,chain,tol):
    env,obs,safety,flow,safe,b=ctx
    delta=(b@np.asarray(eta,float)).reshape(safe.shape)
    if chain=='TT':
        a,diag=safety.project(safe+delta)
        record=[activity(safety,a,tol)]
        status=[str(diag.get('status'))]
    else:
        a,trace=rt.integrate(env,obs,key,delta,safety,traced=True)
        record=[activity(safety,z,tol) for z in trace['targets']]
        status=[str(q.get('solver_status')) for q in trace['steps']]
    return np.asarray(a).ravel(),dict(branch_hash=':'.join(x['hash'] for x in record),
        active_linear=[x['active_linear'] for x in record],
        active_speed=[x['active_speed'] for x in record],
        minimum_linear=min(x['minimum_linear'] for x in record),
        minimum_speed_slack=min(x['minimum_speed_slack'] for x in record),
        solver_status=status)

def jacobian(rt,ctx,key,eta,chain,eps,tol):
    g=[];details=[]
    for axis in range(3):
        v=np.zeros(3);v[axis]=eps
        a,da=action(rt,ctx,key,np.asarray(eta)+v,chain,tol)
        b,db=action(rt,ctx,key,np.asarray(eta)-v,chain,tol)
        g.append((a-b)/(2*eps));details.append((a,b,da,db))
    return np.stack(g,axis=1),details

def one_action_case(rt,st,point,chain,protocol):
    key=key_for(st['uid'],protocol['seed'],0,protocol['future_root'])
    ctx=frozen(rt,st,key);eta=np.asarray(point['eta'],float);tol=protocol['action_active_tolerance']
    center,dc=action(rt,ctx,key,eta,chain,tol)
    eps=np.asarray(protocol['action_eps'],float)
    gs=[];branches=[];one_sided=[]
    for e in eps:
        g,pairs=jacobian(rt,ctx,key,eta,chain,e,tol)
        gs.append(g);branches.append([dc['branch_hash']==da['branch_hash']==db['branch_hash']
                                      for _,_,da,db in pairs])
        one_sided.append([float(np.linalg.norm((a-center)/e-(center-b)/e)) for a,b,_,_ in pairs])
    rng=np.random.default_rng(int(hashlib.sha256((st['uid']+point['name']).encode()).hexdigest()[:8],16))
    direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
    hold=[]
    for e in (float(eps[0]),float(eps[-1])):
        a,d=action(rt,ctx,key,eta+e*direction,chain,tol)
        pred=gs[-1]@(e*direction);actual=a-center
        hold.append(dict(epsilon=e,absolute_error=float(np.linalg.norm(actual-pred)),
            relative_error=float(np.linalg.norm(actual-pred)/max(np.linalg.norm(actual),1e-12)),
            cosine=float(np.dot(actual,pred)/max(np.linalg.norm(actual)*np.linalg.norm(pred),1e-12)),
            branch_same=d['branch_hash']==dc['branch_hash']))
    sig=np.linalg.svd(ctx[-1],compute_uv=False)
    out=dict(uid=st['uid'],scene=st['scenario'],eta_name=point['name'],eta=eta.tolist(),chain=chain,
             center_activity=dc,eps=eps.tolist(),branch_stable_axes=branches,
             one_sided_gap=one_sided,holdout=hold,direction=direction.tolist(),
             relative_G_scale_change=[float(np.linalg.norm(g-gs[-1])/max(np.linalg.norm(gs[-1]),1e-12)) for g in gs],
             basis_singular_values=sig.tolist())
    arrays=dict(B=ctx[-1],G_epsilon=np.asarray(gs),action_center=center)
    return arrays,out

def one_state_case(rt,st,point,chain,h,sign,protocol):
    state=copy.deepcopy(st)
    d=np.asarray(protocol['state_position_directions'][st['uid']],float)
    pos=np.asarray(state['physical']['positions'],float)
    state['physical']['positions']=(pos+sign*h*d).tolist()
    key=key_for(st['uid'],protocol['seed'],0,protocol['future_root'])
    ctx=frozen(rt,state,key);eta=np.asarray(point['eta'],float);tol=protocol['action_active_tolerance']
    a,diag=action(rt,ctx,key,eta,chain,tol)
    g,_=jacobian(rt,ctx,key,eta,chain,protocol['state_jacobian_epsilon'],tol)
    arrays=dict(B=ctx[-1],G=g,action=a,positions=np.asarray(state['physical']['positions']))
    out=dict(uid=st['uid'],scene=st['scenario'],eta_name=point['name'],eta=eta.tolist(),
             chain=chain,state_radius=h,state_sign=sign,activity=diag,
             basis_singular_values=np.linalg.svd(ctx[-1],compute_uv=False).tolist())
    return arrays,out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('action','state'),required=True)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    args=ap.parse_args();protocol=json.loads((HERE/'protocol.json').read_text())
    selected=protocol['states'] if args.mode=='action' else [s for s in protocol['states'] if s['uid'] in protocol['state_probe_uids']]
    jobs=[]
    for st in selected:
        points=protocol['action_points'] if args.mode=='action' else protocol['state_probe_points']
        for point in points:
            for chain in ('TT','FF'):
                if args.mode=='action':jobs.append((st,point,chain,None,0))
                else:
                    for h in [0.]+protocol['state_radii']:
                        for sign in ((0,) if h==0 else (-1,1)):
                            jobs.append((st,point,chain,h,sign))
    # Shards preserve a single scenario per process when shards=4.
    jobs=[j for j in jobs if (0 if j[0]['scenario']=='ring_exchange' else 2)+
          (int(j[0]['uid'][-1])%2)==args.shard] if args.shards==4 else [j for i,j in enumerate(jobs) if i%args.shards==args.shard]
    runtime={}
    for st,point,chain,h,sign in jobs:
        if st['scenario'] not in runtime:runtime[st['scenario']]=make_runtime(st['scenario'])
        rt=runtime[st['scenario']]
        stem=f"{st['uid']}_{point['name']}_{chain}" if h is None else f"{st['uid']}_{point['name']}_{chain}_h{h:g}_s{sign}"
        dest=HERE/('action_results' if h is None else 'state_results')
        meta=dest/(stem+'.json');npz=dest/(stem+'.npz')
        if meta.exists() and npz.exists():continue
        try:
            arrays,out=(one_action_case(rt,st,point,chain,protocol) if h is None else
                        one_state_case(rt,st,point,chain,h,sign,protocol))
            np.savez_compressed(npz,**arrays);meta.write_text(json.dumps(out,indent=2)+'\n')
            print(stem,'ok',flush=True)
        except Exception as exc:
            meta.write_text(json.dumps(dict(stem=stem,error=dict(type=type(exc).__name__,message=str(exc))),indent=2)+'\n')
            print(stem,'ERROR',repr(exc),flush=True)

if __name__=='__main__':main()
