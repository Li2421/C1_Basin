"""Fixed-state joint-cone diagnostics on actual old-scene B3 failures."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import itertools,json
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from scipy.optimize import nnls
from legacy import OUT,runtime,init_worker,dump,jax
from shared_control.correction_b5 import FrozenYield,YieldConfig,yield_menu,tangent,target


def stats(B):
    M=B.reshape(-1,B.shape[-1]);n=np.linalg.norm(M,axis=0);a=np.degrees(np.arccos(np.clip((M.T@M)/np.maximum(n[:,None]*n[None,:],1e-30),-1,1)))
    a[(n[:,None]*n[None,:])<1e-20]=np.nan
    return dict(rank=int(np.linalg.matrix_rank(M,tol=1e-8)),norms=n,angles_deg=a,singular_values=np.linalg.svd(M,compute_uv=False))


def analyze(c):
    jax.config.update('jax_enable_x64',c['scenario']=='toy_give_way');rt=runtime(c['scenario'])
    row=c['row'];tr=np.load(row['trace']);env=rt.make_env(c['state']);p=tr['positions'];g=tr['goals'];speed=np.linalg.norm(tr['actions'],axis=-1).max(axis=1)
    ts=np.arange(len(speed)//3,len(speed));t=int(ts[np.argmin(speed[ts])])
    env.positions=p[t].copy();env.velocities=tr['actions'][t-1].copy()if t else np.zeros_like(p[t]);env.goals=g
    safety=rt.safety(env);v=tr['safe'][t];f=rt.basis.compute(p[t],g,v,rt.config.max_speed);B3=np.stack(f.values,axis=-1)
    fs=[]
    for i in range(len(g)):
        a=safety.geometry['wall_gradient'][i];lower=-rt.cbf.gamma_wall*safety.geometry['wall_h'][i]
        fs.append(SimpleNamespace(A=a,lower=lower,kinds=[['wall',i,k]for k in range(len(a))]))
    p=p[t];d=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(d,np.inf);i,j=map(int,np.unravel_index(np.argmin(d),d.shape))
    length=2*rt.config.agent_radius+rt.config.agent_collision_margin+rt.cbf.separation_buffer
    roles=[r for a,b in [(i,j),(j,i)]for r in yield_menu(a,b)]
    normal=-safety.A[np.abs(safety.lower)<=1e-6].T
    goal_dir=(g-p)/np.maximum(np.linalg.norm(g-p,axis=-1,keepdims=True),1e-30)
    goal_dir[np.linalg.norm(g-p,axis=-1)<=rt.config.goal_tolerance]=0
    witness,_=safety.project(1000*goal_dir)
    configs=[];vertices=[]
    for recipe in ('goal','cluster','wall'):
        T=tangent(recipe,p,f.values[0],rt.config.max_speed,1.25*length,fs)
        for ri,rc in enumerate(roles):
            role=FrozenYield(rc,p,g,length);Y=np.zeros_like(p);Y[rc.yielder]=role.clear(p,rt.config.max_speed)[rc.yielder]-v[rc.yielder]
            B5=np.concatenate([B3,T[:,:,None],Y[:,:,None]],axis=-1)
            M=B3.reshape(-1,3);proj=M@np.linalg.lstsq(M,T.ravel(),rcond=None)[0]
            conf=dict(recipe=recipe,role=rc.__dict__,waypoint=role.waypoint,stats=stats(B5),T_outside_B3=float(np.linalg.norm(T.ravel()-proj)),vertices=[])
            for k,e in enumerate(itertools.product(*zip([.5,-.5,0.,-1.,0.],[1.25,.5,.75,1.,1.]))):
                a=target(v,np.stack(f.values,axis=1),T,role,p,rt.config.max_speed,e,'add')
                residual=nnls(normal,a.ravel(),maxiter=10000)[1]if normal.shape[1]else float(np.linalg.norm(a))
                try:
                    u,_=safety.project(a);r=dict(vertex=k,eta=e,normal_cone_residual=float(residual),projected_norm=float(np.linalg.norm(u)),goal_progress=float(np.sum(goal_dir*u)),certificate=safety.certificate(u))
                except Exception as exc:r=dict(vertex=k,eta=e,error=str(exc))
                conf['vertices'].append(r)
            configs.append(conf)
    result=dict(scenario=c['scenario'],source_trace=row['trace'],source_method=row['method'],step=t,time=t*rt.config.dt,
        positions=p,goals=g,safe=v,actual_action=tr['actions'][t],selected_max_speed=float(speed[t]),b3=stats(B3),
        original_zero_feasible=bool(np.max(safety.lower,initial=0)<=1e-9),joint_rows=safety.A,joint_lower=safety.lower,
        feasible_progress_witness=dict(action=witness,progress=float(np.sum(goal_dir*witness)),certificate=safety.certificate(witness)),
        witness_scope='Projection of 1000*goal direction is a feasible numerical witness, NOT a solved linear-support optimum.',
        normal_cone_scope='32 vertices/fixed ADD config; numerical active tolerance1e-6; no exact certificate; BLEND excluded.',configs=configs)
    dump(OUT/'mechanisms'/('legacy_'+c['scenario']+'.json'),result)
    return dict(scene=c['scenario'],rank=result['b3']['rank'],speed=result['selected_max_speed'],vertices=32*len(configs))


def main():
    tasks=json.load(open(OUT/'LEGACY_FROZEN.json'))['tasks'];rows=list(map(json.loads,(OUT/'legacy_candidates.jsonl').open()));cases=[]
    for sc in ('toy_give_way','double_bottleneck','ring_exchange','four_way_intersection'):
        r=next(r for r in rows if r['scenario']==sc and r['method']=='B3'and not r['success'])
        c=next(c for c in tasks if c['rollout_uid']==r['rollout_uid']);cases.append(dict(c,scenario=sc,row=r))
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=init_worker)as pool:summary=list(pool.map(analyze,cases))
    dump(OUT/'legacy_mechanisms_summary.json',summary);print(summary)

if __name__=='__main__':main()
