"""Read real failed trajectories; solve only fixed-state diagnostics, never rollouts."""
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:18])
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from types import SimpleNamespace
import numpy as np
import clarabel
from scipy import sparse
from scipy.optimize import nnls
from runtime import OUT, RN, dump
from oracle import inputs, worker_init
from shared_control.correction_b5 import FrozenYield,YieldConfig,tangent,target,yield_menu,conflict_components
from shared_control.hard_projection import project_velocity,HardProjectionConfig


def matrix_stats(B):
    B=B.reshape(-1,B.shape[-1]);norm=np.linalg.norm(B,axis=0)
    cosine=(B.T@B)/np.maximum(norm[:,None]*norm[None,:],1e-30)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)));angle[(norm[:,None]*norm[None,:])<1e-20]=np.nan
    return dict(rank=int(np.linalg.matrix_rank(B,tol=1e-8)),norms=norm,angles_deg=angle,
                singular_values=np.linalg.svd(B,compute_uv=False))


def support(f,d,speed):
    # Independent support SOCP over exactly the original local rows and speed cone.
    settings=clarabel.DefaultSettings();settings.verbose=False
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-10
    A=sparse.vstack([-sparse.csc_matrix(f.A),sparse.csc_matrix(np.vstack([np.zeros(2),-np.eye(2)]))],format='csc')
    b=np.r_[-f.lower,speed,0.,0.]
    sol=clarabel.DefaultSolver(sparse.csc_matrix((2,2)),-d,A,b,
        [clarabel.NonnegativeConeT(len(f.lower)),clarabel.SecondOrderConeT(3)],settings).solve()
    if str(sol.status)not in ('Solved','AlmostSolved'):return dict(status=str(sol.status),action=None)
    a=np.asarray(sol.x);return dict(status=str(sol.status),action=a,progress=float(d@a),
        row_residual=float(np.maximum(f.lower-f.A@a,0).max(initial=0)),speed_excess=max(0.,float(np.linalg.norm(a)-speed)))


def analyze(row):
    tr=np.load(row['trace']);scene=inputs()[0][row['scene']];cfg=RN.LiteratureConfig(**scene['config'])
    # Most stationary unfinished state in the final 2/3, without calling it a theorem-certified deadlock.
    speed=np.linalg.norm(tr['actions'],axis=-1).max(axis=1)
    ds=np.linalg.norm(tr['goals'][None]-tr['positions'][:-1],axis=-1)
    candidates=np.flatnonzero((np.arange(len(speed))>=len(speed)//3)&(ds.max(axis=1)>cfg.goal_tolerance))
    t=int(candidates[np.argmin(speed[candidates])]);p=tr['positions'][t];g=tr['goals'];v=tr['safe'][t]
    geo=json.loads(str(tr['first_geometry_json']))[t]
    fs=[SimpleNamespace(A=np.asarray(f['A']),lower=np.asarray(f['lower']),kinds=f['labels'])for f in geo]
    fields=np.stack([RN.distributed_fields(p,g,v,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg))for i in range(len(p))])
    B3=fields.transpose(0,2,1);baseline=tr['actions'][t]
    mask=ds[t]>cfg.goal_tolerance;goal_dir=(g-p)/np.maximum(ds[t,:,None],1e-30);goal_dir[~mask]=0
    witnesses=[support(f,d,cfg.max_speed)for f,d in zip(fs,goal_dir)]
    b3stats=matrix_stats(B3)
    original_zero_feasible=all(np.max(f.lower,initial=0)<=1e-9 for f in fs)
    normals=[-f.A[np.abs(f.lower)<=1e-6].T for f in fs]
    distance=np.linalg.norm(p[:,None]-p[None,:],axis=-1);np.fill_diagonal(distance,np.inf)
    i,j=np.unravel_index(np.argmin(distance),distance.shape)
    roles=[r for a,b in [(int(i),int(j)),(int(j),int(i))]for r in yield_menu(a,b)]
    configs=[];vertices=[]
    lo=[.5,-.5,0.,-1.,0.];hi=[1.25,.5,.75,1.,1.]
    for recipe in ('goal','cluster','wall'):
        T=tangent(recipe,p,fields[:,0],cfg.max_speed,RN.contact_radius(cfg),fs)
        for ri,rc in enumerate(roles):
            role=FrozenYield(rc,p,g,RN.contact_radius(cfg)/1.25);Y=np.zeros_like(p);Y[rc.yielder]=role.clear(p,cfg.max_speed)[rc.yielder]-v[rc.yielder]
            B5=np.concatenate([B3,T[:,:,None],Y[:,:,None]],axis=2)
            stats=matrix_stats(B5);projection_residual=[];cone_residual=[];progress=[];errors=0
            for k,e in enumerate(itertools.product(*zip(lo,hi))):
                a=target(v,fields,T,role,p,cfg.max_speed,e,'add')
                rr=[]
                for normal,ref in zip(normals,a):
                    rr.append(nnls(normal,ref)[1] if normal.shape[1] else float(np.linalg.norm(ref)))
                residual=float(np.linalg.norm(rr));cone_residual.append(residual)
                try:
                    projected=np.concatenate([project_velocity(a[ii:ii+1],f.A,f.lower,cfg.max_speed,HardProjectionConfig())[0]for ii,f in enumerate(fs)])
                    mag=float(np.linalg.norm(projected));pr=float(np.sum(goal_dir*projected))
                    projection_residual.append(mag);progress.append(pr)
                    vertices.append(dict(recipe=recipe,role_index=ri,vertex=k,eta=e,normal_cone_residual=residual,
                         projected_norm=mag,goal_progress=pr,projection_retained=float(np.linalg.norm(projected-baseline))))
                except Exception as exc:
                    errors+=1;vertices.append(dict(recipe=recipe,role_index=ri,vertex=k,eta=e,error=str(exc)))
            configs.append(dict(recipe=recipe,role_index=ri,role=rc.__dict__,waypoint=role.waypoint,stats=stats,
                rank_gain=stats['rank']-b3stats['rank'],normal_cone_residual_max=max(cone_residual),
                projected_norm_max=max(projection_residual,default=None),positive_goal_progress_max=max(progress,default=None),
                solver_unknown=errors))
    # B3 box vertices plus the actually searched projected actions at this fixed state.
    b3vertices=[]
    for e in itertools.product(*zip(lo[:3],hi[:3])):
        a=v.copy()
        for ii in range(len(p)):a[ii]+=sum(float(x)*f for x,f in zip(e,fields[ii]))
        try:
            u=np.concatenate([project_velocity(a[ii:ii+1],f.A,f.lower,cfg.max_speed,HardProjectionConfig())[0]for ii,f in enumerate(fs)])
            b3vertices.append(dict(eta=e,norm=float(np.linalg.norm(u)),progress=float(np.sum(goal_dir*u))))
        except Exception as exc:b3vertices.append(dict(eta=e,error=str(exc)))
    active_pairs=[lab for f in fs for lab,b in zip(f.kinds,f.lower) if lab[0]=='pair' and abs(b)<1e-4]
    active_walls=[lab for f in fs for lab,b in zip(f.kinds,f.lower) if lab[0]=='wall' and abs(b)<1e-4]
    out=dict(scene=row['scene'],controller=row['controller'],source_trace=row['trace'],source_candidate=row['spec'],
        step=t,time=t*cfg.dt,positions=p,goals=g,safe=v,actual_action=baseline,
        selected_max_speed=float(speed[t]),unfinished_agents=int(mask.sum()),arrived_agents=int((~mask).sum()),
        goal_distances=ds[t],b3=b3stats,b3_vertices=b3vertices,configs=configs,
        active_pairs=active_pairs,active_walls=active_walls,contact_components=conflict_components(p,RN.contact_radius(cfg)),
        original_zero_feasible=original_zero_feasible,positive_progress_support=witnesses,
        joint_domain='Cartesian product of original distributed sets; no centralized replacement',
        certificate='Numerical NNLS residuals with 1e-6 active tolerance; 32 ADD vertices per fixed configuration. NOT an exact mathematical certificate. BLEND is not affine and not covered.',
        distinction='Instantaneous positive goal progress or nonzero motion does not prove a successful collision-free rollout.')
    name=row['scene']+'__'+row['controller']+row.get('diagnostic_tag','');dump(OUT/'mechanisms'/f'{name}.json',out)
    dump(OUT/'mechanisms'/f'{name}_vertices.json',vertices)
    return dict(scene=row['scene'],controller=row['controller'],speed=float(speed[t]),rank=b3stats['rank'],
                max_rank=max(c['stats']['rank']for c in configs),vertex_count=len(vertices),arrived=int((~mask).sum()))


def main():
    (OUT/'mechanisms').mkdir(exist_ok=True)
    rows=list(map(json.loads,(OUT/'candidates.jsonl').open()));selected={}
    for r in rows:
        if r['spec']['mode']!='b3' or r['success'] or r['numerical_failure'] or not r['trace']:continue
        key=(r['scene'],r['controller'])
        if key not in selected or sum(r['final_goal_distances'])<sum(selected[key]['final_goal_distances']):selected[key]=r
    with ProcessPoolExecutor(max_workers=min(18,max(1,len(selected))),initializer=worker_init)as pool:
        summaries=list(pool.map(analyze,selected.values()))
    dump(OUT/'mechanisms_summary.json',summaries);print(summaries)


if __name__=='__main__':main()
