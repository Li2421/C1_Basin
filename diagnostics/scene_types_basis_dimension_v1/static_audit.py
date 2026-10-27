"""Mechanism diagnostics on saved failed B3 states; no environment step."""
from core import *
from runtime import fields_for,ROUTE_EVALUATE
from shared_control.basis_dictionary import NEW,ALL,RoleState,SIGNED
from types import SimpleNamespace
import itertools
from scipy.optimize import nnls

def rank(a):
    s=np.linalg.svd(a,compute_uv=False)
    return int(np.sum(s>max(1e-12,1e-10*s[0])))if len(s)else 0

def zero_normal_fit(filters,target,tolerance=1e-7):
    """Floating NNLS residual, explicitly not an exact normal-cone certificate."""
    rows=[]
    for i,f in enumerate(filters):
        active=(np.abs(f.lower)<=tolerance)&(np.linalg.norm(f.A,axis=1)>1e-14)
        normals=-f.A[active].T
        try:
            if normals.shape[1]:
                coefficients,residual=nnls(normals,target[i],maxiter=max(100,10*normals.shape[1]))
            else:coefficients=np.zeros(0);residual=float(np.linalg.norm(target[i]))
            rows.append(dict(agent=i,near_active_rows=int(active.sum()),nnls_residual=float(residual),relative_residual=float(residual/max(np.linalg.norm(target[i]),1e-30)),minimum_multiplier=float(coefficients.min())if len(coefficients)else None))
        except Exception as exc:rows.append(dict(agent=i,nnls_unknown=repr(exc)))
    return dict(agents=rows,active_absolute_tolerance=tolerance,zero_linear_feasibility_residual=min(float(np.min(-f.lower))for f in filters),certificate=False,meaning='Numerical fit target_i = -A_near_active.T lambda, lambda>=0 at zero velocity. Speed disks are inactive at zero. Near-active tolerance and floating fits are residual diagnostics, not exact membership certificates. If zero is infeasible, this fit is not a projection-at-zero test.')

def main():
    frozen=read(HERE/'SCENES_FROZEN.json');scenes={e['scene']['name']:e['scene']for e in frozen['entries']if 'scene'in e}
    chosen={};b3_success={}
    if(HERE/'S1_planned.json').exists():
        for c in read(HERE/'S1_planned.json')['exact_configs']:
            if c['spec'].get('coefficients'):continue
            path=HERE/'raw'/f"{c['rollout_uid']}.json"
            if path.exists():
                row=read(path);k=(row['type_id'],row['controller'],row['state_index']);b3_success[k]=b3_success.get(k,0)+row['success']
    for p in sorted((HERE/'raw').glob('*.json')):
        r=read(p)
        if r['type_id']==2 and(HERE/'SCENE2_GEOMETRY_BLOCKER.json').exists()and not(HERE/'SCENE2_RESOLVED').exists():continue
        if r['solo']is not None or r['success']or r['numerical_failure']or r['purpose']not in('search','acceleration_certificate'):continue
        if r['spec'].get('coefficients')or r['spec']['eta'][0]<.5:continue
        k=(r['type_id'],r['controller']);old=chosen.get(k)
        priority=lambda q:(b3_success.get((q['type_id'],q['controller'],q['state_index']),0)>0,-q['episode_length'],q['rollout_uid'])
        if old is None or priority(r)<priority(old):chosen[k]=r
    results=[]
    for key,r in sorted(chosen.items()):
        s=scenes[r['scene']]
        with np.load(r['trace'])as a:
            if not len(a['safe']):continue
            remaining=a['route_remaining'];window=40;index=len(a['safe'])-1
            for t in range(window,len(remaining)):
                if float(remaining[t-window].sum()-remaining[t].sum())<.01:index=t;break
            state=dict(s['states'][r['state_index']],positions=a['positions'][index].tolist(),velocities=a['velocities'][index].tolist())
            safe=a['safe'][index].copy();nominal=a['nominal'][index].copy();guides=a['route_guides'][index].copy();rem=remaining[index].copy()
        env=RN.SwapEnv(state,RN.LiteratureConfig(**s['config']));env._route_enabled=s['route_guidance']
        filters=[RN.SwapSafety(env.snapshot(),i)for i in range(env.num_agents)];ff=fields_for(env,safe,guides,rem)
        columns,meta=ROUTE_EVALUATE(env.positions,guides,safe,nominal,ff,filters,env.walls,env.config,RoleState(),index,route_remaining=rem)
        B=np.stack([columns[k].reshape(-1)for k in ALL],axis=1);norms=np.linalg.norm(B,axis=0);angles={}
        for j,k in enumerate(ALL):
            for z in range(j):
                if norms[j]>1e-12 and norms[z]>1e-12:angles[k+'/'+ALL[z]]=float(np.degrees(np.arccos(np.clip(B[:,j]@B[:,z]/(norms[j]*norms[z]),-1,1))))
        retained={}
        for k in NEW:
            trials=[]
            for sign in(-1.,1.)if k in SIGNED else(1.,):
                target=safe+sign*columns[k]
                try:
                    projected=np.concatenate([f.project(target[i:i+1])[0]for i,f in enumerate(filters)])
                    delta=np.linalg.norm(projected-safe);trials.append(dict(sign=sign,projected_delta_norm=float(delta),retained_ratio=float(delta/max(norms[ALL.index(k)],1e-30)),basis_norm=float(norms[ALL.index(k)])))
                except Exception as exc:trials.append(dict(sign=sign,numerical_unknown=repr(exc)))
            retained[k]=trials
        vertices=[]
        for eta in itertools.product((.5,1.25),(-.5,.5),(0.,.75),(-1.,1.),(0.,1.)):
            target=safe.copy()
            for k,value in zip(('G','P','R','T1','Y3'),eta):target+=value*columns[k]
            try:
                projected=np.concatenate([f.project(target[i:i+1])[0]for i,f in enumerate(filters)])
                vertices.append(dict(eta=eta,projected_norm=float(np.linalg.norm(projected)),change_from_safe=float(np.linalg.norm(projected-safe)),minimum_original_linear_residual=min(float(np.min(f.A@projected[i]-f.lower))for i,f in enumerate(filters)),zero_normal_cone_residual_estimate=zero_normal_fit(filters,target)))
            except Exception as exc:vertices.append(dict(eta=eta,numerical_unknown=repr(exc)))
        box_check=dict(columns=['G','P','R','T1','Y3'],vertices=vertices,n=32,zero_feasible_up_to_float_tolerance=all(float(np.max(f.lower))<=1e-9 for f in filters),max_projected_norm=max((v['projected_norm']for v in vertices if 'projected_norm'in v),default=None),certificate=False,interpretation='32 production-projection probes at a fixed state, fresh fixed role and the stated five-column box. Numerical finite vertex evidence only; no exact normal-cone membership or continuous-rollout impossibility certificate is asserted.')
        feasible_probes=[]
        for i in range(env.num_agents):
            for axis in range(2):
                for sign in(-1.,1.):
                    target=safe.copy();target[i,axis]+=sign*env.config.max_speed
                    try:
                        u=np.concatenate([f.project(target[j:j+1])[0]for j,f in enumerate(filters)])
                        residual=min(float(np.min(f.A@u[j]-f.lower))for j,f in enumerate(filters))
                        excess=max(0.,float(np.linalg.norm(u,axis=1).max()-env.config.max_speed))
                        feasible_probes.append(dict(agent=i,axis=axis,sign=sign,original_linear_residual=residual,speed_excess=excess,original_feasible=residual>=-1e-7 and excess<=1e-7,action_delta=float(np.linalg.norm(u-safe)),guide_alignment=float(np.sum(ff[:,0]*u))))
                    except Exception as exc:feasible_probes.append(dict(agent=i,axis=axis,sign=sign,numerical_unknown=repr(exc)))
        unfinished=rem>env.config.goal_tolerance
        selected_rows=np.repeat(unfinished,2)
        unfinished_B=B[selected_rows]
        results.append(dict(type_id=r['type_id'],controller=r['controller'],source_rollout_uid=r['rollout_uid'],B3_Oracle16_success_count_for_case=b3_success.get((r['type_id'],r['controller'],r['state_index'])),step=index,position=env.positions,nominal=nominal,safe=safe,B3_rank=rank(B[:,:3]),full_rank=rank(B),unfinished_agents=np.flatnonzero(unfinished),unfinished_only_B3_rank=rank(unfinished_B[:,:3]),unfinished_only_full_rank=rank(unfinished_B),unfinished_safe_speed_norm=float(np.linalg.norm(safe[unfinished])),column_norms=dict(zip(ALL,norms)),column_angles_degrees=angles,rank_with_added_column={k:rank(np.c_[B[:,:3],columns[k].reshape(-1)])for k in NEW},projection_retention=retained,five_column_vertex_check=box_check,unrestricted_axis_action_probes=feasible_probes,role=meta['role'],svd_ranks=meta['svd_ranks'],note='Saved real failed B3 state; prefer cases where all16 B3 candidates failed. Rank of the whole system can include already-arrived robots, so unfinished-only ranks are separate. New roles freshly activated for this static counterfactual. Axis targets are generic diagnostic motions projected through the same production constraints; guide alignment is an instantaneous proxy, not a whole-task or deadlock-escape certificate.'))
    dump(HERE/'STATIC_MECHANISMS.json',dict(states=results,scope='No new rollout or plant step. Original distributed per-agent constraints and original production projection. Rank/angle and finite projection probes do not prove a family can or cannot solve the full task.',persistent_role_rule='The frozen dictionary selects a pair only when role.pair is None. It does not reassign after the pair becomes inactive; T3/Y/S1 can therefore lose activity on later conflicts. T2/C2 cover multiple agents.'))

if __name__=='__main__':main()
