"""Read-only frozen Toy action probes and arithmetic on saved trajectories. No env.step."""
import os
for k,v in {'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
            'MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
            'XLA_FLAGS':'--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'}.items():
    os.environ[k]=v
import sys,json,csv,hashlib,time,sqlite3,struct,inspect,platform
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
HARD=Path('/home/zhihan/research/Basin_C1_hard_field_20261006')
for p in (ROOT,FIELD,FIELD/'source',FIELD/'source/scripts'):sys.path.insert(0,str(p))
import numpy as np
import scipy,jax,jax.numpy as jnp
from scipy.optimize import nnls
from validation_stage2.run_rollouts import make_runtime
from validation_stage2.factorial_controller import action,descriptor

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x))
def dump(name,x): (HERE/name).write_text(json.dumps(x,indent=2,default=serial,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compact(x):return json.dumps(x,default=serial,separators=(',',':'),allow_nan=False)
def csvwrite(name,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:w.writerow({k:compact(v) if isinstance(v,(list,dict,np.ndarray,tuple)) else v for k,v in r.items()})
def key_for(uid,step,root):
    k=jax.random.fold_in(jax.random.PRNGKey(root),int(hashlib.sha256(uid.encode()).hexdigest()[:8],16))
    return jax.random.fold_in(jax.random.fold_in(k,0),step)

COUNTS={'native_samples':0,'projection_requests':0,'ff_integrations':0,'ff_field_evaluations':0,
        'jacobian_evaluations':0,'fd_field_evaluations':0,'canonical_action_checks':0}
def diag(safety,target,u,solver):
    slack=safety.A@u.ravel()-safety.lower
    sp=safety.max_speed-np.linalg.norm(u,axis=1)
    ai=np.flatnonzero(slack<=1e-7);si=np.flatnonzero(sp<=1e-7)
    # Multipliers independently reconstructed for constraints A u>=b and v-||u_i||>=0.
    normals=[-safety.A[i] for i in ai]
    for i in si:
        n=np.zeros(4);n[2*i:2*i+2]=u[i]/np.linalg.norm(u[i]);normals.append(n)
    c=np.asarray(normals).reshape(-1,4)
    lam=nnls(c.T,(target-u).ravel(),maxiter=10000)[0] if len(c) else np.zeros(0)
    stat=np.linalg.norm((u-target).ravel()+c.T@lam,ord=np.inf)
    residual=np.r_[slack[ai],sp[si]]
    comp=float(np.max(abs(lam*residual))) if len(lam) else 0.
    return dict(active_cbf_rows=ai,active_speed_balls=si,dual_multipliers_reconstructed=lam,
        dual_convention='-A linear, u_i/||u_i|| speed; diagnostic NNLS, not solver-returned duals',
        all_cbf_slacks=slack,pair_cbf_slack=float(slack[0]),min_wall_cbf_slack=float(slack[1:].min()),
        speed_ball_slacks=sp,min_speed_ball_slack=float(sp.min()),kkt_stationarity_inf=float(stat),
        kkt_complementarity=comp,kkt_primal_violation=float(max(0,-slack.min(),-sp.min())),
        weak_active=bool(np.any(lam<1e-7)),solver_diagnostics=solver,numerical_failure=False)

def main():
    start=time.monotonic();cpu=time.process_time()
    audit=json.loads((HERE/'pre_evaluation_audit.json').read_text())
    cfg=json.loads((HARD/'screening_manifest.json').read_text());root=cfg['future_root']
    states=json.loads((HARD/'confirmation_manifest.json').read_text())['states']
    states={s['uid']:s for s in states if s['scenario']=='toy_give_way'}
    # Cache audit is read-only and never promotes a partial controller profile to exact reuse.
    db=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
    cache=[]
    for eta in ([0.,0.,0.],[.5,0.,.5]):
        hx=':'.join(struct.pack('>d',x).hex() for x in eta)
        row=db.execute('SELECT eta_uid FROM eta WHERE canonical_hex=?',(hx,)).fetchone()
        rec=[]
        if row:
            rec=db.execute('SELECT r.controller_uid,c.flow_checkpoint_sha256,c.compatibility_quality,count(*) FROM rollout r JOIN controller_config c USING(controller_uid) WHERE r.eta_uid=? GROUP BY r.controller_uid',(row[0],)).fetchall()
        cache.append(dict(eta=eta,eta_uid=row[0] if row else None,controller_coverage=rec))
    db.close();dump('cache_coverage_audit.json',cache)
    rt=make_runtime('toy_give_way');env=rt.make_env(next(iter(states.values())))
    D=2*rt.config.agent_radius+rt.config.agent_collision_margin+rt.cbf.separation_buffer
    Rw=rt.config.agent_radius+rt.config.wall_radius+rt.config.wall_collision_margin+rt.cbf.separation_buffer
    geometry=dict(config=rt.config.to_dict(),cbf=rt.cbf.to_dict(),D=D,Rw=Rw,
        walls=[dict(name=n,endpoints=w) for n,w in zip(env.wall_names,env.walls)],goals=env.goals)
    # Historical center trajectories: success and failure, plus truncated zero baselines.
    old=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'
    paths=list((old/'results_full').glob('*toy*_*_candidate4_center.json'))
    paths+=list((old/'results_h80').glob('*toy*_*_zero_center.json'))
    trajectory=[];selected=[];source_paths=[]
    for uid in ('hard_confirm_toy_1_820001','hard_confirm_toy_1_820013'):
        st=states[uid];selected.append(dict(id=uid+':t0',base_uid=uid,step=0,
            positions=st['physical']['positions'],velocities=st['physical']['velocities'],origin='frozen confirmation true-t0'))
    for path in sorted(paths):
        meta=json.loads(path.read_text());npz=path.with_suffix('.npz')
        with np.load(npz) as z:
            p=z['positions'];a=z['actions'];keys=z['noise_keys']
            v=z['velocities'] if 'velocities' in z else np.concatenate([np.zeros((1,2,2)),a])
        uid=meta['state_uid'];chiY=p[:,1,1]-D;chiP=p[:,0,0]-p[:,1,0]-D
        yield_at=np.flatnonzero(chiY>0);pass_at=np.flatnonzero(chiP>0)
        ystep=int(yield_at[0]) if len(yield_at) else None
        pstep=int(pass_at[0]) if len(pass_at) else None
        # Hold means y2>D throughout horizontal overlap |x1-x2|<=D until passing.
        cross=np.flatnonzero(abs(p[:,0,0]-p[:,1,0])<=D)
        cross=cross[cross<=pstep] if pstep is not None else cross
        hold=float(chiY[cross].min()) if len(cross) else None
        trajectory.append(dict(path=str(path),npz=str(npz),meta_sha256=sha(path),npz_sha256=sha(npz),
            state_uid=uid,placement=meta['placement'],eta=meta['eta'],seed=meta['seed'],termination=meta['termination'],
            steps=len(a),success=meta['success'],error=meta['error'],yield_step=ystep,pass_step=pstep,
            min_chiY_during_overlap=hold,max_chiY=float(chiY.max()),max_chiP=float(chiP.max()),
            complete_yhp=bool(ystep is not None and pstep is not None and ystep<pstep and hold is not None and hold>0),
            reverse_role_max_y=float(p[:,0,1].max()),saved_min_safety_margin=meta['safety_always_margin'],
            all_noise_keys_match=bool(all(np.array_equal(k,key_for(uid,i,root)) for i,k in enumerate(keys))),
            integration_residual=float(np.max(abs(np.diff(p,axis=0)-rt.config.dt*a))) if len(a) else 0.))
        source_paths.extend([path,npz])
        if 'results_full' not in str(path):continue
        env=rt.make_env(states[uid]);mins=[];active_pair=[];active_wall=[]
        for t in range(len(a)):
            env.positions=p[t].copy();env.velocities=v[t].copy();s=rt.safety(env)
            slack=s.A@a[t].ravel()-s.lower
            mins.append(float(s.geometry['min_wall_h']))
            active_pair.append(slack[0]<=1e-7);active_wall.append(slack[1:].min()<=1e-7)
        # Select by saved geometry/constraint diagnostics, not new witness outcomes.
        criteria={'pair_closest':int(np.argmin(np.linalg.norm(p[:-1,0]-p[:-1,1],axis=1))),
                  'wall_closest':int(np.argmin(mins)), 'bay_highest_y2':int(np.argmax(p[:-1,1,1]))}
        both=np.flatnonzero(np.asarray(active_pair)&np.asarray(active_wall))
        if len(both):criteria['recorded_pair_wall_active']=int(both[0])
        for reason,t in criteria.items():
            selected.append(dict(id=f'{path.stem}:t{t}',base_uid=uid,step=t,positions=p[t],velocities=v[t],
                origin=str(npz),selection=reason,recorded_chain=meta['placement'],recorded_eta=meta['eta'],
                recorded_action=a[t],recorded_noise_key=keys[t]))
    # Geometry-only states are explicit constructed probes, never passed off as recorded episodes.
    selected.extend([
        dict(id='constructed_open_yield_region',base_uid='geometry_open',step=0,positions=[[-.30,0],[.02,.13]],velocities=np.zeros((2,2)),origin='constructed geometric witness'),
        dict(id='constructed_pair_floor_contact',base_uid='geometry_contact',step=0,
             positions=[[-D/2,-.2+Rw],[D/2,-.2+Rw]],velocities=np.zeros((2,2)),origin='constructed collision-free CBF boundary')])
    selected=list({s['id']:s for s in selected}.values())
    # Freeze state selection/config before evaluating witness actions.
    dump('selected_states.json',selected);dump('geometry_data.json',geometry);dump('trajectory_metrics.json',trajectory)
    loaded={}
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if f and str(FIELD) in str(f) and str(f).endswith('.py'):loaded[str(Path(f).resolve())]=sha(f)
    audit.update(loaded_module_hashes=loaded,checkpoint=str(rt.checkpoint),checkpoint_sha256=sha(rt.checkpoint),
        runtime_config=rt.config.to_dict(),runtime_agent_config=dict(rt.agent.config),cbf=rt.cbf.to_dict(),
        controller_descriptors={ch:descriptor(rt,ch) for ch in ('TT','FF')},
        packages=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,jax=jax.__version__),
        devices=[str(d) for d in jax.devices()],selected_state_hash=sha(HERE/'selected_states.json'),
        new_rollouts=0,new_training=0,source_paths_reused=[str(p) for p in source_paths])
    dump('evaluation_frozen_manifest.json',audit)
    actions=[];substeps=[];replay=[];failures=[]
    for state in selected:
        uid=state['id'];env=rt.make_env(states['hard_confirm_toy_1_820001'])
        env.positions=np.asarray(state['positions'],float).copy();env.velocities=np.asarray(state['velocities'],float).copy()
        key=key_for(state['base_uid'],state['step'],root)
        try:
            obs,safety,flow,safe,delta,_=rt.prepare(env,key,np.array([.5,0,.5]));COUNTS['native_samples']+=1
            COUNTS['projection_requests']+=safety.calls
            B=rt.basis.compute(env.positions,env.goals,safe,rt.config.max_speed)
            wall=env.walls;a=wall[:,0];seg=wall[:,1]-a
            unclipped=np.sum((env.positions[:,None]-a)*seg,axis=-1)/np.sum(seg*seg,axis=-1)
            branches=np.where(unclipped<=0,'start',np.where(unclipped>=1,'end','interior'))
            meta=dict(state_id=uid,base_uid=state['base_uid'],physical_step=state['step'],seed=0,noise_key=np.asarray(key),
                positions=env.positions,velocities=env.velocities,basis_columns=np.stack(B.values,axis=-1),
                raw_delta=delta,raw_delta2y=float(delta[1,1]),raw_delta1x_minus_delta2x=float(delta[0,0]-delta[1,0]),
                raw_both_positive=bool(delta[1,1]>0 and delta[0,0]-delta[1,0]>0),
                min_wall_h=safety.geometry['min_wall_h'],pair_h=safety.geometry['pairwise_h'],
                nearest_point_branches=branches.tolist(),wall_segment_parameters_unclipped=unclipped,
                native_flow=flow,safe_anchor=safe,chiY=float(env.positions[1,1]-D),chiP=float(env.positions[0,0]-env.positions[1,0]-D),
                inside_workspace=bool(not env.outside(env.positions).any()))
            def wf(h,s):
                z=(h-jnp.asarray(rt.mean).ravel())/jnp.asarray(rt.scale).ravel()
                return jnp.asarray(rt.scale).ravel()*rt.velocity_fn(obs,z,s)
            jac=jax.jit(jax.jacfwd(wf,argnums=0))
            for label,d in [('zero',np.zeros_like(delta)),('witness',delta)]:
                target=safe+d;u,sol=safety.project(target);COUNTS['projection_requests']+=1
                actions.append(dict(**meta,parameter=label,eta=[0,0,0] if label=='zero' else [.5,0,.5],chain='TT',
                    applied_delta=d,target=target,executed=u,u2y=float(u[1,1]),u1x_minus_u2x=float(u[0,0]-u[1,0]),
                    **diag(safety,target,u,sol)))
                z0=np.asarray(jax.random.normal(key,(1,4))).reshape(2,2)
                h=rt.mean+rt.scale*z0;COUNTS['ff_integrations']+=1
                for k in range(rt.steps):
                    s=k/rt.steps;rem=1-s;rho=min(1.,.1/rem)
                    f=np.asarray(wf(jnp.asarray(h.ravel()),jnp.asarray(s))).reshape(2,2);COUNTS['ff_field_evaluations']+=1
                    r=h+rem*(f+d);q,sol=safety.project_field_target(r);COUNTS['projection_requests']+=1
                    hn=(1-rho)*h+rho*q
                    J=np.asarray(jac(jnp.asarray(h.ravel()),jnp.asarray(s)));COUNTS['jacobian_evaluations']+=1
                    E=np.eye(4)+rem*J
                    extra={}
                    if k==9:
                        fds=[]
                        for eps in (1e-4,3e-5):
                            fd=np.column_stack([(np.asarray(wf(jnp.asarray(h.ravel()+np.eye(4)[j]*eps),jnp.asarray(s)))-np.asarray(wf(jnp.asarray(h.ravel()-np.eye(4)[j]*eps),jnp.asarray(s))))/(2*eps) for j in range(4)])
                            COUNTS['fd_field_evaluations']+=8
                            fds.append(dict(epsilon=eps,jacobian_absolute_error=float(np.linalg.norm(fd-J))))
                        extra=dict(D9=h+.1*f,I_plus_point1_J=E,D9_singular_values=np.linalg.svd(E,compute_uv=False),D9_fd_checks=fds)
                    substeps.append(dict(state_id=uid,parameter=label,eta=[0,0,0] if label=='zero' else [.5,0,.5],
                        k=k,s=s,rho=rho,h=h,f=f,delta=d,r=r,q=q,h_next=hn,
                        f_world_jacobian=J,endpoint_jacobian_spectral_norm=float(np.linalg.norm(E,2)),
                        q2y=float(q[1,1]),q1x_minus_q2x=float(q[0,0]-q[1,0]),
                        **diag(safety,r,q,sol),**extra))
                    h=hn
                actions.append(dict(**meta,parameter=label,eta=[0,0,0] if label=='zero' else [.5,0,.5],chain='FF',
                    applied_delta=d,target=r,executed=h,u2y=float(h[1,1]),u1x_minus_u2x=float(h[0,0]-h[1,0]),
                    **diag(safety,r,h,sol)))
            # Independent comparison to canonical runtime, not merely the written recurrence.
            for ch in ('TT','FF'):
                u,cert,calls,_=action(rt,env,key,np.array([.5,0,.5]),ch)
                COUNTS['canonical_action_checks']+=1;COUNTS['native_samples']+=1
                COUNTS['projection_requests']+=calls
                if ch=='FF':COUNTS['ff_integrations']+=1;COUNTS['ff_field_evaluations']+=10
                own=next(r for r in actions if r['state_id']==uid and r['chain']==ch and r['parameter']=='witness')
                replay.append(dict(state_id=uid,chain=ch,canonical_error=float(np.max(abs(u-own['executed'])))))
            if 'recorded_action' in state:
                u,cert,calls,_=action(rt,env,key,np.array(state['recorded_eta']),state['recorded_chain'])
                COUNTS['canonical_action_checks']+=1;COUNTS['native_samples']+=1;COUNTS['projection_requests']+=calls
                if state['recorded_chain']=='FF':COUNTS['ff_integrations']+=1;COUNTS['ff_field_evaluations']+=10
                replay.append(dict(state_id=uid,chain=state['recorded_chain'],historical_action_error=float(np.max(abs(u-state['recorded_action'])))))
        except Exception as exc:
            failures.append(dict(state_id=uid,error_type=type(exc).__name__,error=str(exc)))
            print('FAIL',uid,exc,flush=True)
        print('PROBED',uid,flush=True)
    csvwrite('tt_ff_action_validation.csv',actions);csvwrite('ff_substep_validation.csv',substeps)
    dump('replay_checks.json',replay);dump('numerical_failures.json',failures)
    audit.update(state_ids=[s['id'] for s in selected],counts=COUNTS,action_rows=len(actions),ff_substep_rows=len(substeps),
        numerical_failures=failures,trajectory_records_reused=len(trajectory),new_physical_steps=0,
        elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        script_sha256=sha(__file__),new_fits=0,new_rollouts=0,gpu_seconds=0,
        parameter_ids={'zero':[0,0,0],'witness':[.5,0,.5]},seed=0,
        source_hashes_unchanged=all(sha(p)==h for p,h in audit['source_hashes'].items()))
    dump('validation_manifest.json',audit)
    print('DONE',compact({k:audit[k] for k in ('action_rows','ff_substep_rows','counts','elapsed_seconds','numerical_failures')}),flush=True)

if __name__=='__main__':main()
