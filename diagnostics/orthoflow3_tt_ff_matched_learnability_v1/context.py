"""Matched physical response channels measured with the REAL TT/FF operator.

No success label is read. H20 context is <=1 second of simulated physical
time, and goal fingerprints are single-step action queries, not task rollouts.
"""
from __future__ import annotations
import argparse,copy,hashlib,time
import numpy as np
from .design import ROOT,MAIN,read,write,sha,freeze
from .cache import protocol
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_data import goal_design,clearance
assert __import__('pathlib').Path(rep.__file__).resolve()==MAIN/'diagnostics/orthoflow3_unified_representation_v1/representation.py'
ROOTS=(2026100417,2026100499,2026100521,2026100543)
WIDTH=76


def physical(rt,state):
    import jax
    env=rt.make_env(state);cfg=rt.config;n=len(env.positions)
    _,flow,_=rt.nominal(env,jax.random.PRNGKey(ROOTS[2]));d=vars(cfg)
    if rt.scenario=='ring_exchange':
        center=env.obstacle_center.tolist()
        obstacles=[dict(a=center,b=center,radius=cfg.obstacle_radius,curvature=1/cfg.obstacle_radius,interior_sign=1.),
            dict(a=center,b=center,radius=cfg.outer_radius,curvature=-1/cfg.outer_radius,interior_sign=-1.)]
    else:
        obstacles=[dict(a=np.asarray(w[0]).tolist(),b=np.asarray(w[1]).tolist(),radius=cfg.wall_radius,curvature=0.,interior_sign=0.) for w in env.walls]
    return dict(positions=env.positions.tolist(),velocities=env.velocities.tolist(),goals=env.goals.tolist(),flow=flow.tolist(),
        radius=[cfg.agent_radius]*n,goal_tolerance=[cfg.goal_tolerance]*n,obstacles=obstacles,policy_origin=[0.,0.],policy_axis=[1.,0.],
        policy_axis_required=rt.scenario!='ring_exchange',flow_committed=False,remaining_fraction=1.,remaining_seconds=cfg.dt*cfg.max_steps,
        max_seconds=cfg.dt*cfg.max_steps,dt=cfg.dt,max_speed=cfg.max_speed,wall_margin=d.get('wall_collision_margin',d.get('collision_margin',0.)),
        agent_margin=d.get('agent_collision_margin',0.),monitor_active=d.get('terminate_on_deadlock',False),
        progress_window_seconds=d.get('progress_window_seconds',0.),deadlock_hold_seconds=d.get('deadlock_hold_seconds',0.),
        progress_epsilon=d.get('progress_epsilon',0.),speed_epsilon_fraction=d.get('speed_epsilon_fraction',0.),monitor_elapsed=0.,monitor_history_empty=True)


def response(rt,env,key,eta,chain):
    obs,safety,flow,safe,delta,_=rt.prepare(env,key,eta)
    if chain=='TT':action,_=safety.project(safe+delta)
    else:
        assert chain=='FF';action,_=rt.integrate(env,obs,key,delta,safety)
    safety.assert_safe(action)
    return flow,safe,action


def distances(pos):
    return np.array([np.linalg.norm(pos[i]-pos[j]) for i in range(len(pos)) for j in range(i)])


class Measure:
    def __init__(self,rt,chain):
        self.rt=rt;self.chain=chain;self.nominal={}

    def trajectory(self,state,p,eta,root):
        import jax
        rt=self.rt;env=rt.make_env(state);v=rt.config.max_speed;dt=rt.config.dt
        initial=np.linalg.norm(env.goals-env.positions,axis=1).sum();prev=None;rows=[];agents=[]
        _,offset,_=goal_design(p);ex=offset[0];ey=np.stack([-ex[:,1],ex[:,0]],-1)
        local=lambda a:np.stack([(a*ex).sum(-1),(a*ey).sum(-1)],-1)/v
        for t in range(20):
            key=jax.random.fold_in(jax.random.PRNGKey(root),t)
            flow,safe,action=response(rt,env,key,eta,self.chain)
            agents.append(np.concatenate([local(flow),local(safe),local(action),np.linalg.norm(safe-flow,axis=1)[:,None]/v,np.linalg.norm(action-safe,axis=1)[:,None]/v],-1))
            before=np.linalg.norm(env.goals-env.positions,axis=1).sum();dist=distances(env.positions)
            row=dict(progress=0.,pair_clearance=dist.min(),pair_closing=0.,safety=np.linalg.norm(safe-flow)/v,
                correction=np.linalg.norm(action-safe)/v,flow_change=0. if prev is None else np.linalg.norm(flow-prev)/v,
                speed=np.linalg.norm(action,axis=1).mean()/v)
            prev=flow;env.step(action);after=np.linalg.norm(env.goals-env.positions,axis=1).sum()
            row['progress']=(before-after)/(v*dt*len(env.positions));row['pair_closing']=((dist-distances(env.positions))/(v*dt)).max()
            rows.append(row)
            if env.done:break
        final=np.linalg.norm(env.goals-env.positions,axis=1).sum()
        return rows,env.positions.copy(),(initial-final)/max(initial,1e-12),np.mean(agents,axis=0)

    def h20(self,state,p,eta):
        outputs=[];agent=[]
        def summarize(rows,goal):
            a={k:np.array([r[k] for r in rows]) for k in rows[0]}
            return [a['progress'].sum(),a['progress'][len(rows)//2:].mean(),goal,a['pair_clearance'].min(),a['pair_closing'].max(),
                a['safety'].mean(),a['safety'].max(),(a['safety']>1e-6).mean(),a['speed'][-1],a['flow_change'].mean()],a
        for root in ROOTS[:2]:
            key=(state['state_uid'],root)
            if key not in self.nominal:self.nominal[key]=self.trajectory(state,p,np.zeros(3),root)
            nom,np0,ng,na=self.nominal[key]
            corr,cp,cg,ca0=self.trajectory(state,p,eta,root)
            n,_=summarize(nom,ng);c,ca=summarize(corr,cg)
            outputs.append(n+c[:8]+[ca['correction'].mean(),ca['correction'].max(),(ca['correction']>1e-6).mean(),c[0]-n[0],np.linalg.norm(cp-np0,axis=1).mean(),c[3]-n[3]])
            agent.append(np.concatenate((na,ca0),1).mean(0))
        return np.r_[np.mean(outputs,0),np.mean(agent,0)]

    def goal(self,state,p,eta,moving):
        import jax
        radius,offsets,_=goal_design(p);assert radius is not None
        rt=self.rt;v=rt.config.max_speed;out=[]
        for di,offset in enumerate(offsets):
            pp=copy.deepcopy(state);pos=np.asarray(p['goals'])+radius*offset
            vel=-.5*v*offset if moving else np.zeros_like(offset)
            pp['physical'].update(positions=pos.tolist(),velocities=vel.tolist())
            env=rt.make_env(pp)
            # Toy's native reset intentionally zeros velocities. For a synthetic
            # controller query (not a task state), restore the queried velocity.
            env.velocities=vel.copy();assert clearance(p,env.positions)>1e-4
            along=env.goals-env.positions;along/=np.linalg.norm(along,axis=1,keepdims=True);rr=[]
            for root in ROOTS:
                flow,safe,action=response(rt,env,jax.random.fold_in(jax.random.PRNGKey(root),di),eta,self.chain)
                a,b,c=[(u*along).sum(1)/v for u in (flow,safe,action)];rr.append([a.mean(),b.mean(),c.mean(),c.min()])
            out.extend(np.mean(rr,0))
        return np.asarray(out)

    def one(self,state,p,eta):
        x=np.zeros(76,np.float32);errors=[];radius,_,_=goal_design(p);x[72]=radius/.30 if radius else 0.
        for a,b,flag,fn in ((0,40,73,lambda:self.h20(state,p,eta)),(40,56,74,lambda:self.goal(state,p,eta,False)),(56,72,75,lambda:self.goal(state,p,eta,True))):
            try:
                values=fn();assert values.shape==(b-a,) and np.isfinite(values).all();x[a:b]=values;x[flag]=1.
            except Exception as ex:errors.append(dict(group=a,error=f'{type(ex).__name__}: {ex}'))
        return x,errors


def run(worker,workers):
    from validation_stage2.run_rollouts import make_runtime
    p=protocol();items=[];docs=[];features=[];indices=[];current=None
    for i,state in enumerate(p['states']):
        if i%workers!=worker:continue
        if current!=state['scenario']:
            rt=make_runtime(state['scenario']);current=state['scenario'];measures={chain:Measure(rt,chain) for chain in ('TT','FF')}
        pp=physical(rt,state);features.append(rep.entities(pp));indices.append(i)
        for chain in ('TT','FF'):
            for j,eta in enumerate(p['eta']):
                start=time.perf_counter();c,errors=measures[chain].one(state,pp,np.array(eta))
                items.append(dict(state_index=i,chain=chain,eta_index=j,context=c.tolist(),seconds=time.perf_counter()-start))
                docs.extend(dict(state_index=i,chain=chain,eta_index=j,**e) for e in errors)
    dest=ROOT/'contexts';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/f'entities_worker{worker}.npz',indices=np.array(indices),**rep.batch(features))
    write(dest/f'worker{worker}.json',dict(worker=worker,items=items,errors=docs,context_sha256=sha(__file__),
        protocol_sha256=sha(ROOT/'protocol.json'),representation_sha256=sha(rep.__file__),new_task_rollouts=0))
    print(dict(worker=worker,context_pairs=len(items),invalid_groups=len(docs)),flush=True)


def smoke():
    from validation_stage2.run_rollouts import make_runtime
    import jax
    p=protocol();out=[]
    for scene in ('toy_give_way','ring_exchange'):
        state=next(s for s in p['states'] if s['scenario']==scene and s['metadata']['split']=='train');rt=make_runtime(scene)
        pp=physical(rt,state);eta=np.array(p['eta'][0]);h=rep.entities(pp)
        for chain,mode in (('TT','old'),('FF','field')):
            key=jax.random.PRNGKey(ROOTS[0]);a=response(rt,rt.make_env(state),key,eta,chain)[2]
            b=rt.action(rt.make_env(state),key,eta,mode)[0];err=float(np.max(abs(a-b)));assert err<1e-12
            x,errors=Measure(rt,chain).one(state,pp,eta);assert not errors,(scene,chain,errors)
            out.append(dict(scene=scene,chain=chain,action_parity_error=err,valid_flags=x[73:].tolist(),width=len(x),goal_radius=float(x[72]*.30)))
    freeze(ROOT/'context_smoke.json',dict(cases=out,context_sha256=sha(__file__),representation_sha256=sha(rep.__file__),
        same_76_channel_definitions=True,nominal='Each chain with eta=0; FF nominal includes field safety, not TT terminal projection',
        h_shared_between_chains=True,max_temporal_probe_steps=20,no_success_label_or_scene_ID=True,new_task_rollouts=0))
    print(out,flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('run','smoke'));a.add_argument('--worker',type=int,default=0);a.add_argument('--workers',type=int,default=5);q=a.parse_args()
    smoke() if q.action=='smoke' else run(q.worker,q.workers)
