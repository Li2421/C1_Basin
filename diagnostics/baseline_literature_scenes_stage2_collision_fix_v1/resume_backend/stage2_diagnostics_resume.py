"""Stage-two diagnostics in the actual acceleration/front-velocity coordinates."""
from experiment import *
import numpy as np
from failure_analysis import residual
from states import LiteratureConfig,walls
from stage2_joint_resume import local_rows,coupled_active_rows

def summary(raw,z):
    c=LiteratureConfig(**raw['case']['scenario_payload']['config']);n=c.num_agents
    if raw['numerical_failure']:return dict(local=['unknown']*n,direct_stationary=[False]*n,grid=[])
    p=z['positions'];g=np.asarray(raw['case']['state']['goals']);nom=z['nominal'];T=len(nom)
    arrived=np.linalg.norm(p[-1]-g,axis=-1)<=c.goal_tolerance
    active=json.loads(str(z['local_active_json']));count10=int(np.ceil(10/c.dt));begin=max(0,T-count10)
    cones=np.ones((T-begin,n));has_pair=np.zeros_like(cones,dtype=bool);wallonly=np.zeros_like(has_pair)
    coupled=np.ones(T-begin);product=np.ones(T-begin)
    if not np.all(arrived) and not raw['collision']:
        for t in range(begin,T):
            local=[]
            for i in range(n):
                A,kinds=local_rows(z,t,i,c,active[t][i]);local.append((A,kinds));cones[t-begin,i]=residual(nom[t,i],A)
                has_pair[t-begin,i]=any(k[0]=='pair' for k in kinds)
                wallonly[t-begin,i]=bool(kinds) and all(k[0]=='wall' for k in kinds)
            coupled[t-begin]=residual(nom[t],coupled_active_rows(z,t,c,walls(c.scene),local))
            norm=np.linalg.norm(nom[t]);product[t-begin]=float(np.linalg.norm(np.linalg.norm(nom[t],axis=-1)*cones[t-begin])/norm) if norm>1e-12 else 1.
    grid=[]
    for W in (2.5,5.,10.):
        count=int(np.ceil(W/c.dt))
        for tau in (.02,.05,.1):
            cats=[];coupled_cats=[];product_cats=[]
            for i in range(n):
                if arrived[i]:cats.append('arrived');continue
                if raw['collision']:cats.append('collision');continue
                if T<count:cats.append('insufficient_window');continue
                trajectory=p[-count-1:,i];quiet=np.max(np.linalg.norm(trajectory-trajectory[0],axis=-1))<c.agent_radius
                nonzero=np.linalg.norm(nom[-count:,i],axis=-1)>1e-6
                good=np.asarray(z['projection_rewrite'][-count:,i],bool)&nonzero&(cones[-count:,i]<tau)
                if quiet:
                    if np.mean(good&has_pair[-count:,i])>=.9:cats.append('robot_filter_equilibrium_deadlock');continue
                    if np.mean(good&wallonly[-count:,i])>=.9:cats.append('wall_filter_equilibrium');continue
                    blockers=any(k[0]=='pair' and arrived[k[2] if k[1]==i else k[1]] for item in active[-count:] for k in item[i]['kinds'])
                    cats.append('blocked_by_arrived' if blockers else 'zero_nominal' if np.mean(~nonzero)>=.9 else 'other_stationary')
                else:
                    length=np.linalg.norm(np.diff(trajectory,axis=0),axis=-1).sum();displacement=np.linalg.norm(trajectory[-1]-trajectory[0])
                    cats.append('oscillation' if length>5*displacement else 'slow_progress')
            grid.append(dict(window_seconds=W,cone_tolerance=tau,local=cats))
            for target,values in ((coupled_cats,coupled),(product_cats,product)):
                for i,category in enumerate(cats):
                    if category in ('arrived','collision','insufficient_window'):target.append(category);continue
                    trajectory=p[-count-1:,i];quiet=np.max(np.linalg.norm(trajectory-trajectory[0],axis=-1))<c.agent_radius
                    good=np.asarray(z['projection_rewrite'][-count:,i],bool)&(np.linalg.norm(nom[-count:,i],axis=-1)>1e-6)&(values[-count:]<tau)
                    if quiet and np.mean(good)>=.9:target.append('filter_equilibrium_deadlock')
                    else:target.append('other_stationary' if category in ('robot_filter_equilibrium_deadlock','wall_filter_equilibrium') else category)
            grid[-1].update(canonical_joint=coupled_cats,actual_product_joint=product_cats)
    count=int(np.ceil(5/c.dt));direct=[]
    for i in range(n):
        enough=T>=count
        direct.append(bool(not arrived[i] and enough and np.max(np.linalg.norm(p[-count-1:,i]-p[-count-1,i],axis=-1))<c.agent_radius and np.mean(np.linalg.norm(z['velocities'][-count:,i],axis=-1)<=1e-3*c.max_speed)>=.9))
    primary=next(r for r in grid if r['window_seconds']==5. and r['cone_tolerance']==.05)
    return dict(local=primary['local'],canonical_joint=primary['canonical_joint'],actual_product_joint=primary['actual_product_joint'],direct_stationary=direct,grid=grid,
        coordinates='front_point_velocity' if c.scene=='lit_doorway' else 'acceleration',constraint_violation_steps=raw.get('constraint_violation_steps',0),
        violation_and_4a_prime_overlap=bool(raw.get('constraint_violation_steps',0) and 'robot_filter_equilibrium_deadlock' in primary['local']))
