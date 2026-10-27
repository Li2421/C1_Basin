"""v2 taxonomy on saved first projection geometry; static/moving subcounts."""
import bootstrap
from experiment import *
import numpy as np
from failure_analysis import residual

def summary(raw,z):
    c=raw['case'];s=c['state'];cfg=c['scenario_payload']['config'];n=s['N'];T=len(z['nominal']);dt=cfg['dt'];g=np.asarray(s['goals']);arrived=np.linalg.norm(z['positions'][-1]-g,axis=-1)<=cfg['goal_tolerance']
    if raw['numerical_failure']:return dict(local=['unknown']*n,grid=[],static_wall_agents=0,moving_wall_agents=0)
    geometry=json.loads(str(z['first_geometry_json']));begin=max(0,T-int(np.ceil(10/dt)));local=np.ones((T-begin,n));canonical=np.ones(T-begin);actual=np.ones(T-begin);pair=np.zeros((T-begin,n),bool);wall=np.zeros_like(pair);static=np.zeros_like(pair);moving=np.zeros_like(pair);active=[]
    if not np.all(arrived) and not raw['collision']:
        for t in range(begin,T):
            nom=z['nominal'][t];safe=z['safe'][t];labels=[];joint={}
            for i,item in enumerate(geometry[t]):
                A=np.asarray(item['A']);lower=np.asarray(item['lower']);eff=np.asarray(item['effective_lower']);ids=np.flatnonzero(A@safe[i]-eff<=1e-7);lab=[item['labels'][k] for k in ids];labels.append(lab)
                local[t-begin,i]=residual(nom[i],A[ids]);pair[t-begin,i]=any(k[0]=='pair' for k in lab);wall[t-begin,i]=bool(lab) and all(k[0] in ('wall','moving') for k in lab);static[t-begin,i]=wall[t-begin,i] and any(k[0]=='wall' for k in lab);moving[t-begin,i]=wall[t-begin,i] and any(k[0]=='moving' for k in lab)
                for a,b,k in zip(A,lower,item['labels']):
                    if k[0]=='tautology':continue
                    key=tuple(['pair',*sorted(k[1:])]) if k[0]=='pair' else tuple(k)
                    if key not in joint:joint[key]=[np.zeros(2*n),0.]
                    joint[key][0][2*i:2*i+2]+=a;joint[key][1]+=b
            aa=np.asarray([v[0] for v in joint.values()]).reshape(-1,2*n);ll=np.asarray([v[1] for v in joint.values()]);ii=np.flatnonzero(aa@safe.ravel()-ll<=1e-7)
            canonical[t-begin]=residual(nom,aa[ii]);actual[t-begin]=float(np.linalg.norm(np.linalg.norm(nom,axis=-1)*local[t-begin])/max(np.linalg.norm(nom),1e-12)) if np.linalg.norm(nom)>1e-12 else 1.;active.append(labels)
    def diagnose(W,tau,kind):
        count=max(1,int(np.ceil(W/dt)));out=[]
        for i in range(n):
            if arrived[i]:out.append('arrived');continue
            if raw['collision']:out.append('collision');continue
            if T<count:out.append('insufficient_window');continue
            traj=z['positions'][-count-1:,i];quiet=np.linalg.norm(traj-traj[0],axis=-1).max()<cfg['agent_radius'];nonzero=np.linalg.norm(z['nominal'][-count:,i],axis=-1)>1e-6;changed=np.asarray(z['projection_rewrite'][-count:,i],bool)
            if quiet:
                cone=local[-count:,i] if kind=='local' else canonical[-count:] if kind=='canonical_joint' else actual[-count:];good=changed&nonzero&(cone<tau)
                if kind=='local':
                    if np.mean(good&pair[-count:,i])>=.9:out.append('robot_filter_equilibrium_deadlock');continue
                    if np.mean(good&wall[-count:,i])>=.9:out.append('wall_filter_equilibrium');continue
                elif np.mean(good)>=.9:out.append('filter_equilibrium_deadlock');continue
                blockers=any(k[0]=='pair' and i in k[1:] and arrived[k[2] if k[1]==i else k[1]] for labels in active[-count:] for k in labels[i])
                if blockers:out.append('blocked_by_arrived');continue
                out.append('zero_nominal' if np.mean(~nonzero)>=.9 else 'other_stationary')
            else:
                length=np.linalg.norm(np.diff(traj,axis=0),axis=-1).sum();displacement=np.linalg.norm(traj[-1]-traj[0]);out.append('oscillation' if length>5*displacement else 'slow_progress')
        return out
    grid=[dict(window_seconds=W,cone_tolerance=tau,**{kind:diagnose(W,tau,kind) for kind in ('local','canonical_joint','actual_product_joint')}) for W in (2.5,5.,10.) for tau in (.02,.05,.1)]
    primary=next(r for r in grid if r['window_seconds']==5. and r['cone_tolerance']==.05);count=max(1,int(np.ceil(5/dt)));subtypes=[]
    for i,cat in enumerate(primary['local']):
        good=z['projection_rewrite'][-count:,i]&(np.linalg.norm(z['nominal'][-count:,i],axis=-1)>1e-6)&(local[-count:,i]<.05) if T>=count else np.zeros(0,bool)
        subtypes.append(dict(agent=i,static=bool(cat=='wall_filter_equilibrium' and np.mean(good&static[-count:,i])>=.9),moving=bool(cat=='wall_filter_equilibrium' and np.mean(good&moving[-count:,i])>=.9)))
    return dict(local=primary['local'],canonical_joint=primary['canonical_joint'],actual_product_joint=primary['actual_product_joint'],grid=grid,wall_subtypes=subtypes,static_wall_agents=sum(r['static'] for r in subtypes),moving_wall_agents=sum(r['moving'] for r in subtypes),joint_reference='Original full reciprocal sum within sensing; actual product uses saved effective first rows',new_rollouts=0)
