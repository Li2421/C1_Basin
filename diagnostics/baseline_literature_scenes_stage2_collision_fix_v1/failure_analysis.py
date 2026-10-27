"""Extend the frozen v1 categories using saved evidence; never calls actors."""
from experiment import *
import numpy as np
from scipy.optimize import nnls,lsq_linear
from legacy_failure_analysis import cone_escape
from projection_adapter import constraints
from distributed_safety import LocalSafety,responsibility
from shared_control.hard_projection import HardProjectionConfig
from states import LiteratureConfig,walls

def residual(u,A):
    u=np.asarray(u,float).ravel();A=np.asarray(A,float).reshape(-1,len(u));norm=np.linalg.norm(u)
    if norm<=1e-12:return 1.
    if not len(A):return 1.
    try:_,r=nnls(A.T,-u,maxiter=10000)
    except RuntimeError:
        fit=lsq_linear(A.T,-u,bounds=(0,np.inf),tol=1e-12,max_iter=10000)
        if not fit.success:raise RuntimeError('Posthoc cone diagnostic did not converge')
        r=np.linalg.norm(u+A.T@fit.x)
    return float(r/norm)

def measures(raw,z):
    c=LiteratureConfig(**raw['case']['scenario_payload']['config']);n=c.num_agents
    nom=np.asarray(z['nominal']);safe=np.asarray(z['safe']);T=len(nom)
    begin=max(0,T-int(np.ceil(10/c.dt)))
    local=np.full((T-begin,n),np.nan);canonical_joint=np.full(T-begin,np.nan)
    actual_joint=np.full(T-begin,np.nan);pair=np.zeros((T-begin,n),bool);wallonly=np.zeros_like(pair)
    layer=raw['case'].get('safety_layer',raw.get('safety_layer','central'))
    hp=HardProjectionConfig();details=[];geometry=[]
    for t in range(begin,T):
        snap=dict(positions=z['positions'][t],walls=walls(c.scene),config=c.to_dict())
        A,lower,kinds=constraints(snap,hp);v=safe[t]
        ids=np.flatnonzero(A@v.ravel()-lower<=1e-7)
        canonical_joint[t-begin]=residual(nom[t],A[ids])
        labels=[];geometry_t=[]
        for i in range(n):
            if layer=='distributed':
                f=LocalSafety(snap,i,hp,raw['case'].get('responsibility',responsibility(c.scene)),(A,lower,kinds))
                own=np.flatnonzero(f.A@v[i]-f.lower<=1e-7)
                rows=f.A[own];labels_i=[f.kinds[k] for k in own];bounds=f.lower[own]
            else:
                own=[k for k in ids if np.any(A[k,2*i:2*i+2])]
                rows=A[own,2*i:2*i+2];labels_i=[kinds[k] for k in own]
                bounds=lower[own]-(A[own]@v.ravel()-rows@v[i])
            local[t-begin,i]=residual(nom[t,i],rows)
            pair[t-begin,i]=any(k[0]=='pair' for k in labels_i)
            wallonly[t-begin,i]=bool(labels_i) and all(k[0]=='wall' for k in labels_i)
            labels.append(labels_i)
            geometry_t.append(dict(A=rows.tolist(),conditional_lower=np.asarray(bounds).tolist(),labels=labels_i))
        actual_joint[t-begin]=(float(np.linalg.norm(np.linalg.norm(nom[t],axis=-1)*local[t-begin])/max(np.linalg.norm(nom[t]),1e-12)) if np.linalg.norm(nom[t])>1e-12 else 1.) if layer=='distributed' else canonical_joint[t-begin]
        details.append(labels)
        geometry.append(geometry_t)
    return dict(begin=begin,local=local,canonical_joint=canonical_joint,
                actual_product_joint=actual_joint,has_pair=pair,wall_only=wallonly,active=details,geometry=geometry)

def diagnose(raw,z,W=5.,tau=.05,kind='local',measurement=None):
    s=raw['case']['state'];c=LiteratureConfig(**raw['case']['scenario_payload']['config'])
    p=z['positions'];g=np.asarray(s['goals']);nom=z['nominal'];T=len(nom)
    if raw['numerical_failure']:return ['unknown']*s['N']
    arrived=np.linalg.norm(p[-1]-g,axis=-1)<=c.goal_tolerance
    count=max(1,int(np.ceil(W/c.dt)));categories=[]
    if T>=count:m=measurement or measures(raw,z)
    for i in range(s['N']):
        if arrived[i]:categories.append('arrived');continue
        if raw['collision']:categories.append('collision');continue
        if T<count:categories.append('insufficient_window');continue
        trajectory=p[-count-1:,i];quiet=np.linalg.norm(trajectory-trajectory[0],axis=-1).max()<c.agent_radius
        nonzero=np.linalg.norm(nom[-count:,i],axis=-1)>1e-6
        changed=np.asarray(z['projection_rewrite'][-count:,i],bool)
        if quiet:
            if kind=='local':
                good=changed&nonzero&(m['local'][-count:,i]<tau)
                if np.mean(good&m['has_pair'][-count:,i])>=.9:categories.append('robot_filter_equilibrium_deadlock');continue
                if np.mean(good&m['wall_only'][-count:,i])>=.9:categories.append('wall_filter_equilibrium');continue
            else:
                cone=m['canonical_joint'] if kind=='canonical_joint' else m['actual_product_joint']
                if np.mean(changed&nonzero&(cone[-count:]<tau))>=.9:categories.append('filter_equilibrium_deadlock');continue
            active=m['active'][-count:]
            blockers=any(k[0]=='pair' and i in k[1:] and arrived[k[2] if k[1]==i else k[1]] for labels in active for k in labels[i])
            if blockers:categories.append('blocked_by_arrived');continue
            if np.mean(~nonzero)>=.9:categories.append('zero_nominal');continue
            categories.append('other_stationary')
        else:
            length=np.linalg.norm(np.diff(trajectory,axis=0),axis=-1).sum();displacement=np.linalg.norm(trajectory[-1]-trajectory[0])
            categories.append('oscillation' if length>5*displacement else 'slow_progress')
    return categories

def summary(raw,z):
    # Unknowns remain unknown, and diagnostic errors never relabel physics.
    if raw['numerical_failure']:return dict(local=['unknown']*raw['case']['state']['N'],grid=[])
    c=LiteratureConfig(**raw['case']['scenario_payload']['config'])
    arrived=np.linalg.norm(z['positions'][-1]-np.asarray(raw['case']['state']['goals']),axis=-1)<=c.goal_tolerance
    if np.all(arrived) or raw['collision'] or len(z['nominal'])<int(np.ceil(2.5/c.dt)):
        grid=[]
        for W in (2.5,5.,10.):
            cats=['arrived' if a else 'collision' if raw['collision'] else 'insufficient_window' for a in arrived]
            for tau in (.02,.05,.1):grid.append(dict(window_seconds=W,cone_tolerance=tau,local=cats,canonical_joint=cats,actual_product_joint=cats))
        cats=grid[0]['local']
        return dict(local=cats,canonical_joint=cats,actual_product_joint=cats,grid=grid,cone_computation='Terminal categories resolve all agents without cone optimization')
    m=measures(raw,z);grid=[]
    for W in (2.5,5.,10.):
        for tau in (.02,.05,.1):
            grid.append(dict(window_seconds=W,cone_tolerance=tau,
                **{kind:diagnose(raw,z,W,tau,kind,m) for kind in ('local','canonical_joint','actual_product_joint')}))
    primary=next(r for r in grid if r['window_seconds']==5. and r['cone_tolerance']==.05)
    census=[];count=min(len(z['nominal']),int(np.ceil(5/c.dt)))
    for i,category in enumerate(primary['local']):
        if category not in ('robot_filter_equilibrium_deadlock','wall_filter_equilibrium'):continue
        for t in range(len(z['nominal'])-count,len(z['nominal'])):
            j=t-m['begin'];geo=m['geometry'][j][i]
            census.append(dict(agent=i,step=t,category=category,nominal=z['nominal'][t,i].tolist(),safe=z['safe'][t,i].tolist(),
                safe_speed=float(np.linalg.norm(z['safe'][t,i])),local_cone_residual=float(m['local'][j,i]),
                canonical_joint_cone_residual=float(m['canonical_joint'][j]),actual_joint_cone_residual=float(m['actual_product_joint'][j]),**geo))
    return dict(local=primary['local'],canonical_joint=primary['canonical_joint'],actual_product_joint=primary['actual_product_joint'],grid=grid,census=census)
