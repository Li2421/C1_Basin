"""User clarification: preserve all snapshots; test only the invariant domain."""
import resume_bootstrap
from experiment import *
from unit_gates import project
from states import config
from single_integrator.environment import point_segment_distance
import numpy as np

def interval_clearance(p,q,radius):
    ii,jj=np.triu_indices(len(p),1)
    return float(point_segment_distance(np.zeros(2),p[ii]-p[jj],q[ii]-q[jj]).min()-radius)

def run(scene,chunk):
    gamma=.3 if scene=='lit_narrow_passage' else .2
    states=read(HERE/f'FLOW_STATES_{scene}.json');records=[]
    counts=dict(total=0,precondition_satisfied=0,precondition_not_satisfied=0,feasible=0,infeasible=0,unknown=0,safety_failures=0,permutation_failures=0)
    maximum_permutation_error=0.;minimum_clearance=np.inf
    previous_errors={5383} if scene=='lit_narrow_passage' else {7639}
    for k in range(chunk*1000,(chunk+1)*1000):
        rng=np.random.default_rng(np.random.SeedSequence([2026102301,('lit_narrow_passage','lit_crowded_random','lit_circle_five').index(scene),k]))
        s=states[int(rng.integers(len(states)))];c=config(scene,s['N']);p=np.asarray(s['positions'],float)
        theta=rng.uniform(-np.pi,np.pi,s['N']);speed=np.zeros(s['N']) if k<5000 else c.max_speed*rng.random(s['N'])
        v=speed[:,None]*np.column_stack([np.cos(theta),np.sin(theta)])
        theta2=rng.uniform(-np.pi,np.pi,s['N']);desired=c.max_speed*rng.random(s['N'])[:,None]*np.column_stack([np.cos(theta2),np.sin(theta2)])
        pred=p+c.dt*v;ii,jj=np.triu_indices(len(p),1);predicted_gap=float(np.linalg.norm(pred[ii]-pred[jj],axis=-1).min()-.303101)
        prior_clearance=interval_clearance(p,pred,.3)
        valid=predicted_gap>=0 and prior_clearance>0
        counts['total']+=1;counts['precondition_satisfied' if valid else 'precondition_not_satisfied']+=1
        r=dict(index=k,N=s['N'],state=s['index'],precondition_satisfied=valid,predicted_gap=predicted_gap,prior_interval_clearance=prior_clearance)
        if k in previous_errors:
            counts['unknown']+=1;r.update(status='unknown',source='Preserved original numerical error, not rerun');records.append(r);continue
        if not valid:r['status']='outside_precondition';records.append(r);continue
        try:a,ds=project(p,v,desired,c,gamma)
        except Exception as exc:
            counts['unknown']+=1;r.update(status='unknown',error=str(exc));records.append(r);continue
        feasible=all(d['maximum_violation']<=1e-7 for d in ds)
        counts['feasible' if feasible else 'infeasible']+=1
        if not feasible:r['status']='infeasible';records.append(r);continue
        nxt=pred+c.dt*(v+c.dt*a);clear=interval_clearance(pred,nxt,.3)
        perm=np.arange(len(p))[::-1]
        try:
            ap,_=project(p[perm],v[perm],desired[perm],c,gamma)
            cp=interval_clearance(pred[perm],pred[perm]+c.dt*(v[perm]+c.dt*ap),.3)
            error=float(abs(a[perm]-ap).max());maximum_permutation_error=max(maximum_permutation_error,error)
            permutation_failure=error>1e-4 or (clear<=0)!=(cp<=0)
        except Exception as exc:
            counts['unknown']+=1;r.update(status='unknown_permutation',error=str(exc));records.append(r);continue
        minimum_clearance=min(minimum_clearance,clear)
        counts['safety_failures']+=clear<=0;counts['permutation_failures']+=permutation_failure
        r.update(status='feasible',next_interval_clearance=clear,permutation_error=error,permutation_failure=permutation_failure)
        records.append(r)
    out=dict(scene=scene,chunk=chunk,gamma=gamma,counts=counts,maximum_permutation_error=maximum_permutation_error,
        minimum_next_interval_clearance=minimum_clearance,records=records,
        passed=counts['safety_failures']==0 and counts['permutation_failures']==0,
        clarification='User 20261023 1.4: test k+1 to k+2 only for safe predicted configuration and safe prior interval; unknowns preserved')
    write(HERE/'CLARIFIED_UNITS'/f'{scene}_{chunk}.json',out)
    print(scene,chunk,counts,'passed',out['passed'],flush=True)
if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))
