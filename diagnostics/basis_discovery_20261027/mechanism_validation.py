"""Finite-difference C2 check and T5 threshold/conditioning audit on saved states."""
import json
from types import SimpleNamespace
import numpy as np
from runner import OUT,RN,inputs,dump
from shared_control.basis_dictionary import evaluate,RoleState
from shared_control.diagnostic_corrector import bounded_rows


def main():
    results=[]
    for item in json.load(open(OUT/'static_states.json')):
        c=item['case'];tr=np.load(item['source_trace']);t=item['step'];p=np.asarray(item['positions']);g=np.asarray(item['goals']);safe=np.asarray(item['safe']);nominal=np.asarray(item['nominal'])
        scene=inputs()[0][c['scene']];cfg=RN.LiteratureConfig(**scene['config']);walls=np.asarray(scene['walls']);geo=json.loads(str(tr['first_geometry_json']))[t]
        fs=[SimpleNamespace(A=np.asarray(x['A']),lower=np.asarray(x['lower']),kinds=x['labels'])for x in geo]
        b=np.stack([RN.distributed_fields(p,g,safe,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg))for i in range(len(p))])
        def clearance(q):
            changed=[]
            for i,f in enumerate(fs):
                A=f.A.copy()
                for j,label in enumerate(f.kinds):
                    if label[0]=='pair':A[j]=2*(q[i]-q[int(label[2])])
                    elif label[0]=='wall':
                        a,z=walls[int(label[2])];d=z-a;foot=a+np.clip((q[i]-a)@d/(d@d),0,1)*d
                        delta=q[i]-foot;A[j]=delta/np.linalg.norm(delta)
                changed.append(SimpleNamespace(A=A,lower=f.lower,kinds=f.kinds))
            col,meta=evaluate(q,g,safe,nominal,b,changed,walls,cfg,RoleState(),names=['C2'])
            return meta['clearance']['D'],col['C2']
        D,out=clearance(p);numeric=np.zeros_like(p);eps=1e-6
        for i in range(len(p)):
            for axis in range(2):
                plus=p.copy();minus=p.copy();plus[i,axis]+=eps;minus[i,axis]-=eps
                numeric[i,axis]=(clearance(plus)[0]-clearance(minus)[0])/(2*eps)
        error=float(np.max(np.abs(out-bounded_rows(cfg.max_speed*numeric,cfg.max_speed))))
        # Re-evaluate active nullspaces without changing the deployed tolerance.
        stability=[]
        for relative_tau in (.01,.05,.1):
            for svd_tol in (1e-8,1e-10,1e-12):
                ranks=[];condition=[];residual=[]
                for i,f in enumerate(fs):
                    norm=np.linalg.norm(f.A,axis=1);valid=norm>1e-14
                    slack=(f.A@safe[i]-f.lower)/np.maximum(norm,1e-30)
                    a=f.A[valid&(slack<=relative_tau*cfg.max_speed)]/norm[valid&(slack<=relative_tau*cfg.max_speed),None]
                    sn=np.linalg.norm(safe[i])
                    if sn>=cfg.max_speed*(1-relative_tau)and sn>1e-14:a=np.vstack([a,safe[i]/sn])
                    if len(a):
                        _,s,vt=np.linalg.svd(a,full_matrices=False);rank=int(sum(s>max(1e-12,svd_tol*s[0])));v=vt[:rank]
                        d=np.array([-b[i,0,1],b[i,0,0]]);u=d-v.T@(v@d)
                        condition.append(float(s[0]/s[rank-1])if rank else None);residual.append(float(np.linalg.norm(a@u)))
                    else:rank=0;condition.append(None);residual.append(0.)
                    ranks.append(rank)
                stability.append(dict(near_relative=relative_tau,svd_relative=svd_tol,ranks=ranks,condition=condition,max_nullspace_residual=max(residual)))
        results.append(dict(case=c,C2_fd_max_error=error,C2_fd_epsilon=eps,T5_sensitivity=stability))
    max_error=max(r['C2_fd_max_error']for r in results);assert max_error<1e-5,max_error
    x=dict(passed=True,max_C2_fd_error=max_error,states=results,interpretation='Local fixed-neighbor-set finite differences; numerical verification, not global smoothness at nearest-feature or neighbor-set changes.')
    dump(OUT/'mechanism_validation.json',x);print('C2 finite differences PASS',max_error)

if __name__=='__main__':main()
