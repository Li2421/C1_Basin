"""Supplementary frozen-state falsification checks; no controller mutations."""
import json
import time
from pathlib import Path
import numpy as np
from audit_c1_joint_witness_risk import ROOT, LocalRisk, save, CBFConfig, barrier_constraints, Config, GiveWayEnv

OUT=ROOT/'results/c1_joint_witness_risk_audit'

def main():
    dimensions=[]
    for m in (4,16,64):
        e=np.eye(m);A=np.vstack([e[0],e[1:],-e[1:]])
        q=np.zeros(m);q[:2]=[-.25,np.sqrt(1-.25**2)]
        start=time.perf_counter();model=LocalRisk(A,np.zeros(len(A)),np.full(m//2,.5))
        score=model.score(.1*q)
        assert abs(score['B']-1)<1e-6 and abs(score['M']+.25)<1e-4
        dimensions.append(dict(m=m,seconds=time.perf_counter()-start,**score))
    index=json.loads((OUT/'index.json').read_text());checks=[]
    # Exploratory selection, development partition only: first trajectory per outcome,
    # then its largest local score. No parameters or held-out scores are refitted.
    for source in ['primary','extended_c1']:
        for outcome in ['success','safe_deadlock','other_timeout']:
            records=[r for r in index if r['source']==source and r['outcome']==outcome and r['split']=='development']
            if not records:continue
            rec=records[0];z=np.load(ROOT/rec['score_path']);k=int(np.nanargmax(z['risk']));t=int(z['idx'][k])
            metadata=json.loads((ROOT/rec['config_path']).read_text())
            plant=Config(**metadata['environment']);cbf=CBFConfig(**metadata['cbf']);env=GiveWayEnv(plant)
            A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=plant.to_dict()),cbf)
            model=LocalRisk(A,b,np.full(2,plant.max_speed));v=z['candidate'][t];a=np.linalg.norm(v);q=v/a
            gradients=[]
            for h in (1e-4,3e-5):
                g=np.array([(model.score(a*(q+h*e))['risk']-model.score(a*(q-h*e))['risk'])/(2*h) for e in np.eye(4)])
                gradients.append(g-q*np.dot(q,g))
            g=gradients[-1];qn=q-.02*g/max(np.linalg.norm(g),1e-12);qn/=np.linalg.norm(qn)
            before=model.score(v);after=model.score(a*qn)
            u0=model.physical(v);u1=model.physical(a*qn)
            goals=z['goals'];x=z['positions_before'][t];dt=float(z['dt'])
            D0=np.linalg.norm(x-goals,axis=1).sum()
            def progress(u):return D0-np.linalg.norm(x+dt*u.reshape(2,2)-goals,axis=1).sum()
            checks.append(dict(source=source,rid=rec['rid'],outcome=outcome,t=t,
                risk_before=before['risk'],risk_after=after['risk'],gradient_norm=np.linalg.norm(g),
                fd_relative_difference=np.linalg.norm(gradients[0]-g)/max(np.linalg.norm(g),1e-12),
                applied_speed_before=np.linalg.norm(u0),applied_speed_after=np.linalg.norm(u1),
                one_step_goal_progress_before=progress(u0),one_step_goal_progress_after=progress(u1)))
    coverage=[]
    for source in ['primary','extended_safety','extended_c1']:
        zz=[np.load(ROOT/r['score_path']) for r in index if r['source']==source]
        coverage.append(dict(source=source,states=sum(len(z['valid']) for z in zz),
            valid=sum(int(z['valid'].sum()) for z in zz),
            ambiguous=sum(int(np.sum(z['ambiguous_witnesses']>0)) for z in zz),
            max_feasibility_error=max(float(np.nanmax(z['feasibility_error'])) for z in zz),
            E_min=min(float(np.nanmin(z['E'])) for z in zz),E_max=max(float(np.nanmax(z['E'])) for z in zz)))
    save(OUT/'supplemental_checks.json',dict(high_dimension_rays=dimensions,frozen_state_descent=checks,coverage=coverage,
        note='Finite differences in q only; no autodiff/state-gradient or BPTT validation. One-step progress is not liveness.'))
    print(json.dumps(dict(dimensions=[(r['m'],r['seconds']) for r in dimensions],checks=checks,coverage=coverage),indent=2))

if __name__=='__main__':main()
