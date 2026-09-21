"""Strict pre-event five-second forecast on the fresh 400-trajectory audit set."""
import json
import numpy as np
from scipy.special import expit
from audit_c1_joint_witness_risk import ROOT, LocalRisk, PROTOCOL, save, auc
from audit_c1_completed_waiting_risk import completion, KAPPA, MAXR
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, GiveWayEnv

BASE=ROOT/'results/c1_large_risk_association_400';DT=.05;K=850
def main():
    rows=json.loads((BASE/'rows.json').read_text());out=[];env=GiveWayEnv(Config(corridor_half_length=1.3));cbf=CBFConfig()
    for r in rows:
        with np.load(BASE/'traces'/f'{r["rid"]:04d}.npz') as z:
            event=np.flatnonzero(z['deadlock']);success=np.flatnonzero(z['success']);first_event=event[0]+1 if len(event) else K+1;first_success=success[0]+1 if len(success) else K+1
            for t in range(0,min(first_event,first_success,K-100),20):
                A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cbf)
                local=LocalRisk(A,b,np.full(2,.5)).score(z['candidate'][t]);g=local['risk']/MAXR
                v=z['candidate'][t];speed2=np.sum((v/.5)**2);geom=float(completion(g,speed2));stop=float(KAPPA**2/(KAPPA**2+speed2))
                dist=np.linalg.norm(z['positions_after']-env.goals,axis=-1);denom=np.linalg.norm(z['positions_before'][0]-env.goals,axis=-1).sum()+1e-5
                end=t+1;start=max(end-80,0);rate=((np.linalg.norm(z['positions_before'][start]-env.goals,axis=-1).sum()/denom)-dist[t].sum()/denom)/((end-start)*DT)
                U=1-np.prod(1-expit((dist[t]-.08)/.02));p=float(U*expit((.005-rate)/.00125))
                out.append(dict(rid=r['rid'],split=r['split'],y=bool(t<first_event<=t+100),smooth=.5*(geom+p),geometry=geom,progress=p,stop=stop,hard=.5*(g+p)))
    report={}
    for split in ['all','development','heldout']:
        rr=[r for r in out if split=='all' or r['split']==split];counts={i:sum(r['rid']==i for r in rr) for i in {r['rid'] for r in rr}};w=[1/counts[r['rid']] for r in rr]
        report[split]=dict(n=len(rr),positives=sum(r['y'] for r in rr),episodes=len({r['rid'] for r in rr if r['y']}),
            auc={f:auc([r['y'] for r in rr],[r[f] for r in rr],w) for f in ['hard','smooth','geometry','progress','stop']})
    save(BASE/'local_forecast.json',dict(horizon_seconds=5.,sampling_seconds=1.,label='event after current state and within next 5s; event/post-event states excluded.',report=report))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
