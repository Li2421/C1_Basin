"""Fixed-noise, finite-lookahead geometry attribution on existing closed-loop branches."""
import time
import json
import hashlib
from pathlib import Path
import numpy as np
from audit_c1_joint_witness_risk import ROOT, LocalRisk, save
from audit_c1_completed_waiting_risk import series, MAXR, KAPPA
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFConfig, barrier_constraints

BASE=ROOT/'results/c1_multistep_objective_audit'
OUT=ROOT/'results/c1_clean_geometry_audit'
START=100; END=300; K=850; DT=.05; SEED=20260916

def branch(rid,action):
    dest=OUT/'scores'/f'{rid:04d}_{action}.json'
    if dest.exists(): return json.loads(dest.read_text())
    z=np.load(BASE/'traces'/f'{rid:04d}_{SEED}_{action}.npz')
    arr,_,T,success=series(z,10)
    env=GiveWayEnv(Config(corridor_half_length=1.3)); cbf=CBFConfig()
    dense=[]
    for t in range(START,min(END,T)):
        A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cbf)
        v=z['candidate'][t]; r=LocalRisk(A,b,np.full(2,.5)).score(v)
        if not r['valid']: raise ValueError('Undefined pure geometry: must not impute zero-control direction')
        g=r['risk']/MAXR; S=KAPPA**2/(KAPPA**2+np.sum((v/.5)**2))
        dense.append([g,S,S+(1-S)*g])
    dense=np.pad(np.asarray(dense),((0,END-START-len(dense)),(0,0)))
    costs=dict(P=float(arr['progress'][START:END].mean()),g=float(dense[:,0].mean()),S=float(dense[:,1].mean()),G=float(dense[:,2].mean()))
    # Outcome labels come only from environment execution, including time after lookahead.
    outcome=dict(success=success,deadlock=bool(z['deadlock'].any()),
        stagnation_seconds=float(z['candidate_deadlock'].sum()*DT),
        future_stagnation_seconds=float(z['candidate_deadlock'][END:].sum()*DT),
        remaining=float(np.linalg.norm(z['positions_after'][-1]-z['goals'],axis=1).max()),
        seconds=T*DT,tail_stagnation_fraction=float(np.r_[z['candidate_deadlock'],np.zeros(K-T)][-100:].mean()))
    row=dict(rid=rid,action=action,costs=costs,outcome=outcome)
    save(dest,row);return row

def main():
    OUT.mkdir(exist_ok=True);(OUT/'scores').mkdir(exist_ok=True)
    ids=json.loads((BASE/'protocol.json').read_text())['ids']
    arms={'P':('g',0.),'Pg025':('g',.25),'Pg1':('g',1.),'Pg4':('g',4.),'PS':('S',1.),'PG':('G',1.)}
    protocol=dict(ids=ids,source=str(BASE),seed=SEED,decision_seconds=5,intervention_seconds=2,
        lookahead_seconds=10,outcome_horizon_seconds=42.5,dt=DT,arms=arms,
        primary='Pg1 versus P; weights .25 and 4 are sensitivity checks, not selected by outcomes',
        randomness='Existing stochastic policy under a fixed common random tape; exact prediction/execution match, NOT a noiseless mean policy.',
        design='Nine existing closed-loop branches per state. Score only t=5..15; observe full outcome to42.5. No terminal label in selection.',
        cohort='Previously examined 12 enriched states; mechanism audit, not held-out population inference.',
        geometry='Original g=(B+.25E)/MAXR; no low-speed completion. Reject undefined direction instead of silently completing.',
        progress='Unchanged full-history progress window, scored over5..15.',
        projection='Original controller and joint safety projection unchanged.')
    save(OUT/'protocol.json',protocol)
    rows=[];choices=[];start=time.perf_counter()
    for rid in ids:
        branches=[branch(rid,a) for a in range(9)]; rows.extend(branches)
        for arm,(field,weight) in arms.items():
            selected=min(branches,key=lambda r:(r['costs']['P']+weight*r['costs'][field],r['action']))
            choices.append(dict(arm=arm,**selected))
        print(rid,round(time.perf_counter()-start,1),{r['arm']:r['action'] for r in choices if r['rid']==rid},flush=True)
        save(OUT/'choices.json',choices)
    summary={}
    for arm in arms:
        rr=[r['outcome'] for r in choices if r['arm']==arm]
        summary[arm]={k:float(np.mean([r[k] for r in rr])) for k in rr[0]}
        summary[arm].update(n=len(rr),successes=sum(r['success'] for r in rr),deadlocks=sum(r['deadlock'] for r in rr))
    pairs={}
    for arm in list(arms)[1:]:
        aa=[r for r in choices if r['arm']==arm];bb=[r for r in choices if r['arm']=='P']
        gains=[];losses=[];changed=[]
        for a,b in zip(aa,bb):
            assert a['rid']==b['rid']
            if a['action']!=b['action']: changed.append(dict(rid=a['rid'],baseline_action=b['action'],action=a['action'],baseline=b['outcome'],new=a['outcome']))
            if a['outcome']['success']>b['outcome']['success']:gains.append(a['rid'])
            if a['outcome']['success']<b['outcome']['success']:losses.append(a['rid'])
        pairs[arm]=dict(success_gains=gains,success_losses=losses,changed=changed)
    ceiling=[]
    for rid in ids:
        rr=[r for r in rows if r['rid']==rid]
        ceiling.append(dict(rid=rid,successful_candidates=sum(r['outcome']['success'] for r in rr)))
    save(OUT/'summary.json',dict(arms=summary,paired=pairs,candidate_coverage=ceiling))
    save(OUT/'rows.json',rows)
    save(OUT/'manifest.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
