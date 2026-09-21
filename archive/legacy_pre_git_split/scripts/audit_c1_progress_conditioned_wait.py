"""Matched low-speed replacement; fixed controller, geometry and existing lookahead."""
import json
import time
import numpy as np
from scipy.special import expit
from audit_c1_clean_geometry import BASE, ROOT, START, END, K, DT, SEED
from audit_c1_joint_witness_risk import LocalRisk, save
from audit_c1_completed_waiting_risk import series, MAXR, KAPPA
from single_integrator.environment import Config, GiveWayEnv
from single_integrator.cbf import CBFConfig, barrier_constraints

OUT=ROOT/'results/c1_progress_conditioned_wait_audit'
OLD=ROOT/'results/c1_clean_geometry_audit'

def score(rid,action):
    target=OUT/'scores'/f'{rid:04d}_{action}.json'
    if target.exists():return json.loads(target.read_text())
    z=np.load(BASE/'traces'/f'{rid:04d}_{SEED}_{action}.npz')
    old=json.loads((OLD/'scores'/target.name).read_text())
    arrays,_,T,_=series(z,10)
    env=GiveWayEnv(Config(corridor_half_length=1.3));cfg=CBFConfig()
    # Current state at t*dt; fixed prediction endpoint END*dt. No access beyond END.
    initial=np.linalg.norm(z['positions_before'][0]-z['goals'],axis=1).sum()+1e-5
    terminal=z['positions_after'][min(T,END)-1]
    Dend=np.linalg.norm(terminal-z['goals'],axis=1).sum()/initial
    values=[]
    for t in range(START,min(T,END)):
        x=z['positions_before'][t];v=z['candidate'][t]
        dist=np.linalg.norm(x-z['goals'],axis=1)
        U=1-np.prod(1-expit((dist-.08)/.02))
        rate=(dist.sum()/initial-Dend)/((END-t)*DT)
        C=expit((.005-rate)/.00125)
        A,b,_=barrier_constraints(dict(positions=x,walls=env.walls,config=env.config.to_dict()),cfg)
        r=LocalRisk(A,b,np.full(2,.5)).score(v)
        if not r['valid']:raise ValueError('Undefined pure geometry')
        g=r['risk']/MAXR;S=KAPPA**2/(KAPPA**2+np.sum((v/.5)**2));new=U*S*C
        values.append([g,S,U,C,new,S+(1-S)*g,new+(1-new)*g,new+(1-S)*g,
            rate,float(np.max(np.linalg.norm(z['applied'][t].reshape(2,2),axis=1))<.025),
            float(z['candidate_deadlock'][t])])
    names=['g','S','U','C','Snew','G','Gnew','Gfixed','future_rate','low_applied','candidate_deadlock']
    dense=np.pad(np.asarray(values),((0,END-START-len(values)),(0,0)))
    data={key:dense[:,j] for j,key in enumerate(names)}
    data['P']=arrays['progress'][START:END]
    assert np.all(data['Snew']<=data['S']+1e-12)
    assert np.all(data['Gnew']<=data['G']+1e-12)
    costs={key:float(data[key].mean()) for key in ['P','g','S','Snew','G','Gnew','Gfixed']}
    for key in ['P','g','S','G']:assert abs(costs[key]-old['costs'][key])<1e-10
    row=dict(rid=rid,action=action,costs=costs,outcome=old['outcome'])
    np.savez_compressed(target.with_suffix('.npz'),**data)
    save(target,row);return row

def main():
    OUT.mkdir(exist_ok=True);(OUT/'scores').mkdir(exist_ok=True)
    ids=json.loads((OLD/'protocol.json').read_text())['ids']
    arms={'P':None,'PS':'S','PSnew':'Snew','PG':'G','PGnew':'Gnew','PGfixed':'Gfixed'}
    save(OUT/'protocol.json',dict(ids=ids,seed=SEED,decision_seconds=5,intervention_seconds=2,
        lookahead_endpoint_seconds=15,outcome_endpoint_seconds=42.5,dt=DT,arms=arms,
        replacement='Snew=U*S*sigmoid((.005-rate)/.00125)',
        progress='rate=[D(x_t)-D(x_END)]/[(END-t)*dt]; D=sum goal distances divided by initial sum+1e-5',
        unfinished='U=1-product(1-sigmoid((distance_i-.08)/.02)), evaluated at x_t',
        endpoint='For every scored t in5..15, use remaining available suffix to15; no extra rollout or states beyond15. H_t shrinks near endpoint.',
        primary='PGnew versus PG and PSnew versus PS; literal S substitution Gnew=Snew+(1-Snew)*g.',
        attribution='PGfixed=P+Snew+(1-S)*g isolates additive low-speed reduction with original weighted geometry fixed; diagnostic only.',
        parameters='No new tuning parameters; thresholds, widths, unfinished mask and horizon reused.',
        caveat='Same 12 previously examined enriched states and fixed common noise as clean audit; not held-out inference.'))
    rows=[];choices=[];start=time.perf_counter()
    for rid in ids:
        rr=[score(rid,a) for a in range(9)];rows.extend(rr)
        for arm,field in arms.items():
            r=min(rr,key=lambda r:(r['costs']['P']+(r['costs'][field] if field else 0),r['action']))
            choices.append(dict(arm=arm,**r))
        save(OUT/'choices.json',choices)
        print(rid,round(time.perf_counter()-start,1),{r['arm']:r['action'] for r in choices if r['rid']==rid},flush=True)
    stats={}
    for arm in arms:
        rr=[r['outcome'] for r in choices if r['arm']==arm]
        stats[arm]=dict(n=len(rr),successes=sum(r['success'] for r in rr),deadlocks=sum(r['deadlock'] for r in rr),
            **{k:float(np.mean([r[k] for r in rr])) for k in ['stagnation_seconds','future_stagnation_seconds','remaining','tail_stagnation_fraction']})
    pairs={}
    for a,b in [('PSnew','PS'),('PGnew','PG'),('PGnew','P'),('PGfixed','PG')]:
        aa=[r for r in choices if r['arm']==a];bb=[r for r in choices if r['arm']==b]
        pairs[a+'_vs_'+b]=[dict(rid=x['rid'],old_action=y['action'],new_action=x['action'],old=y['outcome'],new=x['outcome'],old_costs=y['costs'],new_costs=x['costs']) for x,y in zip(aa,bb) if x['action']!=y['action']]
    groups={}
    for label in ['success','deadlock','timeout']:
        rr=[r for r in rows if ('success' if r['outcome']['success'] else 'deadlock' if r['outcome']['deadlock'] else 'timeout')==label]
        data=[np.load(OUT/'scores'/f"{r['rid']:04d}_{r['action']}.npz") for r in rr]
        low=np.concatenate([d['low_applied'] for d in data]).astype(bool)
        S=np.concatenate([d['S'] for d in data]);Sn=np.concatenate([d['Snew'] for d in data])
        groups[label]=dict(branches=len(rr),low_frames=int(low.sum()),
            low_S_mean=float(S[low].mean()) if low.any() else None,low_Snew_mean=float(Sn[low].mean()) if low.any() else None,
            mean_S=float(S.mean()),mean_Snew=float(Sn.mean()))
    save(OUT/'rows.json',rows);save(OUT/'summary.json',dict(arms=stats,paired=pairs,descriptive_groups=groups,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
