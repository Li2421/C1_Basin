"""Two-horizon waiting-only ablation with unchanged geometry and closed-loop branches."""
import json
import numpy as np
from scipy.special import expit
from audit_c1_progress_conditioned_wait import OUT as SINGLE,BASE,ROOT,START,END,K,DT,SEED,save

OUT=ROOT/'results/c1_two_horizon_wait_audit'

def main():
    OUT.mkdir(exist_ok=True);(OUT/'scores').mkdir(exist_ok=True)
    oldrows=json.loads((SINGLE/'rows.json').read_text())
    ids=sorted({r['rid'] for r in oldrows})
    arms={'P':None,'PS':'S','PSsingle':'Snew','PStwo':'Stwo','PSlong':'Slong',
          'PG':'G','PGsingle':'Gnew','PGtwo':'Gtwo','PGlong':'Glong','PGtwofixed':'Gtwofixed'}
    save(OUT/'protocol.json',dict(ids=ids,seed=SEED,decision_seconds=5,intervention_seconds=2,
        score_window=[5,15],short_endpoint=15,long_endpoint=42.5,outcome_endpoint=42.5,arms=arms,
        short='Cs=sigmoid((.005-[D(x_t)-D(x_15)]/(15-t))/.00125), unchanged',
        long='Cl=sigmoid((.005-[D(x_15)-D(x_42.5)]/27.5)/.00125)',
        combination='Ctwo=Cs*Cl+(1-Cs)*Cl**2; Stwo=U*S*Ctwo',
        interpretation='No immediate progress: defer to late progress. Immediate progress: extra discount only when late progress also supports it. Smooth heuristic, not a probability.',
        controls='Slong=U*S*Cl isolates late-information value; Gtwofixed retains old weighted geometry.',
        scope='Only waiting score changes. Same g,P,U,S,thresholds,widths,controller,candidates,initials,noise. Long horizon uses existing42.5 endpoint; no tuned weight.',
        information='Waiting score now sees predicted15..42.5 states. Execution outcome has no separate unseen time tail. Fixed-noise ideal prediction test, not evidence of robust forecasting.',
        completion='Existing success absorption: hold final goal distances after predicted success; no outcome label in penalty. All these branches last beyond15.',
        rationale='Use disjoint late increment to avoid counting early progress again. Binary desired behavior is determined by long evidence; short evidence only modulates soft confidence.'))
    rows=[]
    for old in oldrows:
        rid,a=old['rid'],old['action'];name=f'{rid:04d}_{a}'
        z=np.load(BASE/'traces'/f'{rid:04d}_{SEED}_{a}.npz');d0=np.load(SINGLE/'scores'/f'{name}.npz')
        d={k:d0[k] for k in d0.files}
        assert len(z['positions_after'])>=END
        initial=np.linalg.norm(z['positions_before'][0]-z['goals'],axis=1).sum()+1e-5
        ds=np.linalg.norm(z['positions_after'][END-1]-z['goals'],axis=1).sum()/initial
        dl=np.linalg.norm(z['positions_after'][min(K,len(z['success']))-1]-z['goals'],axis=1).sum()/initial
        late_rate=(ds-dl)/((K-END)*DT);cl=float(expit((.005-late_rate)/.00125))
        cs=d['C'];two=cs*cl+(1-cs)*cl**2
        d.update(Cl=np.full(len(cs),cl),Ctwo=two,Stwo=d['U']*d['S']*two,Slong=d['U']*d['S']*cl)
        d['Gtwo']=d['Stwo']+(1-d['Stwo'])*d['g']
        d['Glong']=d['Slong']+(1-d['Slong'])*d['g']
        d['Gtwofixed']=d['Stwo']+(1-d['S'])*d['g']
        assert np.all((two>=0)&(two<=1))
        costs={**old['costs'],**{k:float(d[k].mean()) for k in ['Stwo','Slong','Gtwo','Glong','Gtwofixed']}}
        row=dict(rid=rid,action=a,costs=costs,outcome=old['outcome'],late_rate=late_rate,Cl=cl)
        rows.append(row);save(OUT/'scores'/f'{name}.json',row);np.savez_compressed(OUT/'scores'/f'{name}.npz',**d)
    choices=[]
    for rid in ids:
        rr=[r for r in rows if r['rid']==rid]
        for arm,field in arms.items():
            chosen=min(rr,key=lambda r:(r['costs']['P']+(r['costs'][field] if field else 0),r['action']))
            choices.append(dict(arm=arm,**chosen))
    stats={}
    for arm in arms:
        rr=[r['outcome'] for r in choices if r['arm']==arm]
        stats[arm]=dict(n=len(rr),successes=sum(r['success'] for r in rr),deadlocks=sum(r['deadlock'] for r in rr),
            **{k:float(np.mean([r[k] for r in rr])) for k in ['stagnation_seconds','future_stagnation_seconds','remaining','tail_stagnation_fraction']})
    pairs={}
    for a,b in [('PStwo','PSsingle'),('PGtwo','PGsingle'),('PGtwo','PG'),('PGtwo','P'),('PStwo','PSlong'),('PGtwo','PGlong'),('PGtwofixed','PGtwo')]:
        aa=[r for r in choices if r['arm']==a];bb=[r for r in choices if r['arm']==b]
        pairs[a+'_vs_'+b]=[dict(rid=x['rid'],old_action=y['action'],new_action=x['action'],old=y['outcome'],new=x['outcome']) for x,y in zip(aa,bb) if x['action']!=y['action']]
    save(OUT/'rows.json',rows);save(OUT/'choices.json',choices);save(OUT/'summary.json',dict(arms=stats,paired=pairs))
    print(json.dumps(dict(arms=stats,paired=pairs),indent=2))

if __name__=='__main__':main()
