"""Horizon-only sensitivity and exhaustive existing-candidate coverage audit."""
import json
import numpy as np
from scipy.special import expit
from audit_c1_progress_conditioned_wait import OUT as SINGLE, BASE,ROOT,END,DT,SEED,save

OUT=ROOT/'results/c1_wait_horizon_sweep'
HORIZONS=[20.,25.,30.,35.,42.5]

def main():
    OUT.mkdir(exist_ok=True)
    source=json.loads((SINGLE/'rows.json').read_text());ids=sorted({r['rid'] for r in source})
    save(OUT/'protocol.json',dict(ids=ids,endpoints_seconds=HORIZONS,decision_seconds=5,
        actual_prediction_lengths_seconds=[h-5 for h in HORIZONS],score_window=[5,15],execution_endpoint=42.5,
        unchanged='controller,g,P,S,U,Cs,threshold.005,width.00125,9 candidates,2s pulse,initials,fixed-noise',
        changed='Only late rate denominator and endpoint: [D(x15)-D(xL)]/(L-15).',
        combination='Ctwo=Cs*Cl+(1-Cs)*Cl**2; two and long-only controls.',
        oracle='Exhaust all original9 candidate outcomes per initial; outcome labels used only for diagnosis, never objective.',
        scope='Previously examined12 enriched initial states; exploratory horizon sensitivity, no claim of optimal horizon or global infeasibility.'))
    rows=[]
    for old in source:
        rid,a=old['rid'],old['action']
        z=np.load(BASE/'traces'/f'{rid:04d}_{SEED}_{a}.npz')
        d=np.load(SINGLE/'scores'/f'{rid:04d}_{a}.npz')
        initial=np.linalg.norm(z['positions_before'][0]-z['goals'],axis=1).sum()+1e-5
        ds=np.linalg.norm(z['positions_after'][END-1]-z['goals'],axis=1).sum()/initial
        for h in HORIZONS:
            last=min(round(h/DT),len(z['positions_after']))-1
            dl=np.linalg.norm(z['positions_after'][last]-z['goals'],axis=1).sum()/initial
            rate=(ds-dl)/(h-15);cl=float(expit((.005-rate)/.00125));two=d['C']*cl+(1-d['C'])*cl**2
            st=d['U']*d['S']*two;sl=d['U']*d['S']*cl
            c={**old['costs'],'Stwo':float(st.mean()),'Slong':float(sl.mean()),
                'Gtwo':float((st+(1-st)*d['g']).mean()),'Glong':float((sl+(1-sl)*d['g']).mean())}
            rows.append(dict(rid=rid,action=a,horizon=h,costs=c,late_rate=rate,Cl=cl,outcome=old['outcome']))
    arms={'PStwo':'Stwo','PGtwo':'Gtwo','PSlong':'Slong','PGlong':'Glong'}
    choices=[];stats=[];coverage=[]
    for rid in ids:
        rr=[r for r in source if r['rid']==rid];good=[r['action'] for r in rr if r['outcome']['success']]
        coverage.append(dict(rid=rid,successful_actions=good,successful_count=len(good),
            candidates=[dict(action=r['action'],**r['outcome']) for r in rr]))
    for h in HORIZONS:
        for arm,field in arms.items():
            selected=[]
            for rid in ids:
                rr=[r for r in rows if r['horizon']==h and r['rid']==rid]
                chosen=min(rr,key=lambda r:(r['costs']['P']+r['costs'][field],r['action']))
                chosen=dict(arm=f'{arm}_L{h:g}',base_arm=arm,**chosen);choices.append(chosen);selected.append(chosen)
            fails=[r for r in selected if not r['outcome']['success']]
            ranking=[r['rid'] for r in fails if next(c for c in coverage if c['rid']==r['rid'])['successful_count']>0]
            missing=[r['rid'] for r in fails if r['rid'] not in ranking]
            stats.append(dict(horizon=h,arm=arm,successes=sum(r['outcome']['success'] for r in selected),
                deadlocks=sum(r['outcome']['deadlock'] for r in selected),
                **{key:float(np.mean([r['outcome'][key] for r in selected])) for key in ['stagnation_seconds','future_stagnation_seconds','remaining','tail_stagnation_fraction']},
                ranking_failure_ids=ranking,no_successful_candidate_ids=missing,
                selected_actions={r['rid']:r['action'] for r in selected}))
    save(OUT/'rows.json',rows);save(OUT/'choices.json',choices);save(OUT/'summary.json',dict(stats=stats,coverage=coverage))
    for r in stats:
        if r['arm'] in ['PStwo','PGtwo']:print(json.dumps(r))

if __name__=='__main__':main()
