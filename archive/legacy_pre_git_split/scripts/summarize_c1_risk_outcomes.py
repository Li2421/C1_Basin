"""Clustered associations and pre-event forecasting; no fitted risk thresholds."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def auc(y,s,w=None):
    y=np.asarray(y,bool);s=np.asarray(s,float)
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    pos=float(w[y].sum());neg=float(w[~y].sum())
    if pos<=0 or neg<=0:return None
    order=np.argsort(s,kind='stable');ys=y[order];ss=s[order];ww=w[order]
    starts=np.r_[0,np.flatnonzero(np.diff(ss))+1]
    pw=np.add.reduceat(ww*ys,starts);nw=np.add.reduceat(ww*~ys,starts)
    return float(np.sum(pw*(np.cumsum(nw)-nw+.5*nw))/(pos*neg))


def stats(rows,key,label,draws,weights=None):
    valid=np.array([r['scores'].get(key) is not None and np.isfinite(r['scores'][key]) for r in rows],bool)
    rr=[r for r,k in zip(rows,valid) if k]
    if not rr:return dict(n=0,positive=0,auc=None,ci95=None),np.full(len(draws),np.nan)
    y=np.array([r[label] for r in rr],bool);s=np.array([r['scores'][key] for r in rr])
    ids=np.array([r['rid']-400000 for r in rr],int)
    w=np.ones(len(rr)) if weights is None else np.asarray(weights)[valid]
    bs=np.array([v if v is not None else np.nan for v in (auc(y,s,w*d[ids]) for d in draws)])
    finite=bs[np.isfinite(bs)]
    result=dict(n=len(rr),positive=int(y.sum()),negative=int((~y).sum()),undefined=len(rows)-len(rr),
        auc=auc(y,s,w),ci95=np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
        valid_bootstraps=len(finite))
    return result,bs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    protocol=json.loads((args.out/'protocol.json').read_text())
    rows=json.loads((args.out/'rows.json').read_text())
    assert len(rows)==384 and (args.out/'complete.json').exists()
    assert len({(r['model'],r['rid'],r['execution_seed']) for r in rows})==384
    assert auc([0,1],[0,1])==1 and auc([0,1],[1,0])==0 and auc([0,1],[1,1])==.5
    assert auc([0,1,1],[0,1,2],[2,1,1])==1
    rng=np.random.default_rng(2026091607)
    draws=rng.multinomial(64,np.ones(64)/64,size=1000)
    summary=dict(scope='Rescoring already-seen Safety and legacy Pg_seed0 policies; NOT a new V3 policy efficacy test',
        bootstrap='1000 paired resamples of64 initial-state clusters;3 noise replicas kept together; one learned-model seed',
        risk_range='P+g is a surrogate score, not calibrated probability',models={},paired={})
    keys=list(rows[0]['scores'])
    for model in protocol['models']:
        rr=[r for r in rows if r['model']==model]
        model_stats=dict(n=len(rr),outcomes={k:sum(r[k] for r in rr) for k in ['success','deadlock','any_deadlock','timeout']},
            undefined_episodes=sum(not r['valid'] for r in rr),scores={},outcome_distributions={},risk_bands={},horizon_differences={})
        boots={}
        for key in keys:
            record={}
            for label in ['deadlock','failure']:
                record[label],boots[(key,label)]=stats(rr,key,label,draws)
            failures=[r for r in rr if r['failure']]
            record['deadlock_vs_other_timeout'],_=stats(failures,key,'deadlock',draws)
            cutoff=float(key.split('L')[-1]) if key.startswith('legacy') else float(key.split('_')[-1])
            # A score at cutoff is a true forecast only if no event/success has
            # yet happened and there is still observation time after cutoff.
            active=[r for r in rr if cutoff<42.5 and (r['first_deadlock'] is None or r['first_deadlock']>cutoff)
                    and (r['first_success'] is None or r['first_success']>cutoff)]
            record['pre_event_future_deadlock'],_=stats(active,key,'deadlock',draws)
            model_stats['scores'][key]=record
        for key in ['Pg_15','Pg_25','Pg_42.5','P_42.5','g_42.5']:
            result={}
            for label in ['success','deadlock','timeout']:
                values=[r['scores'][key] for r in rr if r[label] and r['scores'][key] is not None]
                result[label]=dict(n=len(values),mean=float(np.mean(values)) if values else None,
                    median=float(np.median(values)) if values else None,
                    min=float(np.min(values)) if values else None,max=float(np.max(values)) if values else None)
            model_stats['outcome_distributions'][key]=result
        for key in ['Pg_25','Pg_42.5']:
            bands=[]
            for low,high in zip([0,.25,.5,.75,1.],[.25,.5,.75,1.,2.000001]):
                selected=[r for r in rr if r['scores'][key] is not None and low<=r['scores'][key]<high]
                bands.append(dict(low=low,high=high,n=len(selected),
                    success=sum(r['success'] for r in selected),deadlock=sum(r['deadlock'] for r in selected),
                    timeout=sum(r['timeout'] for r in selected),
                    empirical_deadlock_fraction=sum(r['deadlock'] for r in selected)/len(selected) if selected else None))
            model_stats['risk_bands'][key]=bands
        for a,b in [('Pg_15','Pg_25'),('Pg_25','Pg_42.5'),('legacy_full_L20','legacy_full_L25'),
                    ('legacy_full_L25','legacy_full_L42.5'),('P_25','Pg_25'),('P_42.5','Pg_42.5')]:
            result={}
            for label in ['deadlock','failure']:
                diff=boots[(b,label)]-boots[(a,label)];diff=diff[np.isfinite(diff)]
                aa=model_stats['scores'][a][label]['auc'];bb=model_stats['scores'][b][label]['auc']
                result[label]=dict(auc_difference=bb-aa if aa is not None and bb is not None else None,
                    ci95=np.quantile(diff,[.025,.975]).tolist() if len(diff) else None)
            model_stats['horizon_differences'][b+' minus '+a]=result
        # Strict short-term forecast from observed states, not future rollout
        # scores. Events at/currently before the score time are excluded.
        frames=[];weights=[]
        for r in rr:
            stem=model+f'_{r["rid"]}_{r["execution_seed"]}'
            with np.load(args.out/'scores'/(stem+'.npz')) as z:P=z['P'];G=z['g']
            path=ROOT/f'results/c1_four_objectives_multiseed/evaluation/{model}/{r["rid"]}_{r["execution_seed"]}.npz'
            with np.load(path) as z:u=z['applied'];n=len(u)
            local=[]
            for t in range(100,min(n,750),20):
                now=(t+1)*.05
                if r['first_deadlock'] is not None and r['first_deadlock']<=now:continue
                if r['first_success'] is not None and r['first_success']<=now:continue
                g=float(G[t-100]);pg=float(P[t])+g
                local.append(dict(rid=r['rid'],deadlock=(r['first_deadlock'] is not None and now<r['first_deadlock']<=now+5),
                    scores={'P':float(P[t]),'g':g if np.isfinite(g) else None,
                            'Pg':pg if np.isfinite(pg) else None,
                            'low_speed':-float(np.max(np.linalg.norm(u[t].reshape(2,2),axis=-1)))}))
            frames.extend(local);weights.extend([1/len(local)]*len(local))
        local_stats={key:stats(frames,key,'deadlock',draws,weights)[0] for key in ['P','g','Pg','low_speed']}
        local_stats['definition']='Only current-state scores; event in next5s; pre-event/pre-success only; <=37.5s;1s samples; equal episode total weight'
        local_stats['warning']='Low speed and progress enter the environment detector; strong AUC can reflect its existing hold timer, not deep liveness prediction.'
        model_stats['local_5s_forecast']=local_stats
        summary['models'][model]=model_stats
    lookup={(r['model'],r['rid'],r['execution_seed']):r for r in rows}
    for key in ['Pg_15','Pg_25','Pg_42.5']:
        pairs=[]
        for r in rows:
            if r['model']!='baseline':continue
            b=lookup[('Pg_seed0',r['rid'],r['execution_seed'])]
            if r['scores'][key] is None or b['scores'][key] is None:continue
            delta=b['scores'][key]-r['scores'][key]
            pairs.append(dict(rid=r['rid'],execution_seed=r['execution_seed'],delta_risk=delta,
                before='success' if r['success'] else ('deadlock' if r['deadlock'] else 'timeout'),
                after='success' if b['success'] else ('deadlock' if b['deadlock'] else 'timeout')))
        counts={}
        for pair in pairs:
            kind=pair['before']+' -> '+pair['after']
            counts.setdefault(kind,dict(n=0,risk_down=0,risk_up=0,tie=0))
            counts[kind]['n']+=1;delta=pair['delta_risk']
            counts[kind]['risk_down' if delta < -1e-10 else ('risk_up' if delta>1e-10 else 'tie')]+=1
        summary['paired'][key]=dict(transitions=counts,pairs=pairs)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (args.out/'episode_scores.csv').open('w',newline='') as f:
        fields=['model','rid','execution_seed','success','deadlock','timeout','first_deadlock']+keys
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow({**{k:r[k] for k in fields if k in r},**r['scores']})
    for model,result in summary['models'].items():
        print(model,result['outcomes'])
        for key in ['Pg_15','Pg_20','Pg_25','Pg_42.5','P_42.5','g_42.5','legacy_full_L25']:
            print(key,{k:v for k,v in result['scores'][key].items()})
        print('local',result['local_5s_forecast'])


if __name__=='__main__':main()
