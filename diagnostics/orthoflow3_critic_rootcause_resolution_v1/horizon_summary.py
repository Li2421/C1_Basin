"""Summarize preregistered source-controller holdouts, not final confirmation."""
import json
import numpy as np
from scipy.special import expit
from .response_horizon import OUT,SOURCE,NAMES,SEEDS,read,write,csvwrite

def main():
    d=np.load(SOURCE/'dataset.npz');rows=[];decisions=[];saved={}
    for held in range(4):
      for variant,horizon in [('eta_only',20),('entity_response_mean',20),('entity_response_mean',80)]:
       for seed in SEEDS:
        dest=OUT/'models'/f'held{held}'/f'{variant}_H{horizon}'/f'seed{seed}'
        done=read(dest/'complete.json');pred=np.load(dest/'held_predictions.npz');ii=pred['indices']
        assert np.array_equal(d['eta_index'][ii].reshape(-1,2),np.tile([10,15],(len(ii)//2,1)))
        s=d['success'][held,ii];f=d['failure'][held,ii];q=(s/(s+f)).reshape(-1,2)
        good=d['standard_success'][held,ii].reshape(-1,2)>=15
        bad=d['standard_failure'][held,ii].reshape(-1,2)>=2
        low=d['standard_success'][held,ii].reshape(-1,2)/16
        high=(16-d['standard_failure'][held,ii].reshape(-1,2))/16
        strong=np.where(low[:,0]-high[:,1]>=.25,1,np.where(high[:,0]-low[:,1]<=-.25,-1,0))
        for condition in pred.files:
            if condition=='indices':continue
            z=pred[condition].ravel();p=expit(z).reshape(-1,2);choice=p.argmax(1);rr=np.arange(len(choice))
            pairloss=(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/(s+f)
            contrast=q[:,0]-q[:,1];pc=p[:,0]-p[:,1]
            corr=float(np.corrcoef(contrast,pc)[0,1]) if pc.std()>1e-8 else None
            row=dict(held=NAMES[held],kind=variant,horizon=horizon,seed=seed,condition=condition,
                NLL=float(pairloss.mean()),MAE=float(abs(p-q).mean()),B15=int(good[rr,choice].sum()),
                unknown=int((~good[rr,choice]&~bad[rr,choice]).sum()),cases=len(choice),oracle_B15=int(good.any(1).sum()),
                selected_Q=float(q[rr,choice].mean()),regret=float((q.max(1)-q[rr,choice]).mean()),
                severe=int(((p[rr,choice]>.9)&(high[rr,choice]<=.5)).sum()),contrast_correlation=corr,
                strong_cases=int((strong!=0).sum()),strong_correct=int(((np.sign(pc)==strong)&(strong!=0)).sum()),
                best_step=done['best_step'])
            rows.append(row)
            key=(held,variant,horizon,seed,condition)
            saved[key]=dict(loss=pairloss.reshape(-1,2).mean(1),B15=good[rr,choice].astype(float),Q=q[rr,choice])
            for r in rr:
                decisions.append(dict(held=NAMES[held],kind=variant,horizon=horizon,seed=seed,condition=condition,
                    state_index=int(d['state_index'][ii][2*r]),eta_index=int(d['eta_index'][ii][2*r+choice[r]]),
                    Q=float(q[r,choice[r]]),B15=bool(good[r,choice[r]]),true_contrast=float(contrast[r]),predicted_contrast=float(pc[r])))
    csvwrite(OUT/'source_LCO_metrics.csv',rows);csvwrite(OUT/'source_LCO_decisions.csv',decisions)
    rng=np.random.default_rng(8042026);boot=rng.integers(0,32,(10000,32));comparisons=[]
    for reference in [('eta_only',20),('entity_response_mean',20)]:
      for metric in ('loss','B15','Q'):
        delta=np.array([[[saved[(c,'entity_response_mean',80,s,'logits')][metric][j]-saved[(c,*reference,s,'logits')][metric][j]
            for j in range(32)] for s in SEEDS] for c in range(4)])
        family=delta.mean((0,1));ci=np.quantile(family[boot].mean(1),[.025,.975])
        comparisons.append(dict(H80_reference=f'{reference[0]}_H{reference[1]}',metric=metric,mean_delta=float(family.mean()),
            CI_low=float(ci[0]),CI_high=float(ci[1]),fold_mean_deltas=delta.mean((1,2)).tolist(),
            interpretation='Source-side exploratory representation comparison;family-cluster bootstrap across4controllers and3seeds, not12independentcontroller replications'))
    write(OUT/'source_LCO_comparisons.json',comparisons)
    correct=[r for r in rows if r['condition']=='logits']
    summary=[]
    for held in NAMES:
      for kind,horizon in [('eta_only',20),('entity_response_mean',20),('entity_response_mean',80)]:
        a=[r for r in correct if r['held']==held and r['kind']==kind and r['horizon']==horizon]
        summary.append(dict(held=held,kind=kind,horizon=horizon,NLL=float(np.mean([r['NLL'] for r in a])),
            B15=[r['B15'] for r in a],oracle=a[0]['oracle_B15'],selected_Q=float(np.mean([r['selected_Q'] for r in a]))))
    write(OUT/'source_LCO_summary.json',dict(rows=summary,new_task_rollouts=0,held_natural_controller_count=4,
        outer_families=32,not_final_confirmation=True,comparisons=comparisons))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
