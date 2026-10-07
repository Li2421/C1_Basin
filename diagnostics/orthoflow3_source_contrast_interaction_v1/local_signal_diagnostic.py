"""Post-hoc, non-neural source-family crossfit for controller response signal.

This is diagnostic only. It is not a newly pre-registered independent model
confirmation, and does not touch the separate 16-family validation labels.
"""
from __future__ import annotations
import csv,hashlib,json
import numpy as np
from .pipeline import OUT,CONTROLLERS,read,write

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    clear=abs(y)>=.25
    return {'n':len(y),'clear':int(clear.sum()),'MAE':float(np.mean(abs(y-p))),
            'Pearson':float(np.corrcoef(y,p)[0,1]) if np.std(y)>1e-8 and np.std(p)>1e-8 else None,
            'clear_sign_accuracy':float(np.mean(np.sign(y[clear])==np.sign(p[clear]))) if clear.any() else None}

def main():
    d=np.load(OUT/'dataset.npz');states=read(OUT/'states.json')
    train=sorted(int(v) for v in set(d['state_index'][d['split']=='train']))
    assert len(train)==46
    order=sorted(train,key=lambda i:hashlib.sha256(('source_context_signal_crossfit_v1\0'+states[i]['source_group']).encode()).hexdigest())
    rows=[]
    for fold in range(3):
        held=order[fold::3];fit=sorted(set(train)-set(held))
        for ci,controller in enumerate(CONTROLLERS):
            def values(indices):
                a=np.asarray([2*i for i in indices]);b=a+1
                ca=d['context'][ci,a];cb=d['context'][ci,b]
                # 10 nominal variables + 14 eta-response differences.
                feature=np.concatenate((ca[:,:10],ca[:,10:]-cb[:,10:]),axis=1)
                qa=d['success'][ci,a]/(d['success'][ci,a]+d['failure'][ci,a])
                qb=d['success'][ci,b]/(d['success'][ci,b]+d['failure'][ci,b])
                return feature,qa-qb
            X,y=values(fit);T,truth=values(held)
            center=X.mean(0);scale=np.maximum(X.std(0),.05)
            X=(X-center)/scale;T=(T-center)/scale
            for model in ('controller_constant','ridge_1','ridge_10','ridge_100','knn_3','knn_5'):
                if model=='controller_constant':pred=np.full(len(held),y.mean())
                elif model.startswith('ridge'):
                    alpha=float(model.split('_')[1]);yy=y-y.mean()
                    coef=np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@yy)
                    pred=y.mean()+T@coef
                else:
                    k=int(model.split('_')[1]);dist=np.sum((T[:,None,:]-X[None,:,:])**2,axis=-1)
                    ix=np.argsort(dist,axis=1)[:,:k]
                    pred=y[ix].mean(1)
                rows.extend({'fold':fold,'controller':controller,'state_index':state,
                    'source_group':states[state]['source_group'],'model':model,
                    'true_Q_delta':float(v),'predicted_Q_delta':float(p),
                    'strong_Q_contrast':bool(abs(v)>=.25)}
                    for state,v,p in zip(held,truth,pred))
    write_csv(OUT/'local_signal_source_crossfit.csv',rows)
    summary=[]
    for model in sorted(set(r['model'] for r in rows)):
        for controller in (*CONTROLLERS,'pooled'):
            subset=[r for r in rows if r['model']==model and (controller=='pooled' or r['controller']==controller)]
            summary.append({'model':model,'controller':controller,
                **metrics([r['true_Q_delta'] for r in subset],[r['predicted_Q_delta'] for r in subset])})
    write_csv(OUT/'local_signal_source_crossfit_summary.csv',summary)
    write(OUT/'local_signal_protocol.json',{'posthoc_diagnostic':True,'no_validation_label_for_fit_or_selection':True,
        'only_source_TRAIN_families':True,'3fold_family_crossfit':True,
        'predefined_models':['controller_constant','ridge_1','ridge_10','ridge_100','knn_3','knn_5'],
        'input':'10 nominal H20 physical summaries plus 14 eta-response differences',
        'no_model_chosen_on_frozen_validation':True,'new_rollout':0})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
