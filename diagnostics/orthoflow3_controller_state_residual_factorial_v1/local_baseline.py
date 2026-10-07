"""Fixed-k same-controller, same-eta local source-family diagnostic.

All distances and normalizers use source TRAIN data only. Every variant is
reported; independent source VAL is not used to tune k or feature weights.
"""
import json
import numpy as np
from .pipeline import OUT,write
from diagnostics.orthoflow3_controller_training_repair_v1 import train as prior


def run():
    prior.OUT=OUT/'variants'/'large_20'
    d=prior.load()
    rows=d['rows'];si=d['state_index']
    physical=d['x']
    state=np.concatenate([physical[k].reshape(len(physical[k]),-1) for k in
                          ('agents','pairs','obstacles','globals')],axis=1)
    tr_state=np.unique(si[d['split']=='train'])
    center=state[tr_state].mean(0);scale=np.maximum(state[tr_state].std(0),.1)
    state=(state-center)/scale
    train=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation')
    out=[]
    for feature in ('physical','context','physical_context'):
        for k in (1,3,5):
            z=np.zeros((3,len(rows)),np.float32)
            for c in range(3):
                for i in val:
                    candidates=np.asarray([j for j in train if rows[j]['eta_index']==rows[i]['eta_index'] and d['valid'][c,j]])
                    aa=state[si[candidates]] if feature=='physical' else d['context'][c,candidates]
                    bb=state[si[i]] if feature=='physical' else d['context'][c,i]
                    if feature=='physical_context':
                        aa=np.concatenate((state[si[candidates]],d['context'][c,candidates]),axis=1)
                        bb=np.concatenate((state[si[i]],d['context'][c,i]))
                    distance=np.mean((aa-bb)**2,axis=1)
                    chosen=candidates[np.argsort(distance)[:k]]
                    s=float(d['success'][c,chosen].sum());f=float(d['failure'][c,chosen].sum())
                    p=(s+.5)/(s+f+1.)
                    z[c,i]=np.log(p/(1-p))
            m=prior.metrics(z,d,val)
            out.append({'feature':feature,'k':k,'VAL_NLL':m['NLL'],'VAL_MAE':m['MAE'],
                        'oracle_eligible':m['eligible'],'B15_selected':m['selected_B15'],
                        'observed_Q_regret':m['observed_Q_regret'],
                        'state_reversal_correct':m['state_reversals']['correct'],
                        'state_reversal_total':m['state_reversals']['total'],
                        'controller_reversal_correct':m['controller_reversals']['correct'],
                        'controller_reversal_total':m['controller_reversals']['total']})
    write(OUT/'local_baseline_results.json',out)
    print(out)


if __name__=='__main__':run()
