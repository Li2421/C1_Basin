"""Source-only non-neural controller-by-eta prior diagnostic.

This is a known-controller source-scene reference, never a zero-shot model.
It uses only TRAIN observed successes/failures; VAL stays held out.
"""
import json
from pathlib import Path
import numpy as np
from .pipeline import OUT,write


def evaluate(variant):
    root=OUT/'variants'/variant
    d=dict(np.load(root/'dataset.npz'))
    rows=json.loads((root/'pairs.json').read_text())
    tr=np.flatnonzero(d['split']=='train')
    va=np.flatnonzero(d['split']=='validation')
    assert not set(d['state_index'][tr]) & set(d['state_index'][va])
    pred=np.zeros((3,len(rows)),float)
    counts=[]
    for c in range(3):
        for eta_index in range(16):
            ii=[i for i in tr if rows[i]['eta_index']==eta_index and d['valid'][c,i]]
            s=float(d['success'][c,ii].sum());f=float(d['failure'][c,ii].sum())
            p=(s+.5)/(s+f+1.)
            pred[c,[i for i in va if rows[i]['eta_index']==eta_index]]=p
            counts.append({'controller':c,'eta_index':eta_index,'train_pairs':len(ii),
                           'observed_success':s,'observed_failure':f,'predicted_p':p})
    nll=[];b15=eligible=0;selected_q=[];oracle_q=[]
    picks=[]
    for c in range(3):
        s=d['success'][c,va];f=d['failure'][c,va];p=pred[c,va]
        nll.append(float(-(s*np.log(p)+f*np.log1p(-p)).sum()/max(1.,(s+f).sum())))
        for state in sorted(set(d['state_index'][va])):
            ii=[i for i in va if d['state_index'][i]==state]
            good=d['standard_success'][c,ii]>=15
            if not good.any():continue
            eligible+=1
            winner=ii[int(np.argmax(pred[c,ii]))]
            hit=bool(d['standard_success'][c,winner]>=15)
            b15+=int(hit)
            q=d['success'][c,ii]/np.maximum(1.,d['success'][c,ii]+d['failure'][c,ii])
            selected_q.append(float(d['success'][c,winner]/max(1.,d['success'][c,winner]+d['failure'][c,winner])))
            oracle_q.append(float(q.max()))
            picks.append({'controller':c,'state_index':int(state),'selected_eta_index':int(rows[winner]['eta_index']),
                          'B15':hit,'selected_observed_Q':selected_q[-1]})
    return {'variant':variant,'model':'TRAIN known-controller × exact-eta Jeffreys-smoothed prior',
            'not_zero_shot':True,'TRAIN_state_count':len(set(d['state_index'][tr])),
            'VAL_state_count':len(set(d['state_index'][va])),
            'VAL_NLL':float(np.mean(nll)),'VAL_NLL_by_controller':nll,
            'oracle_eligible':eligible,'B15_selected':b15,
            'selected_observed_Q':float(np.mean(selected_q)),
            'observed_Q_regret':float(np.mean(np.asarray(oracle_q)-selected_q)),
            'state_reversal_prediction_possible':False,'estimates':counts,'picks':picks}


def main():
    result={v:evaluate(v) for v in ('small_20','large_20')}
    write(OUT/'known_controller_eta_prior.json',result)
    print({k:{x:v[x] for x in ('TRAIN_state_count','VAL_state_count','VAL_NLL','oracle_eligible','B15_selected','observed_Q_regret')} for k,v in result.items()})


if __name__=='__main__':main()
