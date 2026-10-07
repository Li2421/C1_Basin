"""Matched-state natural controller response diagnostic; never a selector.

Same physical state and eta across old/held controllers eliminates state
distance as an explanation. Finite distance is NOT exact representation alias.
"""
import csv
import numpy as np
from scipy.special import expit
from .fresh_controller_test import OUT,SOURCE,read,write
from .support_confirmation import OUT as KNOWN
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite

def main():
    d=np.load(OUT/'dataset.npz');old=np.load(KNOWN/'dataset.npz');pred=np.load(OUT/'predictions.npz')
    assert np.array_equal(d['eta'],old['eta']) and np.array_equal(d['state_index'],old['state_index'])
    q=d['success']/(d['success']+d['failure']);oq=old['success']/(old['success']+old['failure'])
    norm=read(SOURCE/'models/expanded206_pair_equal/entity_response_mean/seed17/normalization.json')
    def vector(c,a):
        return np.concatenate([(c-norm['context_center'])/norm['context_scale'],
            ((a-norm['agent_center'])/norm['agent_scale']).mean(-2)],-1)
    c=vector(d['context'],d['agent_response']);oc=vector(old['context'],old['agent_response'])
    distances=np.sqrt(np.mean((oc-c[None])**2,-1));nearest=distances.argmin(0)
    cells=[]
    for i in range(128):
        cells.append(dict(state_index=int(d['state_index'][i]),eta_index=int(d['eta_index'][i]),
            correct_Q=float(q[i]),alt_Q=float(oq[0,i]),second_Q=float(oq[1,i]),
            distance_to_alt=float(distances[0,i]),distance_to_second=float(distances[1,i]),
            nearest_source_controller=('alt','second')[nearest[i]],nearest_source_Q=float(oq[nearest[i],i]),
            abs_Q_difference_to_nearest=float(abs(q[i]-oq[nearest[i],i]))))
    csvwrite(OUT/'matched_state_response_distance.csv',cells)
    bad=[];summary=[]
    for seed in (17,23,41):
        z=pred[f'expanded206_pair_equal__entity_response_mean__{seed}__correct'].reshape(64,2);p=expit(z);ch=z.argmax(1)
        for state in range(64):
            i=2*state+int(ch[state]);prob=float(p[state,ch[state]])
            if prob>.9 and (16-d['failure'][i])/16<=.5:
                bad.append(dict(seed=seed,probability=prob,success=int(d['success'][i]),trials=int(d['success'][i]+d['failure'][i]),**cells[i]))
        summary.append(dict(seed=seed,severe_count=sum(r['seed']==seed for r in bad)))
    csvwrite(OUT/'severe_prediction_matched_controllers.csv',bad)
    write(OUT/'response_distance_adjudication.json',dict(
        same_state_same_eta=True,statistic='RMS of frozen source-normalized24 global plus16 mean agent response coordinates',
        nearest_alt_fraction=float((nearest==0).mean()),
        mean_Q_difference_to_same_state_nearest_source=float(np.mean([v['abs_Q_difference_to_nearest'] for v in cells])),
        fractions_difference_ge_half=float(np.mean([v['abs_Q_difference_to_nearest']>=.5 for v in cells])),
        severe=summary,posthoc_diagnostic_only=True,used_for_model_selection=False,
        caution='Nearest short response is not a proven sufficient statistic. Nonzero distances cannot establish exact aliasing. This is not a deployed nearest-neighbor baseline and uses old confirmation outcomes only as paired counterfactual diagnostics.'))

if __name__=='__main__':main()
