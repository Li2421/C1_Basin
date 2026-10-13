"""Post-hoc frozen-prediction audit of departures from source-fixed eta; no fits."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent

def value(a,s):
    return np.take_along_axis(a,s[...,None],axis=-1)[...,0]

def main():
    with np.load(HERE/'predictions.npz') as z:
        pred=z['prediction'];fixed=z['fixed_eta']
    with np.load(HERE.parent/'controller_q_transfer_round3_v1/P1_LABELS.npz') as z:
        lo=z['Y'].mean(-1);hi=(z['Y']+z['U']).mean(-1)
    q=(lo+hi)/2
    out={'scope':'Post-hoc descriptive audit on opened Q4 data; unknown robustness is not per-case statistical significance. No model or selection changes.', 'new_rollouts':0,'new_fits':0,'models':{}}
    for ai,arm in enumerate(('R0','R5')):
        a=pred[ai];sel=a.argmax(-1);changed=sel!=fixed
        actual=value(q,sel)-value(q,fixed)
        gl=np.where(changed,value(lo,sel)-value(hi,fixed),0)
        gh=np.where(changed,value(hi,sel)-value(lo,fixed),0)
        predicted=value(a,sel)-value(a,fixed)
        others=np.ones(q.shape,bool);np.put_along_axis(others,fixed[...,None],False,axis=-1)
        out['models'][arm]=dict(changed_cases=int(changed.sum()),total_cases=int(changed.size),
            Q4_unknown_robust_gain_cases=int((changed&(gl>0)).sum()),
            Q4_unknown_robust_loss_cases=int((changed&(gh<0)).sum()),
            Q4_mid_ties=int((changed&(actual==0)).sum()),
            unknown_ambiguous_cases=int((changed&(gl<=0)&(gh>=0)&(gl!=gh)).sum()),
            predicted_gain_on_changes=float(predicted[changed].mean()),
            Q4_mid_gain_on_changes=float(actual[changed].mean()),
            fixed_eta_MSE=float(np.mean((value(a,fixed)-value(q,fixed))**2)),
            other_eta_MSE=float(np.mean((a-q)[others]**2)),
            changed_fixed_prediction_mean=float(value(a,fixed)[changed].mean()),
            changed_fixed_Q4_mid_mean=float(value(q,fixed)[changed].mean()),
            changed_selected_prediction_mean=float(value(a,sel)[changed].mean()),
            changed_selected_Q4_mid_mean=float(value(q,sel)[changed].mean()))
    a=out['models']['R0'];b=out['models']['R5']
    out['fraction_of_total_MSE_reduction_from_other_15_eta']=float(15*(a['other_eta_MSE']-b['other_eta_MSE'])/(a['fixed_eta_MSE']-b['fixed_eta_MSE']+15*(a['other_eta_MSE']-b['other_eta_MSE'])))
    (HERE/'selection_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
