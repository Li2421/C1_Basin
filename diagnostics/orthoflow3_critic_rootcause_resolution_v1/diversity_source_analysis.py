"""Source-VAL interaction diagnostics, never substituted for confirmation."""
import itertools
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
from .controller_diversity import OUT,read,write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite

def main():
    d=np.load(OUT/'dataset.npz');ii=np.flatnonzero(d['split']=='validation')
    s=d['success'][:,ii].reshape(4,32,2);f=d['failure'][:,ii].reshape(4,32,2);q=s/(s+f);good=s>=15;bad=f>=2
    delta=q[...,0]-q[...,1];rows=[]
    for entry in read(OUT/'models_frozen.json')['models']:
        z=np.load(entry['path']+'/validation_predictions.npz')['logits'].reshape(4,32,2);p=expit(z);dp=p[...,0]-p[...,1]
        ch=z.argmax(-1);chosen=lambda a:np.take_along_axis(a,ch[...,None],-1)[...,0]
        tr=delta-delta.mean(1,keepdims=True);pr=dp-dp.mean(1,keepdims=True)
        state_reversals=[];controller_reversals=[]
        for c in range(4):
          for a,b in itertools.combinations(range(32),2):
            if abs(delta[c,a])>=.25 and abs(delta[c,b])>=.25 and delta[c,a]*delta[c,b]<0:
                state_reversals.append(bool(dp[c,a]*delta[c,a]>0 and dp[c,b]*delta[c,b]>0))
        for state in range(32):
          for a,b in itertools.combinations(range(4),2):
            if abs(delta[a,state])>=.25 and abs(delta[b,state])>=.25 and delta[a,state]*delta[b,state]<0:
                controller_reversals.append(bool(dp[a,state]*delta[a,state]>0 and dp[b,state]*delta[b,state]>0))
        pc=p-p.mean(0,keepdims=True);tc=q-q.mean(0,keepdims=True)
        rows.append(dict(arm=entry['size'],kind=entry['kind'],seed=entry['seed'],
            cases=128,B15=int(chosen(good).sum()),unknown=int((~chosen(good)&~chosen(bad)).sum()),oracle_B15=int(good.any(-1).sum()),
            selected_Q=float(chosen(q).mean()),regret=float((q.max(-1)-chosen(q)).mean()),
            source_controller_effect_correlation=float(pearsonr(pc.ravel(),tc.ravel()).statistic) if pc.std()>1e-8 else None,
            centered_state_eta_contrast_correlation=float(pearsonr(tr.ravel(),pr.ravel()).statistic) if pr.std()>1e-8 else None,
            state_reversal_pairs=len(state_reversals),state_reversal_both_correct=float(np.mean(state_reversals)) if state_reversals else None,
            controller_reversal_pairs=len(controller_reversals),controller_reversal_both_correct=float(np.mean(controller_reversals)) if controller_reversals else None))
    csvwrite(OUT/'source_VAL_interactions.csv',rows)
    write(OUT/'source_VAL_interactions_audit.json',dict(
        model_selection_data_not_independent_confirmation=True,
        reversal_rule='Opposite empirical Q contrast signs with each absolute contrast>=0.25; defined independently of predictions',
        dependency_warning='State/controller reversal pairs share families and are not independent trials; descriptive accuracy only',
        controller_effect_identification='Same h and eta are evaluated under4 contexts. No controller ID is a network input. Across-controller prediction changes therefore directly show context use.',
        target_labels_read=False))

if __name__=='__main__':main()
