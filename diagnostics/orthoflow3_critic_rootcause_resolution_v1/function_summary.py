"""Source-validation input-use gate; no held target labels."""
import numpy as np
from scipy.special import expit
from .function_models import OUT,SEEDS,KINDS,ARMS,read,write,csvwrite,freeze

def main():
    d=np.load(OUT/'dataset.npz');rows=[];decisions=[]
    for arm in ARMS:
      for kind in KINDS:
       for seed in SEEDS:
        path=OUT/'models'/arm/kind/f'seed{seed}';done=read(path/'complete.json');pred=np.load(path/'validation_predictions.npz');ii=pred['indices']
        for condition in pred.files:
            if condition=='indices':continue
            z=pred[condition];p=expit(z).reshape(12,16,2);s=d['success'][:,ii].reshape(12,16,2);f=d['failure'][:,ii].reshape(12,16,2);q=s/(s+f)
            good=s>=15;bad=f>=2;ch=p.argmax(-1);cs=np.arange(12)[:,None];ss=np.arange(16)[None,:]
            losses=(s*np.logaddexp(0,-z.reshape(12,16,2))+f*np.logaddexp(0,z.reshape(12,16,2)))/(s+f)
            for group,ix in [('original4',np.arange(4)),('new8',np.arange(4,12)),('all12',np.arange(12))]:
                delta=(q[:,:,0]-q[:,:,1])[ix];dp=(p[:,:,0]-p[:,:,1])[ix]
                dc=delta-delta.mean(1,keepdims=True);pc=dp-dp.mean(1,keepdims=True)
                rows.append(dict(arm=arm,kind=kind,seed=seed,condition=condition,controllers=group,
                    NLL=float(losses[ix].mean()),MAE=float(abs(p-q)[ix].mean()),B15=int(good[cs,ss,ch][ix].sum()),
                    unknown=int((~good[cs,ss,ch]&~bad[cs,ss,ch])[ix].sum()),oracle_B15=int(good.any(-1)[ix].sum()),
                    selected_Q=float(q[cs,ss,ch][ix].mean()),cases=len(ix)*16,
                    state_contrast_correlation=float(np.corrcoef(dc.ravel(),pc.ravel())[0,1]) if pc.std()>1e-8 else None,
                    best_step=done['best_step']))
            for c in range(12):
              for j in range(16):decisions.append(dict(arm=arm,kind=kind,seed=seed,condition=condition,controller=c,
                state_index=int(d['state_index'][ii][2*j]),eta_index=(10,15)[ch[c,j]],Q=float(q[c,j,ch[c,j]]),B15=bool(good[c,j,ch[c,j]]),
                true_contrast=float(q[c,j,0]-q[c,j,1]),predicted_contrast=float(p[c,j,0]-p[c,j,1])))
    csvwrite(OUT/'source_validation_metrics.csv',rows);csvwrite(OUT/'source_validation_decisions.csv',decisions)
    def get(kind,condition):return np.mean([r['NLL'] for r in rows if r['arm']=='twelve_controller' and r['kind']==kind and r['condition']==condition and r['controllers']=='all12'])
    correct=get('entity_response_mean','correct');eta=get('eta_only','correct');wrong=get('entity_response_mean','context_state_shuffle')
    gate=bool(correct<eta and correct<wrong)
    write(OUT/'source_gate.json',dict(passed=gate,full_NLL=float(correct),eta_NLL=float(eta),same_controller_wrong_state_context_NLL=float(wrong),
        criterion='Three-seedmean sourceVAL NLL improves versus eta-only and same-controller wrong-state context. This is eligibility for independent confirmation, NOT success claim.',
        target_labels_used=False,known_controllers_seen_source_family_holdout_only=True))
    freeze();print([r for r in rows if r['condition']=='correct' and r['controllers']=='all12'])
    print(dict(source_gate=gate))

if __name__=='__main__':main()
