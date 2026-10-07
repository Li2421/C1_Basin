"""Evaluate frozen input-replacement predictions on independently drawn seeds."""
import json
import numpy as np
from scipy.special import expit
from .replication_analysis import OUT,SRC,load,read,write,csvwrite,block_b15

def main():
    states,y,keys=load();oldstates=read(SRC/'source_states.json');lookup={s['uid']:i for i,s in enumerate(states)}
    paths=[(SRC,k) for k in ('eta_only','full_raw','wide_db_full_raw','wide_db_full_scaled')]
    role=OUT/'role_control'
    if all((role/'models'/k/'fold2_seed41/complete.json').exists() for k in ('wide_db_role_zero','wide_db_role_aware')):
        paths.extend((role,k) for k in ('wide_db_role_zero','wide_db_role_aware'))
    response=OUT/'response_control'
    if all((response/'models'/k/'fold2_seed41/complete.json').exists() for k in ('response_zero','response_mean','response_entity')):
        paths.extend((response,k) for k in ('response_zero','response_mean','response_entity'))
    s=np.nansum(y[...,16:],axis=-1);f=np.nansum(1-y[...,16:],axis=-1);q=s/(s+f)
    rng=np.random.default_rng(2026100423);boot=rng.integers(len(states),size=(5000,len(states)))
    result=[];changes=[]
    conditions=('correct','state_shuffle','context_state_shuffle','joint_state_context_shuffle','wrong_controller')
    for root,kind in paths:
      for seed in (17,23,41):
        predictions={}
        for condition in conditions:
            z=np.full(y.shape[:3],np.nan)
            for fold in range(3):
                p=np.load(root/'models'/kind/f'fold{fold}_seed{seed}'/'predictions.npz')
                for j,i in enumerate(p['indices']):
                    state=oldstates[int(i)//16]['uid'];e=(10,15).index(int(i)%16)
                    if state in lookup:z[:,lookup[state],e]=p['best_'+condition][:,j]
            assert np.isfinite(z).all();predictions[condition]=z
        base=predictions['correct'];basep=expit(base)
        base_loss=(s*np.logaddexp(0,-base)+f*np.logaddexp(0,base))/(s+f)
        base_choice=base.argmax(-1)
        for condition,z in predictions.items():
            p=expit(z);loss=(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/(s+f);choice=z.argmax(-1)
            delta=(loss-base_loss).mean((0,2));interval=np.quantile(delta[boot].mean(1),[.025,.975])
            counts=dict(B15=0,unknown=0,oracle=0,rescue_vs_correct=0,break_vs_correct=0)
            selected=[];regret=[]
            for c in range(2):
             for i in range(len(states)):
                e=choice[c,i];be=base_choice[c,i];selected.append(q[c,i,e]);regret.append(q[c,i].max()-q[c,i,e])
                for b in range(1,4):
                    good,bad,unknown=block_b15(y[c,i,:,b*16:(b+1)*16])
                    counts['B15']+=int(good[e]);counts['unknown']+=int(unknown[e]);counts['oracle']+=int(good.any())
                    counts['rescue_vs_correct']+=int(good[e] and bad[be]);counts['break_vs_correct']+=int(bad[e] and good[be])
            result.append(dict(kind=kind,seed=seed,condition=condition,NLL=float(loss.mean()),MAE=float(abs(p-q).mean()),
                NLL_delta_from_correct=float(delta.mean()),paired_family_CI_lower=float(interval[0]),paired_family_CI_upper=float(interval[1]),
                mean_abs_probability_change=float(abs(p-basep).mean()),changed_top1=int((choice!=base_choice).sum()),
                selected_Q48=float(np.mean(selected)),regret=float(np.mean(regret)),**counts))
    csvwrite(OUT/'fresh_input_replacement_metrics.csv',result)
    print(json.dumps([r for r in result if r['condition'] in ('correct','wrong_controller')],indent=2))

if __name__=='__main__':main()
