"""Isolate controller4/5 controls on SOURCE validation; never confirmation."""
import argparse
import numpy as np
from .experiment import HERE, read, write, sha
from .train import KINDS, SEEDS
from .evaluate import masked_mean, center_known, corr, donor_bootstrap, square_gain_bounds, interval


def run(variant):
    path=HERE/'data'/variant/'dataset.npz';d=np.load(path)
    rows=[];draws={}
    for scene in sorted(set(d['scene'])):
        ids=np.flatnonzero((d['scene']==scene)&(d['split']=='validation')&np.isin(d['controller'],['analytic4','analytic5']))
        n=len(ids)//2
        assert np.all(d['controller'][ids].reshape(n,2)==['analytic4','analytic5'])
        assert np.all(d['state_uid'][ids].reshape(n,2)==d['state_uid'][ids[::2]][:,None])
        s=d['success'][ids].reshape(n,2,16);f=d['failure'][ids].reshape(n,2,16);u=d['unknown'][ids].reshape(n,2,16)
        good=s+f>0;y=s/np.maximum(s+f,1);lo=s/(s+f+u);hi=(s+u)/(s+f+u)
        boot=np.random.default_rng(2026100945).integers(n,size=(20000,n))
        for kind in KINDS:
            pred=[]
            for seed in SEEDS:
                z=np.load(HERE/'fits'/variant/kind/f'seed{seed}'/'source_predictions.npz')['logits'][ids]
                pred.append(1/(1+np.exp(-np.clip(z,-80,80))))
            q=np.mean(pred,0).reshape(n,2,16);wrong=q[:,::-1]
            mse=masked_mean((q-y)**2,good,axis=(1,2))
            control=masked_mean((wrong-y)**2-(q-y)**2,good,axis=(1,2))
            bound,_=square_gain_bounds(wrong,q,lo,hi)
            delta=y[:,0]-y[:,1];dp=q[:,0]-q[:,1];valid=good[:,0]&good[:,1]
            dc=center_known(delta,valid);pc=center_known(dp,valid)
            select=q.argmax(-1);rr=np.arange(n)[:,None];cc=np.arange(2)[None,:]
            matched=lo[rr,cc,select].mean(1)
            value_grid=np.mean(np.stack([lo[:,ci,select[:,ci]] for ci in range(2)],-1),-1)
            expected,bs,valid_draws=donor_bootstrap(value_grid,boot)
            selection_gain=matched[boot].mean(1)-bs
            grid=masked_mean((q[None]-y[:,None])**2,np.broadcast_to(good[:,None],(n,n,2,16)),axis=(2,3))
            wrong_mse,mismatch_draws,valid_mse=donor_bootstrap(grid,boot)
            mismatch_gain=mismatch_draws-mse[boot].mean(1)
            effect_skill=masked_mean(dc**2-(pc-dc)**2,valid,axis=1)
            row=dict(scene=scene,model=kind,physical_states=n,Q_MSE=float(mse.mean()),
                controller4_5_effect_corr=corr(dp,delta,valid),
                state_centered_controller_effect_corr=corr(pc,dc,valid),
                state_effect_corr=corr(center_known(q,good),center_known(y,good),good),
                matched_success=float(matched.mean()),oracle_success=float(lo.max(-1).mean()),
                per_controller_best_fixed_success=float(lo.mean(0).max(-1).mean()),
                matched_minus_state_mismatch_success=float(matched.mean()-expected),
                matched_minus_state_mismatch_success_CI=interval(selection_gain[valid_draws]),
                state_mismatch_minus_matched_Q_MSE=float(wrong_mse-mse.mean()),
                state_mismatch_minus_matched_Q_MSE_CI=interval(mismatch_gain[valid_mse]))
            for label,value in [('controller_mismatch_minus_matched_Q_MSE',control),
                                ('controller_mismatch_minus_matched_Q_MSE_censor_worst',bound.mean((1,2))),
                                ('state_centered_controller_effect_skill',effect_skill)]:
                if label=='state_centered_controller_effect_skill':
                    truth=delta[boot];prediction=dp[boot];mask=valid[boot]
                    truth=truth-masked_mean(truth,mask,axis=1)[:,None,:]
                    prediction=prediction-masked_mean(prediction,mask,axis=1)[:,None,:]
                    bs=masked_mean(truth**2-(prediction-truth)**2,mask,axis=(1,2))
                else:bs=value[boot].mean(1)
                row[label]=float(value.mean());row[label+'_CI']=interval(bs)
                draws[scene+'_'+kind+'_'+label]=bs
            rows.append(row)
    write(HERE/f'source_held_controls_{variant}.json',dict(rows=rows,dataset_sha256=sha(path),
        panel='SOURCE validation controllers4/5 only; used in model selection, not final confirmation',
        uncertainty='Physical-state bootstrap, conditional on two fixed controllers; unknown outcomes censored with worst-case sensitivity',
        final_labels_used=False))
    np.savez_compressed(HERE/f'source_held_controls_{variant}_bootstrap.npz',**draws)
    print('Source-held controls complete',variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',default='source128');a=p.parse_args();run(a.variant)
