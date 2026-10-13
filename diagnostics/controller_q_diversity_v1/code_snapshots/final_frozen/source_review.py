"""Source-only checkpoint review and mechanism controls before confirmation."""
import csv
import argparse
import json
import numpy as np
from .experiment import HERE,read,write,sha
from .train import KINDS,SEEDS
from .evaluate import corr,interval,masked_mean,center_known,square_gain_bounds


def run(variant='source128'):
    path=HERE/'data'/variant/'dataset.npz';d=np.load(path)
    predictions={};fitrows=[]
    for kind in KINDS:
        ps=[]
        for seed in SEEDS:
            folder=HERE/'fits'/variant/kind/f'seed{seed}';r=read(folder/'complete.json')
            assert r['dataset_sha256']==sha(path)
            z=np.load(folder/'source_predictions.npz')['logits'];ps.append(1/(1+np.exp(-np.clip(z,-80,80))))
            hist=read(folder/'history.json')
            fitrows.append(dict(model=kind,seed=seed,selected=r['selected'],last=hist[-1],
                best_NLL_step=min(hist,key=lambda x:x['nll'])['step'],
                best_selection_regret_step=min(hist,key=lambda x:x['selection_regret'])['step'],
                seconds=r['seconds']))
        predictions[kind]=np.mean(ps,0)
    metrics=[];comparisons=[];bootstrap={}
    for scene in sorted(set(d['scene'])):
        for split in ('train','validation'):
            ids=np.flatnonzero((d['scene']==scene)&(d['split']==split));states=sorted(set(d['state_uid'][ids]));all_n=len(states)
            names=sorted(set(d['controller'][ids]));c=len(names)
            states=[s for s in states if np.sum(d['state_uid'][ids]==s)==c]
            ids=ids[np.isin(d['state_uid'][ids],states)];n=len(states);ii=ids.reshape(n,c)
            assert np.all(d['state_uid'][ii]==d['state_uid'][ii[:,0]][:,None])
            assert np.all(d['controller'][ii]==np.asarray(names)[None,:])
            s=d['success'][ids].reshape(n,c,16);f=d['failure'][ids].reshape(n,c,16);u=d['unknown'][ids].reshape(n,c,16)
            known=s+f>0;y=s/np.maximum(s+f,1);lo=s/(s+f+u);hi=(s+u)/(s+f+u)
            boot=np.random.default_rng(2026100937).integers(n,size=(10000,n));bootstrap[scene+'_'+split+'_indices']=boot
            rng=np.random.default_rng(2026100938);permutations=[]
            for _ in range(100):
                while True:
                    permutation=rng.permutation(n)
                    if np.all(permutation!=np.arange(n)):break
                permutations.append(permutation)
            for kind,qall in predictions.items():
                q=qall[ids].reshape(n,c,16)
                for panel in ('all','seen_controller','unseen_controller'):
                    select=np.ones(c,bool) if panel=='all' else np.isin(names,['analytic4','analytic5']) if panel=='unseen_controller' else ~np.isin(names,['analytic4','analytic5'])
                    if not select.any():continue
                    a=q[:,select];b=y[:,select];good=known[:,select]
                    choice=a.argmax(-1);rr=np.arange(n)[:,None];cc=np.arange(select.sum())[None,:]
                    metrics.append(dict(scene=scene,split=split,model=kind,panel=panel,physical_states=n,
                        total_physical_states_in_split=all_n,complete_crossed_panel=True,
                        controllers=[v for v,k in zip(names,select) if k],
                        Q_MSE=float(masked_mean((a-b)**2,good,axis=(0,1,2))),
                        Q_MAE=float(masked_mean(np.abs(a-b),good,axis=(0,1,2))),
                        state_effect_correlation=corr(center_known(a,good),center_known(b,good),good),
                        selected_verified_success=float(lo[:,select][rr,cc,choice].mean()),
                        oracle_verified_success=float(lo[:,select].max(-1).mean()),
                        numerical_count=int(u[:,select].sum())))
                matched=masked_mean((q-y)**2,known,axis=(1,2))
                controller_errors=[];controller_bounds=[]
                for shift in range(1,c):
                    wrong=np.roll(q,shift,axis=1)
                    controller_errors.append(masked_mean((wrong-y)**2,known,axis=(1,2)))
                    bound,_=square_gain_bounds(wrong,q,lo,hi);controller_bounds.append(bound.mean((1,2)))
                state_errors=[];state_bounds=[]
                for permutation in permutations:
                    wrong=q[permutation]
                    state_errors.append(masked_mean((wrong-y)**2,known,axis=(1,2)))
                    bound,_=square_gain_bounds(wrong,q,lo,hi);state_bounds.append(bound.mean((1,2)))
                values={
                    'wrong_controller_minus_matched_Q_MSE':np.mean(controller_errors,0)-matched,
                    'wrong_controller_minus_matched_Q_MSE_censor_worst':np.mean(controller_bounds,0),
                    'wrong_state_minus_matched_Q_MSE':np.mean(state_errors,0)-matched,
                    'wrong_state_minus_matched_Q_MSE_censor_worst':np.mean(state_bounds,0),
                }
                for name,value in values.items():
                    bs=value[boot].mean(1);ci=interval(bs)
                    comparisons.append(dict(scene=scene,split=split,model=kind,comparison=name,
                        physical_states=n,mean_difference=float(value.mean()),CI95=ci))
                    bootstrap['_'.join([scene,split,kind,name])]=bs
    suffix='' if variant=='source128' else '_'+variant
    write(HERE/f'source_review{suffix}.json',dict(metrics=metrics,paired_comparisons=comparisons,fit_histories=fitrows,
        dataset_sha256=sha(path),no_final_labels=True,
        interpretation='Validation labels are source-only. Controller4/5 guide checkpoint and representation selection, so are not final confirmation. All uncertainty uses physical-state clusters and is conditional on this finite controller panel.'))
    np.savez_compressed(HERE/f'source_review_bootstrap{suffix}.npz',**bootstrap)
    concise=[r for r in metrics if r['split']=='validation' and r['panel']=='unseen_controller']
    print(json.dumps(concise,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--variant',default='source128');a=ap.parse_args();run(a.variant)
