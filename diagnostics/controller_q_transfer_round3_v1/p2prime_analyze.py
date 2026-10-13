"""Primary held-placement η-selection test with paired physical-state CI."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text())
def write(name,v):(HERE/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def ci(v,draws,alpha=.05):
    z=np.asarray(v)
    return np.quantile(z[draws].mean(1),[alpha/2,1-alpha/2]).tolist()


def main():
    assert read('P2PRIME_TRAINING.json')['status']=='COMPLETE'
    p=read('P2PRIME_PROTOCOL.json');lab=np.load(HERE/'P2PRIME_LABELS.npz')['outcomes']
    true=lab[5:,288:]
    Y=(true[:,:,:,4:]==1).mean(-1);U=(true[:,:,:,4:]==-1).mean(-1)
    observed=np.divide(Y,1-U,out=np.full_like(Y,np.nan),where=U<1)
    disc_success=(true[:,:,:,:4]==1).sum(-1)
    disc_valid=((true[:,:,:,:4]==1)|(true[:,:,:,:4]==0)).sum(-1)
    discovery=np.divide(disc_success,disc_valid,out=np.full_like(disc_success,np.nan,dtype=float),where=disc_valid>0)
    target_fixed=np.nanmean(discovery,axis=1).argmax(-1)
    target_hindsight=Y.mean(1).argmax(-1)
    fixedmax=Y.mean(1).max(-1)
    source=lab[:5,:256,:,:4]
    historical=np.load(ROOT/'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/source_dataset.npz')
    source_prior=[];source_fixed=[]
    for c in range(5):
        succ=(source[c]==1).sum((0,2));fail=(source[c]==0).sum((0,2))
        if c==0:
            succ=succ+historical['success'][:512].sum(0)
            fail=fail+historical['failure'][:512].sum(0)
        rate=succ/np.maximum(succ+fail,1)
        source_prior.append(rate);source_fixed.append(int(rate.argmax()))
    pooled=np.asarray(source_prior).mean(0);pooled_fixed=int(pooled.argmax())
    source_r4=[]
    for name in p['source_controllers']:
        z=np.load(HERE/'p2prime_features'/f'{name}.npz');source_r4.append(z['R4_action'][256:288].mean(0))
    held_r4=[]
    for name in p['held_controllers']:
        z=np.load(HERE/'p2prime_features'/f'{name}.npz');held_r4.append(z['R4_action'][:32].mean(0))
    dist=np.linalg.norm(np.asarray(held_r4)[:,None]-np.asarray(source_r4)[None],axis=-1)
    nearest=dist.argmin(-1);near_fixed=[source_fixed[i] for i in nearest]
    draws=np.random.default_rng(2026101208).integers(64,size=(5000,64))
    predictions={}
    for count in (3,5):
        for arm in ('R0','R1','R3','R4_action','R3_R4'):
            z=np.load(HERE/'p2prime_fits'/f'C{count}_{arm}'/'predictions.npz')
            predictions[f'C{count}_{arm}']=dict(matched=z['held'],mismatched=z['mismatched'])
    rows=[]
    for count in (3,5):
        for arm in ('R0','R1','R3','R4_action','R3_R4'):
            logits=predictions[f'C{count}_{arm}']['matched'];mismatch=predictions[f'C{count}_{arm}']['mismatched']
            R0=predictions[f'C{count}_R0']['matched']
            for c,name in enumerate(p['held_controllers']):
                ii=np.arange(64)
                choices=logits[:,c].argmax(-1);wrong=mismatch[:,c].argmax(-1);r0=R0[:,c].argmax(-1)
                selected_per_fit=np.take_along_axis(Y[c][None],choices[:,:,None],axis=2)[:,:,0]
                selected=selected_per_fit.mean(0)
                selected_unknown=np.take_along_axis(U[c][None],choices[:,:,None],axis=2)[:,:,0].mean(0)
                selected_observed=np.nanmean(np.take_along_axis(observed[c][None],choices[:,:,None],axis=2)[:,:,0],axis=0)
                wrong_per_fit=np.take_along_axis(Y[c][None],wrong[:,:,None],axis=2)[:,:,0]
                wrongvalue=wrong_per_fit.mean(0)
                wrongunknown=np.take_along_axis(U[c][None],wrong[:,:,None],axis=2)[:,:,0].mean(0)
                r0value=np.take_along_axis(Y[c][None],r0[:,:,None],axis=2)[:,:,0].mean(0)
                r0unknown=np.take_along_axis(U[c][None],r0[:,:,None],axis=2)[:,:,0].mean(0)
                baselines=dict(pooled=Y[c,:,pooled_fixed],nearest=Y[c,:,near_fixed[c]],R0=r0value,
                               target_discovery=Y[c,:,target_fixed[c]],target_hindsight=Y[c,:,target_hindsight[c]])
                baseline_unknown=dict(pooled=U[c,:,pooled_fixed],nearest=U[c,:,near_fixed[c]],R0=r0unknown,
                                      target_discovery=U[c,:,target_fixed[c]],target_hindsight=U[c,:,target_hindsight[c]])
                diffs={key:selected-value for key,value in baselines.items()}
                fixed_index=dict(pooled=pooled_fixed,nearest=near_fixed[c],target_discovery=target_fixed[c],target_hindsight=target_hindsight[c])
                robust_diffs={key:(selected_per_fit-Y[c,:,j][None]-U[c,:,j][None]*(choices!=j)).mean(0)
                              for key,j in fixed_index.items()}
                robust_diffs['R0']=(selected_per_fit-np.take_along_axis(Y[c][None],r0[:,:,None],axis=2)[:,:,0]
                                   -np.take_along_axis(U[c][None],r0[:,:,None],axis=2)[:,:,0]*(choices!=r0)).mean(0)
                ci_dict={key:ci(value,draws) for key,value in diffs.items()}
                family_ci={key:ci(value,draws,.05/6) for key,value in diffs.items()}
                robust_ci={key:ci(value,draws,.05/6) for key,value in robust_diffs.items()}
                matched_diff=selected-wrongvalue
                robust_matched_diff=(selected_per_fit-wrong_per_fit
                                     -np.take_along_axis(U[c][None],wrong[:,:,None],axis=2)[:,:,0]*(choices!=wrong)).mean(0)
                allfixed_boot=selected[draws].mean(1)-Y[c][draws].mean(1).max(1)
                robust_allfixed=(selected_per_fit[:,:,None]-Y[c][None]-U[c][None]*(choices[:,:,None]!=np.arange(16))).mean(0)
                allfixed_robust_boot=robust_allfixed[draws].mean(1).min(1)
                qtruth=np.divide(Y[c],1-U[c],out=np.full_like(Y[c],np.nan),where=U[c]<1)
                mse=float(np.nanmean((expit(logits[:,c])-qtruth[None])**2))
                mean_prediction=expit(logits[:,c]).mean(0)
                state_rank=[]
                for state_index in range(64):
                    valid=np.isfinite(qtruth[state_index])
                    if valid.sum()<3 or np.std(qtruth[state_index,valid])<1e-12:continue
                    state_rank.append(float(spearmanr(mean_prediction[state_index,valid],qtruth[state_index,valid]).statistic))
                oracle_best=qtruth[np.arange(64),mean_prediction.argmax(-1)]
                valid_oracle=np.isfinite(oracle_best) & np.isfinite(qtruth).any(-1)
                oracle_tie_match=float(np.mean(oracle_best[valid_oracle]>=np.nanmax(qtruth[valid_oracle],axis=-1)-1e-12)) if valid_oracle.any() else None
                row=dict(source_controller_count=count,arm=arm,held_controller=name,states=64,fit_seeds=[17,23,41],
                         selected_success=float(selected.mean()),selected_success_observed=float(np.nanmean(selected_observed)),
                         selected_success_upper=float((selected+selected_unknown).mean()),selected_unknown=float(selected_unknown.mean()),
                         success_by_fit=np.take_along_axis(Y[c][None],choices[:,:,None],axis=2)[:,:,0].mean(1).tolist(),
                         baselines={key:float(value.mean()) for key,value in baselines.items()},
                         baseline_unknown={key:float(value.mean()) for key,value in baseline_unknown.items()},
                         gains={key:float(value.mean()) for key,value in diffs.items()},gain_ci95=ci_dict,
                         gain_familywise_ci=family_ci,unknown_robust_gains={key:float(value.mean()) for key,value in robust_diffs.items()},
                         unknown_robust_familywise_ci=robust_ci,
                         matched_minus_mismatched=float(matched_diff.mean()),matched_minus_mismatched_ci95=ci(matched_diff,draws),
                         mismatch_familywise_ci=ci(matched_diff,draws,.05/6),
                         unknown_robust_mismatch_familywise_ci=ci(robust_matched_diff,draws,.05/6),
                         all16_fixed_simultaneous_gain=float(selected.mean()-fixedmax[c]),
                         all16_fixed_simultaneous_ci95=np.quantile(allfixed_boot,[.025,.975]).tolist(),
                         all16_fixed_familywise_ci=np.quantile(allfixed_boot,[.05/12,1-.05/12]).tolist(),
                         unknown_robust_all16_fixed_familywise_ci=np.quantile(allfixed_robust_boot,[.05/12,1-.05/12]).tolist(),
                         worst_unknown_gain_vs_pooled=float((selected-baselines['pooled']-U[c,:,pooled_fixed]).mean()),
                         finite_seed_oracle_regret=float((Y[c].max(1)-selected).mean()),q_mse=mse,
                         state_eta_rank_spearman=float(np.mean(state_rank)) if state_rank else None,
                         state_eta_rank_evaluable_states=len(state_rank),state_eta_oracle_tie_match=oracle_tie_match)
                rows.append(row)
    ycontrast=Y[1].mean(0)-Y[0].mean(0)
    pattern=[]
    for count in (3,5):
        for arm in ('R0','R1','R3','R4_action','R3_R4'):
            logits=predictions[f'C{count}_{arm}']['matched'];model=expit(logits).mean((0,2))
            pred=model[1]-model[0]
            corr=float(np.corrcoef(ycontrast,pred)[0,1]) if np.std(pred)>1e-10 and np.std(ycontrast)>0 else None
            pattern.append(dict(source_controller_count=count,arm=arm,controller_eta_pattern_rho=corr))
    qualifications=[]
    for r in rows:
        if r['source_controller_count']!=5 or r['arm'] in ('R0','R1'):continue
        zero_shot=all(r['unknown_robust_familywise_ci'][key][0]>0 for key in ('pooled','nearest','R0')) and r['unknown_robust_mismatch_familywise_ci'][0]>0
        state_gain=zero_shot and r['unknown_robust_all16_fixed_familywise_ci'][0]>0
        qualifications.append(dict(arm=r['arm'],held_controller=r['held_controller'],source_prior_transfer_confirmed=zero_shot,
                                   state_selection_gain_confirmed=state_gain))
    strong=any(q['state_selection_gain_confirmed'] for q in qualifications)
    result=dict(status='COMPLETE' if strong else 'COMPLETE; NO RELIABLE STATE SELECTION TRANSFER',
                primary_confirmed=strong,source_prior_transfer_confirmed=any(q['source_prior_transfer_confirmed'] for q in qualifications),qualifications=qualifications,rows=rows,
                source_best_fixed_indices=source_fixed,pooled_best_fixed_eta=pooled_fixed,nearest_source_indices=nearest.tolist(),
                nearest_fixed_eta=near_fixed,static_r4_distances=dist.tolist(),target_discovery_best_fixed_eta=target_fixed.tolist(),
                target_hindsight_best_fixed_eta=target_hindsight.tolist(),controller_eta_pattern=pattern,
                bootstrap_unit='64 jointly held physical states; 5000 paired resamples',
                new_rollouts=read('P2PRIME_COST.json')['new'],unknown_total=int((lab==-1).sum()),
                interpretation='Held TF/FT placement on native v10 Ring, not held real Flow checkpoint or cross-geometry transfer.')
    write('P2PRIME_RESULTS.json',result)
    print(json.dumps({'primary_confirmed':result['primary_confirmed'],'qualifications':qualifications,
                      'C5':[dict(arm=r['arm'],held=r['held_controller'],success=r['selected_success'],
                                 gain_pool=r['gains']['pooled'],ci_pool=r['gain_ci95']['pooled'],
                                 matched_minus_mismatched=r['matched_minus_mismatched']) for r in rows if r['source_controller_count']==5]},indent=2))


if __name__=='__main__':main()
