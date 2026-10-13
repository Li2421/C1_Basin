"""Exploratory, label-safe P1 static representation/behavior diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE=Path(__file__).resolve().parent
P1=HERE.parent/'controller_q_transfer_round3_v1'
def output(obj):(HERE/'representation_results.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def corr(a,b):
    value=spearmanr(a,b).statistic
    return float(value) if np.isfinite(value) else None

def main():
    p=json.loads((P1/'P1_PROTOCOL.json').read_text());names=p['controller_order'];n=len(names)
    with np.load(P1/'P1_LABELS.npz') as z:
        assert z['controller_names'].tolist()==names
        y=z['Y'].copy();u=z['U'].copy()
    q=y.mean(-1);qhi=(y+u).mean(-1)
    features={k:[] for k in ('R3','R4_action','R5','raw_behavior','near_goal_moving')}
    replay=[]
    for name in names:
        with np.load(HERE/'features'/(name+'.npz')) as z, np.load(P1/'p1_features'/(name+'.npz')) as old:
            assert z['state_uids'].tolist()==old['state_uids'].tolist()
            features['R3'].append(old['R3'].copy());features['R4_action'].append(old['R4_action'].copy())
            features['R5'].append(z['R5'].copy())
            features['raw_behavior'].append(np.concatenate((z['nominal'],z['raw_field']),axis=-1))
            near=z['raw_field'].reshape(24,4,7,3,5,2)[:,:,4:7].reshape(24,-1)
            features['near_goal_moving'].append(near)
            r5_near=z['R5'].reshape(24,4,238)[:,:,148:].reshape(24,-1)
            replay.append(float(np.max(np.abs(near-r5_near))))
    for k in features:features[k]=np.stack(features[k])
    assert max(replay)<1e-6
    pairs=[]
    for c in range(n):
        for d in range(c+1,n):
            for s in range(24):
                r=dict(c=names[c],d=names[d],state_index=s,
                       Q_abs_mean=float(np.abs(q[c,s]-q[d,s]).mean()),
                       Q_abs_unknown_safe_lower=float(np.maximum(0,np.maximum(q[c,s]-qhi[d,s],q[d,s]-qhi[c,s])).mean()),
                       empirical_pref_conflict=bool(np.argmax(q[c,s])!=np.argmax(q[d,s])))
                for k,a in features.items():r[k+'_distance']=float(np.linalg.norm(a[c,s]-a[d,s]))
                pairs.append(r)
    dist={k:[r[k+'_distance'] for r in pairs] for k in features}
    qdiff=[r['Q_abs_mean'] for r in pairs]
    overall=dict(source_states=24,controllers=n,controller_pairs=n*(n-1)//2,
                 evaluated_pair_states=len(pairs),R5_near_goal_replay_max_error=max(replay),
                 distance_vs_raw_behavior_spearman={k:corr(dist[k],dist['raw_behavior']) for k in ('R3','R4_action','R5')},
                 distance_vs_executed_action_probe_spearman={k:corr(dist[k],dist['R4_action']) for k in ('R3','R5')},
                 distance_vs_Q_abs_spearman={k:corr(dist[k],qdiff) for k in ('R3','R4_action','R5')},
                 distance_vs_near_goal_moving_spearman={k:corr(dist[k],dist['near_goal_moving']) for k in ('R3','R4_action','R5')},
                 exact_R3_alias_with_nonzero_raw_behavior=sum(r['R3_distance']<1e-6 and r['raw_behavior_distance']>1e-4 for r in pairs),
                 exact_R5_alias_with_nonzero_raw_behavior=sum(r['R5_distance']<1e-6 and r['raw_behavior_distance']>1e-4 for r in pairs))
    overall['closest_5pct_sampled_behavior']={}
    for k in ('R3','R4_action','R5'):
        threshold=float(np.quantile(dist[k],.05))
        subset=[r for r in pairs if r[k+'_distance']<=threshold]
        overall['closest_5pct_sampled_behavior'][k]=dict(threshold=threshold,count=len(subset),
            max_raw_behavior_distance=max(r['raw_behavior_distance'] for r in subset),
            max_executed_action_probe_distance=max(r['R4_action_distance'] for r in subset),
            max_Q_abs_mean=max(r['Q_abs_mean'] for r in subset))
    # Source-only feature scaling and source-only Q; held checkpoint outcomes
    # are accessed only for scoring after neighbor/prediction are fixed.
    loco={k:[] for k in ('R3','R4_action','R5')}
    for held in range(n):
        sources=[i for i in range(n) if i!=held]
        for k in loco:
            a=features[k]
            train=a[sources].reshape(-1,a.shape[-1]);mean=train.mean(0);scale=np.maximum(train.std(0),.05)
            ds=np.linalg.norm((a[sources]-(a[held][None]))/scale,axis=-1) # source x state
            nearest=np.asarray(sources)[np.argmin(ds,axis=0)]
            prediction=q[nearest,np.arange(24)]
            target=q[held];target_hi=qhi[held]
            selected=np.argmax(prediction,axis=1)
            selected_value=target[np.arange(24),selected]
            selected_high=target_hi[np.arange(24),selected]
            source_fixed=np.argmax(q[sources].mean(axis=(0,1)))
            fixed_value=target[:,source_fixed]
            fixed_high=target_hi[:,source_fixed]
            target_best=target.max(axis=1)
            near_dist=np.linalg.norm(features['near_goal_moving'][nearest,np.arange(24)]-features['near_goal_moving'][held],axis=-1)
            dlow=np.maximum(0,np.maximum(target-prediction,prediction-target_hi))
            dhigh=np.maximum(np.abs(prediction-target),np.abs(prediction-target_hi))
            gain_lower=np.where(selected==source_fixed,0,selected_value-fixed_high)
            gain_upper=np.where(selected==source_fixed,0,selected_high-fixed_value)
            loco[k].append(dict(held=names[held],q_mse_lower=float(np.mean((prediction-target)**2)),
                q_mse_unknown_safe_interval=[float(np.mean(dlow**2)),float(np.mean(dhigh**2))],
                selected_Q_lower=float(selected_value.mean()),source_fixed_eta=int(source_fixed),
                source_fixed_Q_lower=float(fixed_value.mean()),oracle_K16_Q_lower=float(target_best.mean()),
                selected_minus_fixed_Q_lower=float((selected_value-fixed_value).mean()),
                selected_minus_fixed_unknown_safe_interval=[float(gain_lower.mean()),float(gain_upper.mean())],
                mean_nearest_near_goal_moving_distance=float(near_dist.mean()),
                nearest_controller_names=[names[i] for i in nearest]))
    overall['loco_exploratory']={k:dict(mean_Q_MSE=float(np.mean([r['q_mse_lower'] for r in v])),
        mean_selected_minus_source_fixed=float(np.mean([r['selected_minus_fixed_Q_lower'] for r in v])),
        mean_nearest_near_goal_moving_distance=float(np.mean([r['mean_nearest_near_goal_moving_distance'] for r in v])),
        per_held=v) for k,v in loco.items()}
    overall['discovery_pair']=[r for r in pairs if r['c']=='base_u_v2_macflow' and r['d']=='base_u_v10_local_macflow']
    overall['limitations']='Opened P1 Q4 states/checkpoints; pair-state observations dependent; distance correlations and LOCO nearest-source estimates are exploratory, not generalization guarantees.'
    overall['new_outcomes']=0;overall['new_training']=0
    output(overall)
    print(json.dumps({k:v for k,v in overall.items() if k not in ('discovery_pair','loco_exploratory')},indent=2))

if __name__=='__main__':main()
