"""Locate the train -> same-scene -> held-scene generalization break."""
import numpy as np
from .learn import FOLDS, CONDITIONS, SEEDS, SCENES, INDICES
from .review import corr, centered_axes
from .study import HERE, read, write


def run():
    d=dict(np.load(HERE/'dataset.npz'));raw=dict(np.load(HERE/'raw_labels.npz'))
    rows,curves=[],[]
    for fold,held in FOLDS.items():
        for c in CONDITIONS:
            qs=[]
            for seed in SEEDS:
                p=HERE/'fits'/fold/c/f'seed{seed}'
                qs.append(np.load(p/'predictions.npz')['probability'])
                h=read(p/'history.json');b=read(p/'complete.json')['selected']
                curves.append(dict(fold=fold,condition=c,seed=seed,selected_step=b['step'],
                    start_training_nll=h[0]['training_nll'],end_training_nll=h[-1]['training_nll'],
                    best_selection_regret=b['selection_regret_upper'],last_selection_regret=h[-1]['selection_regret_upper'],
                    selected_nll=b['nll'],last_nll=h[-1]['nll'],minimum_validation_nll=min(z['nll'] for z in h),
                    selected_ranking=b['strong_pair_accuracy'],last_ranking=h[-1]['strong_pair_accuracy']))
            pred=np.mean(qs,0)
            for sc in SCENES:
                for split in ('train','validation'):
                    for ref in ('flow','goal'):
                        ix=np.flatnonzero((d['scene']==sc)&(d['split']==split)&(d['kind']==ref))
                        ix=np.array(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
                        n=len(ix)//5;shape=(n,5,4)
                        s,f,u=(d[k][ix][:,list(INDICES)].reshape(shape) for k in ('success','failure','unknown'))
                        counts=s+f+u;lo=s/counts;hi=(s+u)/counts;mid=(lo+hi)/2;q=pred[ix].reshape(shape)
                        rs,ru=(raw[k][ix][:,list(INDICES)].reshape(shape+(4,)) for k in ('success','unknown'))
                        nt=2 if split=='train' else 4
                        q0=(rs[...,:nt//2]+.5*ru[...,:nt//2]).mean(-1)
                        q1=(rs[...,nt//2:nt]+.5*ru[...,nt//2:nt]).mean(-1)
                        state_pred,state_true=(centered_axes(a,[0]) for a in (q,mid))
                        int_pred,int_true=(centered_axes(a,[0,2]) for a in (q,mid))
                        role='held_scene_development' if sc==held else 'source_training' if split=='train' else 'source_validation_checkpoint_pool'
                        if c=='encounter_goal_only' and ref=='flow' and split=='train' and sc!=held:
                            role='source_controller_reference_not_trained'
                        rows.append(dict(fold=fold,condition=c,scene=sc,split=split,reference=ref,role=role,N=n,
                            Q_MSE_midpoint_sensitivity=float(np.mean((q-mid)**2)),
                            state_effect_corr_midpoint_sensitivity=corr(state_pred,state_true),
                            state_eta_corr_midpoint_sensitivity=corr(int_pred,int_true),
                            independent_half_state_eta_corr_midpoint_sensitivity=corr(centered_axes(q0,[0,2]),centered_axes(q1,[0,2])),
                            independent_half_Q_corr_midpoint_sensitivity=corr(q0,q1),
                            state_eta_target_energy_midpoint_sensitivity=float(np.mean(centered_axes(q0,[0,2])*centered_axes(q1,[0,2]))),
                            numerical_unknown=int(u.sum())))
    write(HERE/'fit_generalization_diagnostics.json',dict(rows=rows,curves=curves,
        warning='Opened development labels. Source validation selected checkpoints and is optimistic. Training prediction/label agreement is not generalization. Censor midpoint correlations are sensitivity diagnostics, not identified truth.'))
    for r in rows:
        if r['condition']=='encounter_same_all_labels' and r['reference']=='flow' and (r['fold']=='joint' or r['role']=='held_scene_development'):
            print({k:r[k] for k in ('fold','scene','split','state_effect_corr_midpoint_sensitivity','state_eta_corr_midpoint_sensitivity','independent_half_state_eta_corr_midpoint_sensitivity')},flush=True)


if __name__=='__main__':
    run()
