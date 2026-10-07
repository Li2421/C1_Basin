"""One causally motivated input ablation, reuse the frozen matched trainer."""
import argparse
import numpy as np
from . import motion_factorial_train as base
from .motion_factorial_analysis import compare,causal_details
from .phase_learning_diagnosis import summarize as prior_diagnosis
OUT=base.OUT.parent/'motion_drop_rest_control'
RULE=base.OUT.parent/'motion_drop_rest_protocol.json'


def train(index):
    fold,si=index//3,index%3
    base.KINDS=('motion_only','UNUSED')
    base.RULE=RULE
    base.train(fold*12+6+si)
    path=base.OUT/'cv'/f'fold{fold}'/'motion_intervention'/'motion_only'/f'seed{base.SEEDS[si]}'
    pred=np.load(path/'predictions.npz')
    for step in base.STEPS:
        z=pred[f'step{step}']
        assert np.array_equal(z[[12,13]],z[[14,15]]), 'Static response must not affect the prediction'
    base.write(path/'input_invariance_audit.json',dict(phase_motion_logits_equal_all_checkpoints=True,
        wrapper_sha256=base.sha(__file__),rule_sha256=base.sha(RULE),new_rollouts=0))


def summarize():
    OUT.mkdir(exist_ok=True)
    rows=[];candidates=[]
    for step in base.STEPS:
        rr=[]
        for fold in range(3):
            for seed in base.SEEDS:
                path=base.OUT/'cv'/f'fold{fold}'/'motion_intervention'/'motion_only'/f'seed{seed}'
                base.read(path/'complete.json');base.read(path/'input_invariance_audit.json')
                h=next(r for r in base.read(path/'history.json') if r['step']==step)
                rr.append(dict(fold=fold,seed=seed,step=step,**h['held_native_VAL']))
        rows+=rr
        candidates.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),
            B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in base.SEEDS],cases_per_seed=192))
    choice=min(candidates,key=lambda a:a['NLL'])
    d=np.load(base.OUT/'dataset.npz');tr,va=[np.flatnonzero(d['split']==v) for v in ('train','validation')]
    s,f=[d[k][:,va].reshape(16,16,2) for k in ('success','failure')];q=s/(s+f)
    tq=(d['success'][:,tr]/(d['success'][:,tr]+d['failure'][:,tr])).reshape(16,64,2)
    prior=np.broadcast_to(tq.mean(1)[:,None],(16,16,2));boot=np.random.default_rng(202610042325).integers(16,size=(10000,16))
    details=[];controls=[]
    for fold in range(3):
        for seed in base.SEEDS:
            path=base.OUT/'cv'/f'fold{fold}'/'motion_intervention'/'motion_only'/f'seed{seed}'
            pp=np.load(path/'predictions.npz');z=pp[f"step{choice['step']}"].reshape(16,16,2)
            for ev,cs in (('held_native',base.folds()[fold]),('motion',[14,15])):
                details.append(dict(fold=fold,seed=seed,step=choice['step'],evaluation=ev,constituents_seen=fold!=2,
                    **prior_diagnosis(z,q,s,f,prior,cs,boot),**causal_details(z,q,boot)))
                for condition in base.CONTROLS[1:]:
                    zz=pp[f"step{choice['step']}__{condition}"].reshape(16,16,2)
                    controls.append(dict(fold=fold,seed=seed,condition=condition,evaluation=ev,constituents_seen=fold!=2,
                        **compare(z,zz,q,s,f,cs,boot)))
    base.csvwrite(OUT/'source_trajectories.csv',rows);base.csvwrite(OUT/'selected_diagnosis.csv',details);base.csvwrite(OUT/'input_controls.csv',controls)
    base.write(OUT/'source_selection.json',dict(selection=choice,criterion='pooledsourceheld-nativecontrollerVALNLL',
        source_only=True,target_labels_used=False,new_rollouts=0,independent_confirmation=False))
    print(choice,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    train(a.index) if a.action=='train' else summarize()
