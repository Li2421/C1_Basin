"""One joint ablation, reusing the identical source training implementation."""
import argparse
from pathlib import Path
from . import raw_input_ablation as inherited
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'raw_minimum_source'
RULE=ROOT/'raw_minimum_protocol.json'


def configure():
    inherited.OUT,inherited.RULE=OUT,RULE
    inherited.ARMS=('without_explicit_h',)
    original=inherited.data.load
    def load(fit):
        d,x,tr,va,norm=original(fit)
        d={**d,'input_context':d['input_context'].copy()}
        d['input_context'][:,:,:88]=0
        return d,x,tr,va,norm
    inherited.data.load=load


def summarize():
    # Training implementation is unchanged; retain its byte-exact snapshot.
    archive=ROOT/'raw_minimum_source_training_v1.py'
    assert inherited.sha(archive)=='5514118a038174c8a057ca4dcbc4fcaabdb1e5a9ef0b84d2f1dcfc15766868a9'
    d=dict(np.load(inherited.data.OUT/'dataset.npz'));va=np.arange(320,352)
    rows=[];controls=[];predictions={};choices=[]
    for fold in range(3):
        for seed in inherited.base.SEEDS:
            path=OUT/'without_explicit_h'/f'fold{fold}/seed{seed}'
            guard=inherited.read(path/'joint_deletion_wrapper.json')
            assert guard['wrapper_sha256'] in (inherited.sha(__file__),inherited.sha(archive))
            assert guard['underlying_code_sha256']==inherited.sha(inherited.__file__)
            predictions[(fold,seed)]=dict(np.load(path/'predictions.npz'))
    for step in inherited.base.STEPS:
        rr=[]
        for (fold,seed),p in predictions.items():
            r=dict(fold=fold,seed=seed,step=step,**inherited.base.metrics(p[f'step{step}'],d,va,inherited.base.folds()[fold]))
            rows.append(r);rr.append(r)
        choices.append(dict(step=step,NLL=float(np.mean([r['NLL'] for r in rr])),
            B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in inherited.base.SEEDS]))
    best=min(choices,key=lambda r:r['NLL'])
    for (fold,seed),p in predictions.items():
        for cond in inherited.data.CONTROLS:
            z=p[f'step{best["step"]}'+('' if cond=='correct' else '__'+cond)]
            controls.append(dict(fold=fold,seed=seed,condition=cond,
                **inherited.base.metrics(z,d,va,inherited.base.folds()[fold]),**inherited.base.causal(z,d,va)))
    inherited.base.csvwrite(OUT/'trajectories.csv',rows);inherited.base.csvwrite(OUT/'input_controls.csv',controls)
    inherited.write(OUT/'selection_frozen.json',dict(selected=best,
        full_reference=inherited.read(ROOT/'context_raw_bypass/selection_frozen.json')['selected'],
        source_only=True,target_labels_used=False,new_rollouts=0,code_sha256=inherited.sha(__file__),
        training_wrapper_sha256=inherited.sha(archive),protocol_sha256=inherited.sha(RULE),
        interpretation='Joint deletion: variable inputs are normalized eta and32state-conditioned goal-response summaries; entity masks remain.'))
    print(best,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','summarize'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args();configure()
    if a.action=='train':
        inherited.train(a.index)
        path=OUT/'without_explicit_h'/f'fold{a.index//3}/seed{inherited.base.SEEDS[a.index%3]}'/'joint_deletion_wrapper.json'
        inherited.write(path,dict(wrapper_sha256=inherited.sha(__file__),protocol_sha256=inherited.sha(RULE),
            underlying_code_sha256=inherited.sha(inherited.__file__),input_context_channels_zeroed=[0,88],
            explicit_h_zeroed=True,entity_masks_preserved=True,target_labels_used=False,new_rollouts=0))
    else:summarize()
