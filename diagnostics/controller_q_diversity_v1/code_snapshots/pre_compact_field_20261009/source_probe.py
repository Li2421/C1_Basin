"""Cheap falsification using only the already collected TRAIN pool.

Fit controllers0/1/2 on48 states; diagnose controller3 and the next16 states.
This is DEVELOPMENT, never the independent final confirmation.
"""
from collections import defaultdict
import json
import numpy as np
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .experiment import HERE, read, write, freeze, sha, connect
from .train import fit


def assemble():
    p=read(HERE/'protocol.json')
    requests=[r for r in read(HERE/'batches/source_q1/cases.json')
        if r['split']=='train' and r['state_index']<64 and r['controller_name'].startswith('analytic')]
    states={s['state_uid']:s for s in p['states']};groups=defaultdict(list)
    for r in requests:groups[r['state_uid'],r['controller_name']].append(r)
    items=[];meta=[];counts=[];nominal=[];field=[]
    with connect(True) as db:
        for (sid,ctl),cases in sorted(groups.items()):
            s=states[sid];c=np.zeros((16,3),int)
            assert len(cases)==16
            for r in cases:
                record=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
                if record is None:
                    record=read(HERE/'batches/source_q1/raw'/(r['rollout_uid']+'.json'))
                    assert record['case']==r and record['execution_semantics_version']=='scene_precision_guard_v1'
                else:assert not record['conflict_quarantined']
                u=record['numerical_failure'];z=record['success'];c[r['eta_index']]=[z,1-z-u,u]
            f=np.load(HERE/'features'/sid/(ctl+'.npz'))
            items.append({k[7:]:f[k] for k in f.files if k.startswith('entity_')})
            nominal.append(np.pad(f['nominal'],((0,4-len(f['nominal'])),(0,0))))
            field.append(np.pad(f['field'],((0,4-len(f['field'])),(0,0))))
            train=s['study_index']<48 and ctl!='analytic3'
            meta.append(dict(state_uid=sid,scene=s['scenario'],controller=ctl,
                split='train' if train else 'validation',state_index=s['study_index'],trials=1))
            counts.append(c)
    counts=np.asarray(counts)
    out={f'entity_{k}':v for k,v in rep.batch(items).items()}
    out.update(eta=np.asarray(p['eta'],np.float32),nominal=np.asarray(nominal),field=np.asarray(field),
        success=counts[...,0],failure=counts[...,1],unknown=counts[...,2],
        **{k:np.asarray([m[k] for m in meta]) for k in meta[0]})
    folder=HERE/'data/source64';folder.mkdir(parents=True,exist_ok=True);target=folder/'dataset.npz'
    if target.exists():
        d=np.load(target)
        for k,v in out.items():np.testing.assert_array_equal(d[k],v)
    else:np.savez_compressed(target,**out)
    freeze(folder/'manifest.json',dict(dataset_sha256=sha(target),final_labels_used=False,
        source='Original TRAIN only; first64 physical states/scene',
        development_train='first48 states, analytic0/1/2',
        development_validation='analytic3 all64 states plus next16 states all four controllers',
        independent_final_confirmation=False,attempts=int(counts.sum()),unknown=int(counts[...,2].sum())))


def analyze():
    d=np.load(HERE/'data/source64/dataset.npz');s=d['success'];f=d['failure'];known=(s+f)>0
    y=np.divide(s,s+f,out=np.zeros_like(s,dtype=float),where=known)
    rows=[]
    for kind in ('eta_only','state_only','state_context','state_field'):
        folder=HERE/'fits/source64'/kind/'seed17'
        z=np.load(folder/'source_predictions.npz')['logits'];q=1/(1+np.exp(-np.clip(z,-80,80)))
        for scene in sorted(set(d['scene'])):
            for panel in ('train','unseen_state_seen_controller','seen_state_unseen_controller','unseen_both'):
                select=(d['scene']==scene)
                select&=(d['state_index']<48) if panel in ('train','seen_state_unseen_controller') else (d['state_index']>=48)
                select&=(d['controller']=='analytic3') if 'unseen_controller' in panel or panel=='unseen_both' else (d['controller']!='analytic3')
                rows.append(dict(model=kind,scene=scene,panel=panel,state_controller_rows=int(select.sum()),
                    Q_MSE=float(np.mean((q[select]-y[select])[known[select]]**2)),
                    Q_MAE=float(np.mean(np.abs(q[select]-y[select])[known[select]])),
                    numerical_count=int(d['unknown'][select].sum())))
        # Each matched controller is tested against every other TRAIN controller
        # at exactly the same physical state; labels remain TRAIN-only evidence.
        for scene in sorted(set(d['scene'])):
            ids=np.flatnonzero((d['scene']==scene)&(d['state_index']>=48))
            a=q[ids].reshape(16,4,16);b=y[ids].reshape(16,4,16);m=known[ids].reshape(16,4,16)
            delta_pred=a[:,:,None]-a[:,None,:];delta_true=b[:,:,None]-b[:,None,:]
            mask=m[:,:,None]&m[:,None,:]&(~np.eye(4,dtype=bool)[None,:,:,None])
            rows.append(dict(model=kind,scene=scene,panel='unseen_state_controller_contrast',
                contrast_MSE=float(np.mean((delta_pred-delta_true)[mask]**2)),
                zero_contrast_MSE=float(np.mean(delta_true[mask]**2))))
    write(HERE/'development_probe64_results.json',dict(rows=rows,
        final_labels_used=False,caveat='Exploratory reuse of original TRAIN states; Q1. Not final confirmation.',
        training_seed=17,controller_training_count=3))
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    assemble()
    for kind in ('eta_only','state_only','state_context','state_field'):fit(kind,17,64)
    analyze()
