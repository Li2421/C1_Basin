"""Strict split extraction from the exact-seed cache, with censored failures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import numpy as np
from shared_rollout_db.src.rollout_db import connect
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .experiment import HERE,read,write,freeze,sha,case


def assemble(stage='source_budget',train_limit=128):
    p=read(HERE/'protocol.json');b=read(HERE/'execution_budget.json')
    rows=read(HERE/'batches'/stage/'cases.json')
    final=stage=='confirmation_budget'
    if final:assert (HERE/'models_frozen.json').exists()
    rows=[r for r in rows if r['split']!='train' or r['state_index']<train_limit]
    groups=defaultdict(list)
    for r in rows:groups[r['state_uid'],r['controller_name']].append(r)
    states={s['state_uid']:s for s in p['states']}
    items=[];meta=[];counts=[];seed_success=[];seed_unknown=[];seed_collision=[]
    nominal=[];field=[];missing=[]
    with connect(True) as db:
        for (sid,name),cases in sorted(groups.items()):
            s=states[sid]
            assert (s['study_split']=='final')==final
            nseeds=16 if final else b['train_seeds'] if s['study_split']=='train' else b['validation_seeds']
            success=np.zeros((16,16),np.int8);unknown=success.copy();collision=success.copy()
            c=np.zeros((16,3),np.int32)
            for r in cases:
                old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
                if old is None:missing.append(r['rollout_uid']);continue
                assert not old['conflict_quarantined']
                assert all(old[k]==r[k] for k in ['state_uid','eta_uid','controller_uid','seed_key'])
                j,k=r['eta_index'],r['seed'];u=old['numerical_failure'];z=old['success']
                assert not (u and z)
                c[j]+=[z,int(not(z or u)),u]
                success[j,k]=z;unknown[j,k]=u;collision[j,k]=old['collision']
            assert len(cases)==16*nseeds
            f=np.load(HERE/'features'/sid/(name+'.npz'))
            items.append({k[7:]:f[k] for k in f.files if k.startswith('entity_')})
            nominal.append(np.pad(f['nominal'],((0,4-len(f['nominal'])),(0,0))))
            field.append(np.pad(f['field'],((0,4-len(f['field'])),(0,0))))
            meta.append(dict(state_uid=sid,controller=name,scene=s['scenario'],split=s['study_split'],
                             state_index=s['study_index'],trials=nseeds))
            counts.append(c);seed_success.append(success);seed_unknown.append(unknown);seed_collision.append(collision)
    if missing:
        raise RuntimeError(f'Unattempted cases: {len(missing)}; refusing incomplete panel')
    folder=HERE/'data'/('final' if final else f'source{train_limit}')
    folder.mkdir(parents=True,exist_ok=True)
    ent=rep.batch(items);out={f'entity_{k}':v for k,v in ent.items()}
    counts=np.asarray(counts);assert counts.shape[1:]==(16,3)
    for i,m in enumerate(meta):assert np.all(counts[i].sum(-1)==m['trials'])
    out.update(eta=np.asarray(p['eta'],np.float32),nominal=np.asarray(nominal),field=np.asarray(field),
        success=counts[...,0],failure=counts[...,1],unknown=counts[...,2],
        seed_success=np.asarray(seed_success),seed_unknown=np.asarray(seed_unknown),seed_collision=np.asarray(seed_collision),
        **{k:np.asarray([m[k] for m in meta]) for k in meta[0]})
    target=folder/'dataset.npz'
    if target.exists():
        prior=np.load(target)
        assert set(prior.files)==set(out)
        for k in out:np.testing.assert_array_equal(prior[k],out[k])
    else:np.savez_compressed(target,**out)
    manifest=dict(rows=len(meta),physical_states=len(set(m['state_uid'] for m in meta)),
        state_eta_pairs=len(meta)*16,attempts=int(counts.sum()),unknown=int(counts[...,2].sum()),
        stage=stage,train_limit=train_limit,dataset_sha256=sha(target),
        protocol_sha256=sha(HERE/'protocol.json'),budget_sha256=sha(HERE/'execution_budget.json'),
        feature_code_sha256=sha(HERE/'features.py'),final_labels_used=final,
        numerical_policy='Unknown; excluded from training likelihood; lower/upper bounds retained for evaluation')
    freeze(folder/'manifest.json',manifest)
    print(manifest,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='source_budget');ap.add_argument('--train-limit',type=int,default=128)
    a=ap.parse_args();assemble(a.stage,a.train_limit)
