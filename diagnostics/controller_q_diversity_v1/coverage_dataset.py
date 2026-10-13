"""Reuse labels to make equal-label-budget and all-data coverage comparisons."""
import numpy as np
from collections import defaultdict
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .experiment import HERE,read,write,freeze,sha,connect


def run():
    original=np.load(HERE/'data/source128/dataset.npz');groups=defaultdict(list)
    for r in read(HERE/'batches/source_extra/cases.json'):groups[r['state_uid'],r['controller_name']].append(r)
    physical={s['state_uid']:s for s in read(HERE/'protocol.json')['states']}
    entities=[];nominal=[];field=[];success=[];unknown=[];collision=[];meta=[]
    with connect(True) as db:
        for (sid,name),cases in sorted(groups.items()):
            s=physical[sid];assert s['study_split']=='train' and s['study_index']<64 and len(cases)==16
            ss=np.zeros((16,16),np.int8);uu=ss.copy();cc=ss.copy()
            for r in cases:
                v=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
                assert v is not None and not v['conflict_quarantined'],r['rollout_uid']
                assert all(v[k]==r[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))
                j,k=r['eta_index'],r['seed'];ss[j,k]=v['success'];uu[j,k]=v['numerical_failure'];cc[j,k]=v['collision']
            f=np.load(HERE/'features'/sid/(name+'.npz'))
            entities.append({k[7:]:f[k] for k in f.files if k.startswith('entity_')})
            nominal.append(np.pad(f['nominal'],((0,4-len(f['nominal'])),(0,0))))
            field.append(np.pad(f['field'],((0,4-len(f['field'])),(0,0))))
            success.append(ss);unknown.append(uu);collision.append(cc)
            meta.append(dict(state_uid=sid,controller=name,scene=s['scenario'],split='train',state_index=s['study_index'],trials=1))
    ss=np.asarray(success);uu=np.asarray(unknown);cc=np.asarray(collision)
    extra={f'entity_{k}':v for k,v in rep.batch(entities).items()}
    extra.update(nominal=np.asarray(nominal),field=np.asarray(field),success=ss.sum(-1),unknown=uu.sum(-1),
        failure=1-ss.sum(-1)-uu.sum(-1),seed_success=ss,seed_unknown=uu,seed_collision=cc,
        **{k:np.asarray([r[k] for r in meta]) for k in meta[0]})
    assert set(extra)==set(original.files)-{'eta'}
    full={k:np.concatenate((original[k],extra[k]),0) for k in extra};full['eta']=original['eta']
    order=np.lexsort((full['controller'],full['state_uid']))
    full={k:v if k=='eta' else v[order] for k,v in full.items()}
    for variant in ('source_diverse64','source_augmented128'):
        keep=np.ones(len(full['scene']),bool)
        if variant=='source_diverse64':keep&=~((full['split']=='train')&(full['controller']!='flow')&(full['state_index']>=64))
        out={k:v if k=='eta' else v[keep] for k,v in full.items()}
        if variant=='source_diverse64':
            assert out['success'].shape==original['success'].shape
            assert int((out['success']+out['failure']+out['unknown']).sum())==int((original['success']+original['failure']+original['unknown']).sum())
        folder=HERE/'data'/variant;folder.mkdir(parents=True,exist_ok=True);dest=folder/'dataset.npz'
        if dest.exists():
            old=np.load(dest)
            for k,v in out.items():np.testing.assert_array_equal(old[k],v)
        else:np.savez_compressed(dest,**out)
        freeze(folder/'manifest.json',dict(dataset_sha256=sha(dest),final_labels_used=False,variant=variant,
            original_dataset_sha256=sha(HERE/'data/source128/dataset.npz'),extra_profiles_sha256=sha(HERE/'extra_controller_profiles.json'),
            validation='Exactly original source validation; no extra held controllers or labels',
            rows=len(out['scene']),physical_states=len(set(out['state_uid'])),
            train_controller_state_cases=int(np.sum(out['split']=='train')),
            attempts=int((out['success']+out['failure']+out['unknown']).sum())))
        print(variant,len(out['scene']),'rows',flush=True)


if __name__=='__main__':run()
