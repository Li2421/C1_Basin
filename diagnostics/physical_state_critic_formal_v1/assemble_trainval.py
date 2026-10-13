"""Isolate complete TRAIN/VAL labels for fair paired critic fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.state_nuisance_rootcause_v1.adaptive_selection_predict_v1 import raw23

HERE=Path(__file__).resolve().parent
PANELS={
 'within_toy_TT':('matched',[('toy_give_way','TT')]),
 'within_toy_FF':('matched',[('toy_give_way','FF')]),
 'within_ring_TT':('matched',[('ring_exchange','TT')]),
 'within_db_FF':('loso',[('double_bottleneck',None)]),
 'within_four_FF':('loso',[('four_way_intersection',None)]),
 'loso_heldout_toy':('loso',[(s,None) for s in ('double_bottleneck','four_way_intersection','ring_exchange')]),
 'loso_heldout_db':('loso',[(s,None) for s in ('toy_give_way','four_way_intersection','ring_exchange')]),
 'loso_heldout_four':('loso',[(s,None) for s in ('toy_give_way','double_bottleneck','ring_exchange')]),
 'loso_heldout_ring':('loso',[(s,None) for s in ('toy_give_way','double_bottleneck','four_way_intersection')]),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main(name):
    mode,sources=PANELS[name]
    output=HERE/'fits'/name
    output.mkdir(parents=True,exist_ok=True)
    target=output/'trainval.npz'
    if target.exists():
        print(json.dumps(dict(panel=name,already_assembled=True,sha256=sha(target))))
        return
    all_rows=[];manifests=[]
    for split in ('train','validation'):
        for scene,chain in sources:
            panel=(HERE/'matched'/scene/chain/split if mode=='matched'
                   else HERE/'loso'/scene/split)
            outcome=panel/'outcomes.npz'
            d=np.load(outcome)
            assert d['seen'].all(),f'incomplete {panel}'
            assert d['values'].shape==({'train':(512,16,4,5),'validation':(32,17,16,5)}[split])
            values=d['values'][:,:16]
            success=np.sum((values[...,0]==1)&(values[...,4]==0),axis=2).astype(np.int16)
            numerical=np.sum(values[...,4]==1,axis=2).astype(np.int16)
            failure=np.sum((values[...,0]==0)&(values[...,4]==0),axis=2).astype(np.int16)
            assert np.all(success+failure+numerical==(4 if split=='train' else 16))
            if mode=='matched':
                protocol=read(HERE/'matched_protocol.json')
                states=[s for s in protocol['states'] if s['scenario']==scene and s['metadata']['split']==split]
                physical=[s['physical'] for s in states]
                ids=[s['state_uid'] for s in states]
            else:
                file=read(HERE/f'states_{scene}.json')
                source=[s for s in file['states'] if s['split']==split]
                proposals=read(panel/'proposals.json')['states']
                by_alias={s['alias']:s for s in source}
                states=[by_alias[p['state_alias']] for p in proposals]
                physical=[s['physical'] for s in states]
                ids=[p['state_uid'] for p in proposals]
            assert len(ids)==len(d['state_uids']) and list(d['state_uids'])==ids
            eta=d['eta'][:,:16] if mode=='loso' else np.broadcast_to(d['eta'][:16],(len(ids),16,3))
            for i in range(len(ids)):
                all_rows.append(dict(scene=scene,split=split,state_uid=ids[i],physical=physical[i],
                                     eta=eta[i],success=success[i],failure=failure[i],numerical=numerical[i]))
            manifests.append(dict(scene=scene,split=split,outcome_sha256=sha(outcome),
                                  candidate_manifest_sha256=sha(panel/'proposals.json') if mode=='loso' else sha(HERE/'matched_protocol.json')))
    assert len(all_rows)==len(sources)*(512+32)
    assert len({r['state_uid'] for r in all_rows})==len(all_rows)
    train=[r for r in all_rows if r['split']=='train']
    val=[r for r in all_rows if r['split']=='validation']
    rows=train+val
    eta=np.stack([r['eta'] for r in rows]).astype(np.float32)
    if mode=='matched':
        normal=np.load(HERE.parent/'state_nuisance_rootcause_v1'/'real_state_expansion'/'physical_inputs.npz')
        np.testing.assert_array_equal(eta[0],np.asarray(read(HERE/'matched_protocol.json')['eta'],np.float32))
        # The prior fixed-bank eta preprocessing is fixed before this experiment.
        raw_bank=np.asarray(read(HERE/'matched_protocol.json')['eta'],np.float32)
        a=np.column_stack([raw_bank,np.ones(16,np.float32)])
        transform=np.linalg.lstsq(a,normal['eta'],rcond=None)[0]
        np.testing.assert_allclose(a@transform,normal['eta'],rtol=1e-6,atol=1e-6)
        eta=np.broadcast_to(normal['eta'],eta.shape).copy()
    else:
        eta=(eta-np.array([.875,0.,.375],np.float32))/np.array([.375,.5,.375],np.float32)
    dataset=dict(eta=eta,success=np.stack([r['success'] for r in rows]),
                 failure=np.stack([r['failure'] for r in rows]),
                 numerical=np.stack([r['numerical'] for r in rows]),
                 state_uids=np.asarray([r['state_uid'] for r in rows]),
                 scene=np.asarray([r['scene'] for r in rows]),
                 split=np.asarray([r['split'] for r in rows]))
    if name=='within_ring_TT':
        normal=np.load(HERE.parent/'state_nuisance_rootcause_v1'/'real_state_expansion'/'physical_inputs.npz')
        raw=raw23([dict(physical=r['physical']) for r in rows])
        dataset['physical23']=(raw-normal['physical_center'])/normal['physical_scale']
        assert dataset['physical23'].shape==(544,23)
    else:
        x=physical_entities.batch([dict(scene=r['scene'],physical=r['physical']) for r in rows])
        dataset.update({'entity_'+k:v for k,v in x.items()})
        assert all(np.isfinite(v).all() for v in x.values())
    assert np.isfinite(eta).all()
    np.savez_compressed(target,**dataset)
    manifest=dict(schema='formal_q_trainval_isolated_v1',panel=name,
                  sources=[s for s,_ in sources],train_states=len(train),validation_states=len(val),
                  train_pairs=len(train)*16,validation_pairs=len(val)*16,
                  input='physical23+eta' if name=='within_ring_TT' else 'masked physical entities+eta',
                  test_labels_opened=False,source_manifests=manifests,
                  trainval_sha256=sha(target),plan_sha256=sha(HERE/'unified_train_plan.json'))
    (output/'trainval_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(panel=name,train_states=len(train),validation_states=len(val),sha256=sha(target))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=tuple(PANELS),required=True)
    a=p.parse_args();main(a.panel)
