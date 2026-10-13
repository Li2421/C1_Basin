"""Freeze all remaining TEST selections before their outcome labels open."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.physical_state_critic_formal_v1 import physical_entities
from diagnostics.physical_state_critic_formal_v1.train_ring_ff import StateCritic,EtaCritic
from diagnostics.physical_state_critic_formal_v1.assemble_trainval import PANELS
from diagnostics.state_nuisance_rootcause_v1.adaptive_selection_predict_v1 import raw23

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def test_panel(panel):
    mode,sources=PANELS[panel]
    if mode=='matched':
        scene,chain=sources[0]
        protocol=read(HERE/'matched_protocol.json')
        states=[s for s in protocol['states'] if s['scenario']==scene and s['metadata']['split']=='test']
        assert len(states)==128
        ids=[s['state_uid'] for s in states]
        raw_eta=np.broadcast_to(np.asarray(protocol['eta'],np.float32),(128,16,3)).copy()
        norm=np.load(HERE.parent/'state_nuisance_rootcause_v1/real_state_expansion/physical_inputs.npz')
        eta=np.broadcast_to(norm['eta'],(128,16,3)).copy()
        if panel=='within_ring_TT':
            x=(raw23(states)-norm['physical_center'])/norm['physical_scale']
            return ids,raw_eta,eta,jnp.asarray(x),True
        rows=[dict(scene=scene,physical=s['physical']) for s in states]
    else:
        heldout=(sources[0][0] if panel.startswith('within_') else
                 {'loso_heldout_toy':'toy_give_way','loso_heldout_db':'double_bottleneck',
                  'loso_heldout_four':'four_way_intersection','loso_heldout_ring':'ring_exchange'}[panel])
        proposals=read(HERE/'loso'/heldout/'test'/'proposals.json')['states']
        states={s['alias']:s for s in read(HERE/f'states_{heldout}.json')['states'] if s['split']=='test'}
        assert len(proposals)==len(states)==128
        ids=[r['state_uid'] for r in proposals]
        raw_eta=np.asarray([r['eta'] for r in proposals],np.float32)
        eta=(raw_eta-np.array([.875,0.,.375],np.float32))/np.array([.375,.5,.375],np.float32)
        rows=[dict(scene=heldout,physical=states[r['state_alias']]['physical']) for r in proposals]
    x={k:jnp.asarray(v) for k,v in physical_entities.batch(rows).items()}
    return ids,raw_eta,eta,x,False


def freeze_all():
    panels=list(PANELS)
    models={p:read(HERE/'fits'/p/'models_frozen.json') for p in panels}
    for panel,m in models.items():
        assert m['test_labels_used'] is False
        assert m['trainval_sha256']==sha(HERE/'fits'/panel/'trainval.npz')
        assert m['plan_sha256']==sha(HERE/'unified_train_plan.json')
    output=dict(schema='formal_q_all_fits_frozen_before_test_v1',test_labels_used=False,
                plan_sha256=sha(HERE/'unified_train_plan.json'),
                panels={p:sha(HERE/'fits'/p/'models_frozen.json') for p in panels},
                ring_ff_models_sha256=sha(HERE/'ring_ff_models_frozen.json'),
                ring_ff_predictions_sha256=sha(HERE/'ring_ff_test_predictions.npz'))
    target=HERE/'all_models_frozen.json'
    if target.exists():assert read(target)==output
    else:target.write_text(json.dumps(output,indent=2)+'\n')
    loso=dict(schema='formal_q_loso_folds_frozen_v1',test_labels_used=False,
              folds={p:output['panels'][p] for p in panels if p.startswith('loso_')},
              all_models_sha256=sha(target))
    q=HERE/'loso_models_frozen.json'
    if q.exists():assert read(q)==loso
    else:q.write_text(json.dumps(loso,indent=2)+'\n')
    print(json.dumps(dict(panels=len(panels),all_models_sha256=sha(target))))


def predict(panel):
    assert panel in PANELS
    global_freeze=read(HERE/'all_models_frozen.json')
    assert global_freeze['test_labels_used'] is False
    m=read(HERE/'fits'/panel/'models_frozen.json')
    assert global_freeze['panels'][panel]==sha(HERE/'fits'/panel/'models_frozen.json')
    ids,raw_eta,eta,x,ring=test_panel(panel)
    assert len(ids)==128 and len(set(ids))==128
    assert eta.shape==(128,16,3) and np.isfinite(eta).all()
    e=jnp.asarray(eta)
    if ring:
        gather=lambda ii:x[ii]
    else:
        gather=lambda ii:{k:v[ii] for k,v in x.items()}
    scores={};hashes={}
    for kind in ('state','eta_only'):
        info=m[kind]['selected']
        ck=HERE/'fits'/panel/kind/f"seed{info['seed']}"/'checkpoint.msgpack'
        assert sha(ck)==info['checkpoint_sha256']
        net=(EtaCritic() if kind=='eta_only' else StateCritic() if ring else rep.Critic())
        template=(net.init(jax.random.PRNGKey(info['seed']),e[:1,:1].reshape(1,3))
                  if kind=='eta_only' else
                  net.init(jax.random.PRNGKey(info['seed']),gather(jnp.array([0])),e[:1,:1].reshape(1,3)))
        params=serialization.from_bytes(template,ck.read_bytes())
        values=[]
        all_i=np.repeat(np.arange(128),16)
        all_j=np.tile(np.arange(16),128)
        for start in range(0,len(all_i),256):
            ii=jnp.asarray(all_i[start:start+256]);jj=jnp.asarray(all_j[start:start+256])
            v=(net.apply(params,e[ii,jj]) if kind=='eta_only' else
               net.apply(params,gather(ii),e[ii,jj]))
            values.append(np.asarray(v))
        scores[kind]=np.concatenate(values).reshape(128,16)
        assert np.isfinite(scores[kind]).all()
        hashes[kind]=sha(ck)
    dest=HERE/'fits'/panel/'test_predictions.npz'
    np.savez_compressed(dest,state_logits=scores['state'],eta_only_logits=scores['eta_only'],
                        state_choice=np.argmax(scores['state'],axis=1),
                        eta_only_choice=np.argmax(scores['eta_only'],axis=1),
                        state_uids=np.asarray(ids),eta=raw_eta)
    audit=dict(panel=panel,states=128,checkpoint_hashes=hashes,
               model_freeze_sha256=sha(HERE/'fits'/panel/'models_frozen.json'),
               all_models_sha256=sha(HERE/'all_models_frozen.json'),
               test_labels_used=False,predictions_sha256=sha(dest),
               selected_eta_indices={kind:np.bincount(np.argmax(scores[kind],axis=1),minlength=16).tolist()
                                     for kind in ('state','eta_only')})
    (HERE/'fits'/panel/'prediction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


def bundle_loso():
    folds=('loso_heldout_toy','loso_heldout_db','loso_heldout_four','loso_heldout_ring')
    picks=[];ids=[]
    for name in folds:
        audit=read(HERE/'fits'/name/'prediction_audit.json')
        file=HERE/'fits'/name/'test_predictions.npz'
        assert audit['predictions_sha256']==sha(file)
        d=np.load(file)
        picks.append(np.stack((d['state_choice'],d['eta_only_choice']),axis=-1))
        ids.append(np.asarray(d['state_uids']))
    target=HERE/'loso_test_predictions.npz'
    np.savez_compressed(target,choices=np.stack(picks),state_uids=np.stack(ids),folds=np.asarray(folds))
    audit=dict(folds=list(folds),states_per_fold=128,
               test_labels_used=False,prediction_sha256=sha(target),
               loso_models_sha256=sha(HERE/'loso_models_frozen.json'))
    (HERE/'loso_test_prediction_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze_all','predict','bundle_loso'))
    p.add_argument('--panel',choices=tuple(PANELS))
    a=p.parse_args()
    if a.action=='freeze_all':freeze_all()
    elif a.action=='bundle_loso':bundle_loso()
    else:
        assert a.panel
        predict(a.panel)
