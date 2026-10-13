"""Reuse sealed confirmation rollouts for source-scene-only frozen fits.

Gradients, normalization and checkpoint/variant choice are source-scene-only.
The representation family was investigated with both DEVELOPMENT scenes, so
this is fresh-state confirmation of domain transfer, not pristine domain-blind
four-scene LOSO. It uses no new simulation and never changes the main model.
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from .experiment import HERE, read, write, freeze, sha
from .predict import Arrays
from .train import Critic, inputs, SEEDS
from .factorial import execution_constants

OUT = HERE/'confirmation_scene_transfer'
MODELS = ('eta_only', 'state_only', 'state_contact_bilinear')
SOURCES = {'toy':'toy_give_way', 'ring':'ring_exchange'}


def link(target, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_symlink():
        assert not destination.exists()
        destination.symlink_to(target, target_is_directory=target.suffix != '.json')


def prepare():
    assert not (HERE/'evaluation_complete.json').exists()
    assert not (OUT/'frozen.json').exists()
    original = np.load(HERE/'confirmation_predictions.npz')
    entities, nominal, field = [], [], []
    for sid, ctl in zip(original['state_uid'], original['controller']):
        f = np.load(HERE/'features'/str(sid)/(str(ctl)+'.npz'))
        entities.append({k:f[k] for k in f.files if k.startswith('entity_')})
        nominal.append(np.pad(f['nominal'],((0,4-len(f['nominal'])),(0,0))))
        field.append(np.pad(f['field'],((0,4-len(f['field'])),(0,0))))
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    batched = rep.batch([{k[7:]:v for k,v in row.items()} for row in entities])
    base = Arrays({f'entity_{k}':v for k,v in batched.items()})
    base.update(nominal=np.asarray(nominal),field=np.asarray(field),
                eta=np.asarray(read(HERE/'protocol.json')['eta'],np.float32))
    for k in ('state_uid','scene','controller'):base[k]=original[k]
    all_records = []
    for source, scene in SOURCES.items():
        variants = [(HERE, f'factorial_{source}'),
                    (HERE, f'factorial_{source}_constants'),
                    (HERE/'checks/factorial_breadth8',f'factorial_{source}')]
        scores = []
        for root, variant in variants:
            score = np.mean([read(root/'fits'/variant/'state_contact_bilinear'/f'seed{s}'/'complete.json')['selected']['score'] for s in SEEDS])
            scores.append((float(score),root,variant))
        score, root, variant = min(scores,key=lambda v:v[0])
        folder = OUT/source
        folder.mkdir(parents=True,exist_ok=True)
        d = Arrays({k:v.copy() for k,v in base.items()})
        if variant.endswith('_constants'):
            for sc in set(d['scene']):d['entity_globals'][d['scene']==sc]=execution_constants(str(sc))
        result = {k:original[k] for k in ('state_uid','scene','controller','controller_swap','state_permutations')}
        swap = result['controller_swap']; n=len(swap); records=[]
        for kind in MODELS:
            predictions=[]; wrong=[]
            for seed in SEEDS:
                path=root/'fits'/variant/kind/f'seed{seed}'
                record=read(path/'complete.json')
                assert record['checkpoint_sha256']==sha(path/'checkpoint.msgpack')
                record.update(path=str(path)); records.append(record)
                x,c,e=inputs(d,read(path/'normalization.json'));net=Critic(kind)
                init=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
                params=serialization.from_bytes(init,(path/'checkpoint.msgpack').read_bytes())
                @jax.jit
                def infer(ii,jj,cc):return net.apply(params,{k:v[ii] for k,v in x.items()},c[cc],e[jj])
                def predict(mapping):
                    ii=np.repeat(np.arange(n),16);jj=np.tile(np.arange(16),n);ci=np.repeat(mapping,16);zs=[]
                    for start in range(0,len(ii),256):
                        zs.append(np.asarray(infer(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]),jnp.asarray(ci[start:start+256]))))
                    return np.asarray(jax.nn.sigmoid(jnp.asarray(np.concatenate(zs).reshape(n,16))))
                predictions.append(predict(np.arange(n)));wrong.append(predict(swap))
            result[kind]=np.asarray(predictions);result[kind+'_controller_mismatch']=np.asarray(wrong)
            if kind in ('eta_only','state_only'):np.testing.assert_array_equal(result[kind],result[kind+'_controller_mismatch'])
        target=folder/'confirmation_predictions.npz';assert not target.exists();np.savez_compressed(target,**result)
        freeze(folder/'models_frozen.json',dict(records=records,primary='state_contact_bilinear',
            data_variant=variant,train_limit=128,source_scene=scene,
            scope='Fresh-state and fresh-controller confirmation of source-scene-only gradient/normalization/checkpoint/variant fit. Architecture-family development previously used both scenes; not domain-blind four-scene LOSO.'))
        freeze(folder/'confirmation_predictions_frozen.json',dict(predictions_sha256=sha(target),
            main_prediction_hash=sha(HERE/'confirmation_predictions.npz'),final_labels_opened=False,
            generated_after_rollout_started=True,rollout_outcomes_inspected=False,
            selection='Minimum mean3seed source-only validation score among original4C,restored-constants4C,and equal-budget8C,for the frozen common bilinear architecture'))
        link(root/'data'/variant,folder/'data'/variant)
        link(HERE/'data/final',folder/'data/final')
        link(HERE/'batches/safety_confirmation',folder/'batches/safety_confirmation')
        all_records.append(dict(source=source,source_scene=scene,selected_root=str(root),selected_variant=variant,
            score=score,candidates=[dict(score=s,root=str(r),variant=v) for s,r,v in scores],
            prediction_hash=sha(target),additional_rollouts=0))
    freeze(OUT/'frozen.json',dict(folds=all_records,no_confirmation_labels_inspected=True,
        code_sha256=sha(__file__),main_confirmation_unchanged=True,
        limitation='Held-out scene development labels informed representation-family investigation. Fresh confirmation states/controllers remained untouched. This does not restore the stronger never-seen-scene-label methodology.'))
    print('Both source-scene-only confirmation predictions frozen; no labels opened')


def evaluate(source):
    from . import evaluate as evaluation
    evaluation.HERE=OUT/source
    evaluation.KINDS=MODELS
    evaluation.run()
    path=evaluation.HERE/'evaluation_complete.json';meta=read(path)
    meta.update(source_scene=SOURCES[source],
        scope='New physical states and unseen analytic parameters; same-source versus held-out-scene gradient training. Source-only normalization/checkpoint/variant selection; representation family investigated using both development scenes. NOT pristine domain-blind LOSO.')
    write(path,meta)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','evaluate']);p.add_argument('--source',choices=list(SOURCES));a=p.parse_args()
    prepare() if a.action=='prepare' else evaluate(a.source)
