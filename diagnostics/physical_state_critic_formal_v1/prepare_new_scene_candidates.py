"""Freeze the original stochastic K16 proposals for fresh Four/Ring states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax.numpy as jnp
import numpy as np

from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
from diagnostics.orthoflow3_ring_revision_v1 import run_fresh as fresh
from shared_rollout_db.src.rollout_db import uid

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(scene, split):
    assert scene in ('four_way_intersection', 'ring_exchange')
    assert split in ('train', 'validation', 'test')
    source_path = HERE / f'states_{scene}.json'
    source = json.loads(source_path.read_text())
    states = [s for s in source['states'] if s['split'] == split]
    assert len(states) == {'train':512, 'validation':32, 'test':128}[split]
    target = HERE / 'loso' / scene / split / 'proposals.json'
    if target.exists():
        existing = json.loads(target.read_text())
        assert existing['states_sha256'] == sha(source_path)
        print(json.dumps(dict(scene=scene, split=split, proposals=len(existing['states']),
                              already_frozen=True)))
        return
    runtime = frozen.FrozenNewRuntime(scene)
    norm = json.loads(frozen.NORMALIZATION.read_text())
    generator, parameters, _, _ = fresh.load_models(norm)
    result = []
    for index, row in enumerate(states):
        content = uid('content', dict(physical=row['physical'], reset='empty monitor true-t0'))
        state_uid = uid('state', dict(scenario=runtime.scenario_uid, content=content))
        state = dict(uid=state_uid, alias=row['alias'], index=index,
                     content_hash=content, physical=row['physical'])
        flat, descriptor = runtime.conditioning(state)
        n = norm['scenarios'][scene]
        h = ((np.asarray(flat,np.float32)-np.asarray(n['h_mean'],np.float32)) /
             np.asarray(n['h_std'],np.float32))
        c = fresh.context_vector(descriptor,norm,scene)
        assert np.isfinite(h).all() and np.isfinite(c).all()
        raw = np.asarray(generator.apply(parameters,jnp.asarray(h[None]),jnp.asarray(c[None]),
                                         method=getattr(generator,frozen.METHOD[scene])))[0]
        proposal_seed = frozen.stable_int('generator-v1-proposals',41,scene,state_uid)
        noise = np.random.default_rng(proposal_seed).standard_normal((16,3))
        etas = np.asarray(frozen.eta_from_noise(jnp.asarray(raw)[None].repeat(16,0),
                                                jnp.asarray(noise,jnp.float32)),float)
        assert etas.shape == (16,3) and np.isfinite(etas).all()
        result.append(dict(scene=scene, split=split, state_uid=state_uid,
                           state_alias=row['alias'], physical_hash=row['physical_hash'],
                           proposal_seed=proposal_seed, eta=etas.tolist(),
                           generator_mean=np.asarray(frozen.eta_mean(jnp.asarray(raw)),float).tolist(),
                           controller_uid=runtime.controllers['orthoflow3']['uid'],
                           safety_controller_uid=runtime.controllers['hard_safety_q16']['uid'],
                           conditioning_hash=hashlib.sha256(np.asarray(flat,np.float32).tobytes()).hexdigest()))
        if (index+1)%32 == 0:
            print(json.dumps(dict(scene=scene,split=split,completed=index+1,
                                  total=len(states))),flush=True)
    payload = dict(schema='formal_q_original_stochastic_K16_v1',scene=scene,split=split,
                   states_sha256=sha(source_path),generator_sha256=sha(frozen.GENERATOR_CKPT),
                   generator_normalization_sha256=sha(frozen.NORMALIZATION),
                   proposal_rule='original SHA256(generator-v1-proposals|41|scene|state_uid); 16 stochastic samples, mean excluded',
                   no_test_labels_used=True,states=result)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(scene=scene,split=split,proposals=len(result),sha256=sha(target))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=('four_way_intersection','ring_exchange'),required=True)
    p.add_argument('--split',choices=('train','validation','test'),required=True)
    a=p.parse_args()
    main(a.scene,a.split)
