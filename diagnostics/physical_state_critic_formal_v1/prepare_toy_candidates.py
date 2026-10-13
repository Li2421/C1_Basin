"""Freeze the original Toy stochastic K16 generator proposals outcome-blind."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1 import freeze_proposals as prior
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import SCENARIO
from shared_rollout_db.src.rollout_db import uid, canonical

HERE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(split):
    assert split in ('train','validation','test')
    source=HERE/'states_toy_give_way.json'
    states=[s for s in json.loads(source.read_text())['states'] if s['split']==split]
    scenario_uid=SCENARIO
    assert len(states)=={'train':512,'validation':32,'test':128}[split]
    target=HERE/'loso'/'toy_give_way'/split/'proposals.json'
    if target.exists():
        assert json.loads(target.read_text())['states_sha256']==sha(source)
        print(json.dumps(dict(split=split,already_frozen=True,states=len(states))))
        return
    wide=json.loads((prior.WIDE/'frozen_benchmark_manifest.json').read_text())
    offset={'train':948700000,'validation':948800000,'test':948900000}[split]
    episodes=[dict(rollout_id=offset+i,initial_positions=s['physical']['positions'],
                   source_group=s['alias']) for i,s in enumerate(states)]
    h=prior.features_for_cohort(episodes,wide['environment'],wide['cbf'])
    selected=json.loads((prior.OUT/'generator_training.json').read_text())['selected']
    assert selected['seed']==23
    model=prior.genlib.Generator()
    template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214),jnp.float32),'Toy')
    parameters=serialization.from_bytes(template,Path(selected['checkpoint']).read_bytes())
    norm=json.loads((prior.OLD/'dataset_manifest.json').read_text())['state_normalization']['Toy']
    x=(h-np.asarray(norm['mean'],np.float32))/np.asarray(norm['std'],np.float32)
    raw=np.asarray(model.apply(parameters,jnp.asarray(x),'Toy'))
    mu,sigma=map(np.asarray,prior.genlib.dist_params(jnp.asarray(raw)))
    result=[]
    for i,(state,episode) in enumerate(zip(states,episodes)):
        noise_seed=int(hashlib.sha256(f'c1-generalization|generator-seed23|{episode["source_group"]}'.encode()).hexdigest()[:16],16)
        noise=np.random.default_rng(noise_seed).standard_normal((16,3))
        eta=prior.genlib.CENTER+prior.genlib.RADIUS*np.tanh(mu[i]+sigma[i]*noise)
        assert eta.shape==(16,3) and np.isfinite(eta).all()
        content=hashlib.sha256(canonical({'initial_positions':state['physical']['positions']}).encode()).hexdigest()
        result.append(dict(scene='toy_give_way',split=split,state_alias=state['alias'],
                           state_uid=uid('state',dict(scenario=scenario_uid,content=content)),
                           physical_hash=state['physical_hash'],rollout_id=episode['rollout_id'],
                           noise_seed=noise_seed,eta=eta.tolist(),
                           generator_mean=(prior.genlib.CENTER+prior.genlib.RADIUS*np.tanh(mu[i])).tolist(),
                           conditioning_sha256=hashlib.sha256(h[i].astype(np.float64).tobytes()).hexdigest()))
    payload=dict(schema='formal_q_original_toy_stochastic_K16_v1',split=split,
                 states_sha256=sha(source),generator_sha256=sha(selected['checkpoint']),
                 source_benchmark_sha256=sha(prior.WIDE/'frozen_benchmark_manifest.json'),
                 proposal_rule='original c1-generalization generator seed23; 16 stochastic samples, mean excluded',
                 test_labels_used=False,states=result)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(split=split,states=len(result),sha256=sha(target))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','validation','test'),required=True)
    a=p.parse_args();main(a.split)
