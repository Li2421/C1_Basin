"""Freeze original DB K16 generator proposals on new physical states."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','1')

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
from diagnostics.orthoflow3_ring_revision_v1 import run_fresh as fresh
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old
from shared_rollout_db.src.rollout_db import connect, uid

HERE=Path(__file__).resolve().parent
CURRENT_ROOT=948610000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(split):
    assert split in ('train','validation','test')
    source=HERE/'states_double_bottleneck.json'
    states=[s for s in json.loads(source.read_text())['states'] if s['split']==split]
    assert len(states)=={'train':512,'validation':32,'test':128}[split]
    target=HERE/'loso'/'double_bottleneck'/split/'proposals.json'
    if target.exists():
        assert json.loads(target.read_text())['states_sha256']==sha(source)
        print(json.dumps(dict(split=split,already_frozen=True,states=len(states))))
        return
    with connect(True) as db:
        scenario_uid=db.execute("SELECT scenario_uid FROM scenario WHERE name='DoubleBottleneck_4A'").fetchone()[0]
    dataset=old.FlowBC4ADataset(next(iter(frozen.DOUBLE_POOLS.values())),'all')
    policy,_=old.load_checkpoint(old.CHECKPOINT,dataset.environment_fingerprint)
    norm=json.loads(frozen.NORMALIZATION.read_text())
    generator,parameters,_,_=fresh.load_models(norm)
    name='double_bottleneck';n=norm['scenarios'][name]
    c=fresh.context_vector(dataset.config,norm,name)
    result=[]
    for i,row in enumerate(states):
        physical=row['physical']
        spec=row['metadata']['original_sampler_spec']
        episode=SimpleNamespace(initial_positions=np.asarray(physical['positions'],float),
                                initial_velocities=np.asarray(physical['velocities'],float),
                                regime=spec['regime'])
        env=old._initialize_env(old.Config(**dataset.config),episode)
        content=uid('content',dict(physical=physical,reset='empty monitor true-t0'))
        state_uid=uid('state',dict(scenario=scenario_uid,content=content))
        token=frozen.state_token(state_uid)
        current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(CURRENT_ROOT),token),0)
        raw_action=np.asarray(policy.sample_actions(env.observation()[None],current)[0],float)
        flat=np.r_[np.asarray(env.observation(),float).ravel(),raw_action.ravel()]
        h=(np.asarray(flat,np.float32)-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
        assert np.isfinite(h).all()
        raw=np.asarray(generator.apply(parameters,jnp.asarray(h[None]),jnp.asarray(c[None]),
                                       method=getattr(generator,frozen.METHOD[name])))[0]
        seed=frozen.stable_int('generator-v1-proposals',41,name,state_uid)
        noise=np.random.default_rng(seed).standard_normal((16,3))
        etas=np.asarray(frozen.eta_from_noise(jnp.asarray(raw)[None].repeat(16,0),
                                              jnp.asarray(noise,jnp.float32)),float)
        assert etas.shape==(16,3) and np.isfinite(etas).all()
        result.append(dict(scene=name,split=split,state_uid=state_uid,
                           state_alias=row['alias'],physical_hash=row['physical_hash'],
                           regime=spec['regime'],current_root=CURRENT_ROOT,
                           state_token=token,proposal_seed=seed,
                           conditioning_hash=hashlib.sha256(np.asarray(flat,np.float32).tobytes()).hexdigest(),
                           eta=etas.tolist(),
                           generator_mean=np.asarray(frozen.eta_mean(jnp.asarray(raw)),float).tolist()))
        if (i+1)%32==0:
            print(json.dumps(dict(split=split,completed=i+1,total=len(states))),flush=True)
    payload=dict(schema='formal_q_original_db_stochastic_K16_v1',split=split,
                 states_sha256=sha(source),generator_sha256=sha(frozen.GENERATOR_CKPT),
                 generator_normalization_sha256=sha(frozen.NORMALIZATION),
                 flow_checkpoint_sha256=sha(old.CHECKPOINT),
                 proposal_rule='original SHA256(generator-v1-proposals|41|scene|state_uid); 16 stochastic samples, mean excluded',
                 current_flow_rule='frozen Flow checkpoint with independent state-token key at true t0',
                 no_test_labels_used=True,states=result)
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(dict(split=split,states=len(result),sha256=sha(target))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=('train','validation','test'),required=True)
    a=p.parse_args();main(a.split)
