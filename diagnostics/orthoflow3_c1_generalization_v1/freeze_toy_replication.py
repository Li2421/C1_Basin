"""Outcome-blind 48-state wide-IC Toy replication with frozen K16 proposals.

The sample distribution, environment and controller protocol match the old
wide-IC test. New source groups and proposal RNG namespace are disjoint.
Only freezes states/proposals and writes planned cache requests; no rollout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist

from offline_baselines import ROOT, OUT, dump_json
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1 import freeze_proposals as prior
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1 import prepare_cache_plan as cacheplan
from shared_rollout_db.src.rollout_db import uid, canonical, eta_identity
from train_toy_eta_only import EtaOnly


COHORT_SEED=2026100201
STATE_N=48
GENERATOR_SEED=23
FROZEN=OUT/'toy_replication'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    FROZEN.mkdir(exist_ok=True)
    manifest_path=FROZEN/'frozen_proposals.json'
    if manifest_path.exists():
        raise RuntimeError('Toy replication already frozen; no resampling or refreezing')
    wide=json.loads((prior.WIDE/'frozen_benchmark_manifest.json').read_text())
    rng=np.random.default_rng(COHORT_SEED)
    episodes=[]
    for i in range(STATE_N):
        xabs=rng.uniform(.55,1.05,size=2)
        y=rng.uniform(-.025,.025,size=2)
        positions=[[-float(xabs[0]),float(y[0])],[float(xabs[1]),float(y[1])]]
        episodes.append({'episode_index':i,'rollout_id':20000+i,'initial_positions':positions,
                         'source_group':f'c1_generalization_wide_{COHORT_SEED}_{i:03d}'})
    h=prior.features_for_cohort(episodes,wide['environment'],wide['cbf'])
    np.savez_compressed(FROZEN/'cohort_features.npz',h_raw=h)
    selected=json.loads((prior.OUT/'generator_training.json').read_text())['selected']
    assert selected['seed']==GENERATOR_SEED
    gen=prior.genlib.Generator()
    p0=gen.init(jax.random.PRNGKey(0),jnp.zeros((1,214),jnp.float32),'Toy')
    gp=serialization.from_bytes(p0,Path(selected['checkpoint']).read_bytes())
    norm=json.loads((prior.OLD/'dataset_manifest.json').read_text())
    hn=norm['state_normalization']['Toy']
    x=(h-np.asarray(hn['mean'],np.float32))/np.asarray(hn['std'],np.float32)
    raw=np.asarray(gen.apply(gp,jnp.asarray(x),'Toy'))
    mu,sigma=map(np.asarray,prior.genlib.dist_params(jnp.asarray(raw)))
    center=prior.genlib.CENTER;radius=prior.genlib.RADIUS
    means=center+radius*np.tanh(mu)
    samples=[]
    for i,ep in enumerate(episodes):
        seed=int(hashlib.sha256(f'c1-generalization|generator-seed{GENERATOR_SEED}|{ep["source_group"]}'.encode()).hexdigest()[:16],16)
        noise=np.random.default_rng(seed).standard_normal((16,3))
        samples.append(center+radius*np.tanh(mu[i]+sigma[i]*noise))
    samples=np.asarray(samples)
    critic_logits,critics=prior.critic_scores(h,samples)
    # All eta-only controls were fit/selected before this cohort was drawn.
    frozen_kernel=np.load(OUT/'toy_eta_kernel_model.npz')
    z=(samples-frozen_kernel['center'])/frozen_kernel['scale']
    k=np.exp(-cdist(z.reshape(-1,3),frozen_kernel['ztrain'],'sqeuclidean')/
             (2*float(frozen_kernel['bw'])**2))
    kernel_score=np.clip(float(frozen_kernel['prior'])+k@frozen_kernel['alpha'],0,1).reshape(STATE_N,16)
    etamodel=EtaOnly()
    eparams=serialization.from_bytes(etamodel.init(jax.random.PRNGKey(0),jnp.zeros((1,3))),
              (OUT/'toy_eta_only_mlp/seed23/checkpoint.msgpack').read_bytes())
    mlp_logits=np.asarray(etamodel.apply(eparams,jnp.asarray(z.reshape(-1,3),jnp.float32))).reshape(STATE_N,16)
    fixed=np.asarray([.7421875,.46875,.7265625],float)
    rows=[]
    for i,ep in enumerate(episodes):
        etas={f'sample_{j}':samples[i,j].tolist() for j in range(16)}
        etas.update({'generator_mean':means[i].tolist(),'fixed_common':fixed.tolist()})
        rows.append({**ep,'h_sha256':hashlib.sha256(h[i].astype(np.float64).tobytes()).hexdigest(),
                     'eta':etas,'sigma':sigma[i].tolist(),
                     'critic_scores':critic_logits[i].tolist(),
                     'critic_choice':int(np.argmax(critic_logits[i])),
                     'eta_only_kernel_scores':kernel_score[i].tolist(),
                     'eta_only_kernel_choice':int(np.argmax(kernel_score[i])),
                     'eta_only_mlp_scores':mlp_logits[i].tolist(),
                     'eta_only_mlp_choice':int(np.argmax(mlp_logits[i]))})
    payload={'cohort_rule':'48 new independent initial states; x_abs~U(0.55,1.05), y~U(-0.025,0.025) per agent; no rejection or outcome selection',
             'cohort_seed':COHORT_SEED,'selection_before_outcomes':True,
             'generator_checkpoint':selected['checkpoint'],'generator_sha256':sha(selected['checkpoint']),
             'critic_checkpoints':critics,
             'eta_only_kernel_sha256':sha(OUT/'toy_eta_kernel_model.npz'),
             'eta_only_mlp_sha256':sha(OUT/'toy_eta_only_mlp/seed23/checkpoint.msgpack'),
             'proposal_seed_rule':'SHA256(c1-generalization|generator-seed23|source_group)',
             'controller_environment_sha256':sha(prior.WIDE/'frozen_benchmark_manifest.json'),
             'states':rows}
    manifest_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    # Precompute exact DB keys and four staged, non-overlapping continuation
    # batches. Each rollout batch is capped at 4096 truly new continuations.
    mapping=[];request_batches={f'proposal_batch{i}':[] for i in range(3)}
    request_batches['baselines']=[]
    for row in rows:
        physical={'initial_positions':row['initial_positions']}
        content_hash=hashlib.sha256(canonical(physical).encode()).hexdigest()
        state_uid=uid('state',{'scenario':cacheplan.SCENARIO,'content':content_hash})
        row['state_uid']=state_uid
        for kind,eta in row['eta'].items():
            eta_uid,_,_=eta_identity(eta);controller_uid=uid('ctl',cacheplan.CORRECTION_CONFIG)
            mapping.append({'episode_index':row['episode_index'],'kind':kind,'state_uid':state_uid,
                            'eta_uid':eta_uid,'controller_uid':controller_uid})
            batch=f'proposal_batch{row["episode_index"]//16}' if kind.startswith('sample_') else 'baselines'
            request_batches[batch].append({'state_uid':state_uid,'eta_uid':eta_uid,'controller_uid':controller_uid,
                'seed_keys':[canonical({'future_index':j}) for j in range(16)]})
        safety_eta_uid,_,_=eta_identity([0.,0.,0.])
        safety_ctl=uid('ctl',cacheplan.CORRECTION_CONFIG)
        mac_ctl=uid('ctl',cacheplan.MAC_CONFIG)
        for kind,cid in [('safety',safety_ctl),('mac_only',mac_ctl)]:
            mapping.append({'episode_index':row['episode_index'],'kind':kind,'state_uid':state_uid,
                            'eta_uid':safety_eta_uid,'controller_uid':cid})
            request_batches['baselines'].append({'state_uid':state_uid,'eta_uid':safety_eta_uid,'controller_uid':cid,
                'seed_keys':[canonical({'future_index':j}) for j in range(16)]})
    # Update now that exact state_uid has been resolved; this is still before
    # any outcome query or rollout, and immediately freeze it.
    manifest_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    with (FROZEN/'candidate_cache_keys.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    for name,requests in request_batches.items():
        assert len(requests)*16<=4096
        (FROZEN/f'planned_{name}.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    dump_json('toy_replication_plan.json',{'cohort':str(manifest_path),'cohort_sha256':sha(manifest_path),
             'states':STATE_N,'batch_requests':{k:len(v)*16 for k,v in request_batches.items()},
             'total_requested_continuations':sum(len(v)*16 for v in request_batches.values()),
             'cache_preflight_required_before_rollout':True,'new_rollout_so_far':0})
    print(json.dumps({'states':STATE_N,'batch_continuations':{k:len(v)*16 for k,v in request_batches.items()}},indent=2))


if __name__=='__main__':main()
