"""Phase A: frozen-generator proposal-aligned Q evidence and critic closure.

This isolated driver never rewrites canonical models, normalization, or v2.
Each seed execution uses the existing runtime and persistent exact cache.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as learn
from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
from diagnostics.orthoflow3_ring_revision_v1 import run_fresh as fresh
from new_benchmark_common import basin_dataset_v1 as bd
from new_benchmark_common.safety_eta3 import DatabaseSink, request, preflight
from shared_rollout_db.src.rollout_db import connect, canonical, eta_identity, uid

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'datasets/orthoflow3_basin_dataset_v2_audited'
SCENARIOS = learn.SCENARIOS
SEEDS = tuple(range(16))
K = 16


def decode(value):
    # v2 preserved some native v1 JSON inside another JSON string.
    for _ in range(6):
        if not isinstance(value, str):
            return value
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    raise ValueError('Unexpected conditioning serialization depth')


def load(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def states():
    return pq.read_table(DATA / 'states.parquet').to_pylist()


def inputs(row, norm):
    n = norm['scenarios'][row['scenario']]
    flat = np.asarray(decode(row['conditioning'])['flat'], np.float32)
    h = (flat - np.asarray(n['h_mean'], np.float32)) / np.asarray(n['h_std'], np.float32)
    c = fresh.context_vector(decode(row['environment_descriptor']), norm, row['scenario'])
    assert np.isfinite(h).all() and np.isfinite(c).all()
    return h, c


def register(runtime):
    runtime.output = OUT / 'phase_a/evidence' / runtime.name
    runtime.output.mkdir(parents=True, exist_ok=True)
    runtime.experiment_uid = uid('exp', {'path': str(OUT), 'phase': 'A'})
    with connect() as con:
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,metadata_json) VALUES(?,?,?,?,?,?)',
                    (runtime.experiment_uid, 'k16_critic_canonicalization_phase_a', str(OUT),
                     digest(load(OUT / 'protocol.json')), sha(__file__),
                     canonical({'population': 'audited_v2_train_dev', 'K': K, 'full_Q16': True})))
        con.commit()


def runtimes(scenario=None):
    # Source runtime constructors are reused; output/provenance are isolated.
    bd.OUT = OUT / 'phase_a/evidence'
    bd.WORK = bd.OUT
    rows = states()
    expected = {r['state_uid']: r for r in rows}
    for sc in SCENARIOS:
        if scenario and sc != scenario:
            continue
        if sc == 'double_bottleneck':
            groups = defaultdict(list)
            metadata = bd._double_source_metadata()
            with connect(True) as con:
                scenario_uid = con.execute("SELECT scenario_uid FROM scenario WHERE name='DoubleBottleneck_4A'").fetchone()[0]
            for v2 in rows:
                if v2['scenario'] != sc: continue
                row = {'uid':v2['state_uid'],'alias':v2['state_id'],'conditioning':decode(v2['conditioning']),
                       'controller_uid':v2['controller_uid'],'scenario_uid':scenario_uid,
                       'metadata':metadata[v2['state_id']]}
                groups[row['controller_uid']].append(row)
            for controller, rr in sorted(groups.items()):
                r = bd.DoubleTrainingRuntime(rr, controller)
                register(r)
                yield sc, r
        else:
            source = load(ROOT / 'datasets/orthoflow3_basin_dataset_v1/work' / sc / 'sampled_state_manifest.json')
            rr = [r for r in source['states'] if r['uid'] in expected]
            r = bd.TrainingRuntime(sc, rr)
            register(r)
            for s in rr:
                assert expected[s['uid']]['controller_uid'] == r.controllers['orthoflow3']['uid'], (sc, 'controller mismatch')
            yield sc, r


def freeze():
    if (OUT / 'phase_a/proposals.json').exists():
        raise RuntimeError('Proposal manifest already frozen; use existing evidence')
    protocol = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'sequential_phases': ['A_critic_only_then_freeze_report', 'B_conditioning_then_generator_first_retraining'],
        'K_stochastic': K, 'proposal_set': 'same deterministic mean +16 stochastic samples as frozen K16',
        'no_mean_only_deployment': True, 'no_eta0_or_fixed_eta_in_selection': True,
        'proposal_seed': 'SHA256(generator-v1-proposals|41|scenario|state_uid)',
        'Q_protocol': {'seeds': list(SEEDS), 'robust': '>=15 successes /16', 'full_Q_required': True,
                       'numerical_retry': 'initial execution + at most3 identical retries; no imputation'},
        'phase_a': {'generator_frozen': True, 'normalization_frozen': True,
                    'critic_architecture': 'unchanged', 'critic_loss': 'canonical empirical-Q binomial BCE',
                    'sampling': 'equal scenarios; uniform state; 80% uniform frozen proposal,20% v2 eta',
                    'optimizer': 'AdamW lr1e-3 wd1e-4 clip5;4000steps;96states/scenario;seeds17,23,41',
                    'selection': 'train/dev only; Ring gap/exploitation improve, DB/Four loss<=5pp',
                    'fresh_confirmation_counts': {'double_bottleneck':24, 'four_way_intersection':24, 'ring_exchange':60}},
        'no_test_derived_training': True,
        'superseded_objectives_not_repeated': ['ranking_aware_v1', 'LCB_uncertainty'],
        'phase_b_not_started': True,
    }
    dump('protocol.json', protocol)
    paths = [frozen.GENERATOR_CKPT, frozen.CRITIC_CKPT, frozen.NORMALIZATION,
             DATA/'states.parquet', DATA/'eta_labels.parquet', DATA/'manifest.json']
    for directory in ['shared_control', 'four_way_intersection', 'ring_exchange', 'double_bottleneck', 'toy_giveway', 'new_benchmark_common']:
        paths.extend(sorted((ROOT / directory).rglob('*.py')))
    dump('canonical_hashes_before.json', {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in paths})
    dump('git_status_before.json', {'status': subprocess.check_output(['git','status','--short'], cwd=ROOT,text=True)})
    norm = load(frozen.NORMALIZATION)
    gm,gp,cm,cp = fresh.load_models(norm)
    result = []
    for row in states():
        sc = row['scenario']; h,c = inputs(row,norm)
        raw = gm.apply(gp,jnp.asarray(h[None]),jnp.asarray(c[None]),method=getattr(gm,learn.METHOD[sc]))[0]
        rng = np.random.default_rng(learn.stable_int('generator-v1-proposals',41,sc,row['state_uid']))
        samples = np.asarray(learn.eta_from_noise(raw[None].repeat(K,0),jnp.asarray(rng.standard_normal((K,3)),jnp.float32)),float)
        mean = np.asarray(learn.eta_mean(raw),float)
        etas = np.asarray([mean,*samples])
        logits = np.asarray(cm.apply(cp,jnp.asarray(h[None].repeat(K+1,0)),jnp.asarray(c[None].repeat(K+1,0)),
                                     jnp.asarray((etas-learn.CENTER)/learn.RADIUS),method=getattr(cm,learn.METHOD[sc])))
        result.append({'scenario':sc,'state_uid':row['state_uid'],'state_id':row['state_id'],'split':row['split'],
                       'parent_episode_id':row['parent_episode_id'],'etas':etas.tolist(),'old_scores':(1/(1+np.exp(-logits))).tolist(),
                       'old_index':int(np.argmax(logits)),'proposal_hash':digest(etas.tolist()),
                       'conditioning_hash':digest(decode(row['conditioning']))})
    dump('phase_a/proposals.json', {'frozen_before_outcomes':True,'generator_sha':sha(frozen.GENERATOR_CKPT),
                                   'normalization_sha':sha(frozen.NORMALIZATION),'states':result})
    return {'states':len(result),'proposal_tuples':len(result)*(K+1),'requested_Q16_seed_slots':len(result)*(K+1)*16}


def jobs(runtime, manifest):
    by = {r['state_uid']:r for r in manifest['states']}
    result = []
    for state in runtime.states:
        row = by[state['uid']]
        # B0 is evaluated separately for rescue/break; it is NOT a proposal.
        for index,eta in enumerate(row['etas']+[[0.,0.,0.]]):
            result.append({'state':state,'eta':eta,'chain':'orthoflow3','seeds':list(SEEDS),
                           'metadata':{'proposal_index':index if index<=K else None,'B0_reference':index>K,
                                       'proposal_set_hash':row['proposal_hash'],'split':row['split'],
                                       'task':'phase_a_critic_alignment','training_population':True}})
    return result


def evidence(runtime,job):
    chain=job.get('chain','orthoflow3')
    found,missing = bd.cached_rows(runtime,job['state'],job['eta'],chain,SEEDS)
    numerical = {}
    if missing:
        with connect(True) as con:
            rr = con.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND numerical_failure=1',
                (job['state']['uid'],eta_identity(job['eta'])[0],runtime.controllers[chain]['uid'])).fetchall()
        for r in rr:
            seed = json.loads(r['seed_key']).get('future_index')
            if seed in missing: numerical[seed] = dict(r)
    return found,numerical,[s for s in missing if s not in numerical]


def cache_preflight():
    manifest=load(OUT/'phase_a/proposals.json'); totals=Counter(); details=[]
    for sc,runtime in runtimes():
        jj=jobs(runtime,manifest)
        name=f"phase_a/cache/requests_{sc}_{runtime.controllers['orthoflow3']['uid'][-8:]}.json"
        dump(name,{'requests':[request(runtime,j['state'],j['eta'],'orthoflow3',SEEDS) for j in jj]})
        pf=preflight(OUT/name);dump(name.replace('requests_','standard_preflight_'),pf)
        for j in jj:
            found,numerical,missing=evidence(runtime,j)
            totals['requested']+=16;totals['reused']+=len(found);totals['numerical_uncertified']+=len(numerical);totals['missing']+=len(missing)
            details.append({'scenario':sc,'state_uid':j['state']['uid'],'eta_uid':eta_identity(j['eta'])[0],
                            'proposal_index':j['metadata']['proposal_index'],'missing':missing,'numerical':list(numerical)})
    value={'summary':dict(totals),'details':details,'database':str(ROOT/'shared_rollout_db/rollout.sqlite')}
    if not (OUT/'phase_a/initial_cache_preflight.json').exists():dump('phase_a/initial_cache_preflight.json',value)
    dump('phase_a/cache_preflight.json',value)
    return dict(totals)


def run(shard,shards):
    manifest=load(OUT/'phase_a/proposals.json');totals=Counter(); global_index=0
    for sc,runtime in runtimes():
        selected=[]
        for j in jobs(runtime,manifest):
            if global_index%shards==shard:selected.append(j)
            global_index+=1
        if not selected:continue
        tag=f'aligned_{shard}of{shards}_{runtime.controllers["orthoflow3"]["uid"][-8:]}'
        name=f'phase_a/cache/{sc}_{tag}.json'
        dump(name,{'requests':[request(runtime,j['state'],j['eta'],'orthoflow3',SEEDS) for j in selected]})
        dump(name.replace('.json','_preflight.json'),preflight(OUT/name))
        sink=DatabaseSink(runtime,tag)
        try:
            for j in selected:
                found,numerical,missing=evidence(runtime,j)
                totals['requested']+=16;totals['reused']+=len(found);totals['existing_numerical']+=len(numerical)
                for seed in missing:
                    for attempt in range(4):
                        row=runtime.rollout(j['state'],np.asarray(j['eta'],float),seed,'orthoflow3')
                        row.update({**j['metadata'],'execution_attempt':attempt,'stage':tag})
                        sink.insert(row);totals['physical_attempts']+=1
                        if not row['numerical_failure']:break
                    totals['completed_seeds']+=1
                    totals['numerical_unresolved']+=bool(row['numerical_failure'])
                    if totals['physical_attempts']%25==0:print(json.dumps({'shard':shard,**totals}),flush=True)
        finally:sink.finalize()
    dump(f'phase_a/run_shard{shard}of{shards}.json',dict(totals));return dict(totals)


def collect():
    manifest=load(OUT/'phase_a/proposals.json');rows=[];missing=[]
    for sc,runtime in runtimes():
        for j in jobs(runtime,manifest):
            found,numerical,absent=evidence(runtime,j)
            if absent:missing.append({'state_uid':j['state']['uid'],'eta':j['eta'],'seeds':absent})
            success=sum(bool(r['success']) for r in found.values());fail=len(found)-success
            rows.append({'scenario':sc,'state_uid':j['state']['uid'],'controller_uid':runtime.controllers['orthoflow3']['uid'],
                         'eta':j['eta'],'eta_uid':eta_identity(j['eta'])[0],**j['metadata'],
                         'successes':success,'failures':fail,'valid_seeds':len(found),'numerical_seeds':list(numerical),
                         'Q16':success/16 if len(found)==16 else None,'Q_lower':success/16,'Q_upper':(16-fail)/16,
                         'robust':True if success>=15 else False if fail>=2 else None,
                         'rollout_uids':[r['rollout_uid'] for r in found.values()],
                         'collisions':sum(bool(r['collision']) for r in found.values()),
                         'terminal_summary':dict(Counter(r['outcome'] for r in found.values()))})
    dump('phase_a/proposal_Q_evidence.json',{'complete':not missing,'rows':rows,'missing':missing})
    return {'eta_tuples':len(rows),'missing_tuples':len(missing),'complete_Q16':sum(r['Q16'] is not None for r in rows)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','preflight','run','collect']);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    result=freeze() if a.stage=='freeze' else cache_preflight() if a.stage=='preflight' else run(a.shard,a.shards) if a.stage=='run' else collect()
    print(json.dumps(result,indent=2),flush=True)
