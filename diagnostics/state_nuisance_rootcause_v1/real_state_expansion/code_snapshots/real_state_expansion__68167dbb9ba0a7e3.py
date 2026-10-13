"""Real, controller-input-free Ring Q(s, eta) data expansion.

The label-generating execution callable is frozen. Workers only run truly
missing seeds and write immutable evidence; the explicit-identity merger is
single-writer (the legacy generic importer cannot preserve these modern IDs).
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE.parent / 'orthoflow3_tt_ff_matched_learnability_v1'
FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
OUT = HERE / 'real_state_expansion'
EXPERIMENT = 'exp_ring_physical_state_expansion_20261007_v1'
FUTURE_ROOT = 2026100403
sys.path.insert(0, str(REPO))
import numpy as np
from shared_rollout_db.src.rollout_db import canonical, uid, eta_identity, connect, transaction
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import ROOT as DBROOT


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(p)


def freeze(p, x):
    if Path(p).exists():
        assert read(p) == x, ('Frozen artifact differs', str(p))
    else:
        write(p, x)


def seed_key(s, k):
    return canonical(dict(future_index=k, future_root=FUTURE_ROOT, rng_namespace=s['rng_namespace']))


def case_identity(c):
    return dict(state=c['state_uid'], eta=c['eta_uid'], controller=c['controller_uid'], seed=c['seed_key'])


def prepare():
    from ring_exchange.environment import sample_instance, LocalFrameConfig
    old = read(SOURCE / 'protocol.json')
    env = old['environments']['ring_exchange']
    cfg = LocalFrameConfig(**env['config']['config'])
    assert sha(REPO / 'ring_exchange/environment.py') == env['config']['environment_code_sha256']
    states = []
    totals = dict(train=512, validation=32, test=64)
    for sp, total in totals.items():
        prior = [dict(s, expansion_origin='original') for s in old['states']
                 if s['scenario'] == 'ring_exchange' and s['metadata']['split'] == sp]
        chosen = prior.copy()
        for j in range(total - len(prior)):
            draw = {'train': 946100000, 'validation': 946200000, 'test': 946300000}[sp] + j
            ins = sample_instance(draw, 'development', cfg)
            physical = dict(positions=ins.positions.tolist(), velocities=ins.velocities.tolist(),
                            goals=ins.goals.tolist(), timestep=0)
            alias = f'physical_expansion_v1_ring_{sp}_{draw}'
            content = uid('content', dict(physical=physical, reset='empty monitor true-t0'))
            chosen.append(dict(uid=alias, scenario='ring_exchange', scenario_uid=env['scenario_uid'],
                physical=physical, content_hash=content,
                state_uid=uid('state', dict(scenario=env['scenario_uid'], content=content)),
                rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8], 16),
                family=f'physical_expansion_v1:{draw}', expansion_origin='fresh',
                metadata=dict(split=sp, draw_seed=draw, outcome_blind=True,
                              source='fresh unchanged native development IC generator')))
        states.extend(chosen)
    assert len({s['state_uid'] for s in states}) == len(states) == 608
    assert len({s['rng_namespace'] for s in states}) == len(states)
    assert len({canonical(s['physical']) for s in states}) == len(states)
    protocol = dict(schema='real_physical_state_expansion_v1', experiment_uid=EXPERIMENT,
        scenario='ring_exchange', states=states, eta=old['eta'], environment=env,
        controller=old['controllers']['ring_exchange']['FF'],
        total_states=totals, train_prefixes=[32, 64, 128, 256, 512],
        labels='FF own valid Bernoulli trials consistently at all sizes, numerical failures unknown; no TT intersection',
        trials=dict(train=4, validation=16, test=16), future_root=FUTURE_ROOT,
        predictor_inputs=['physical local92 plus eta3', 'physical minimal23 plus eta3',
                          'physical minimal23 with16 fixed-eta heads'],
        excluded_inputs=['controller ID', 'controller weights', 'nominal flow', 'bounded flow',
                         'H20 trajectories', 'response fingerprints', 'state identifiers', 'random seed'],
        fixed_input_normalization='original32 TRAIN states only, unchanged across training sizes',
        selection='mean3-seed validation NLL, same32 validation states at every training size',
        training=dict(seeds=[17, 23, 41], batch_size=128, lr=.0008, weight_decay=.0001,
                      clip=5., hidden=[64, 64], max_steps=12000, checkpoints_every=200,
                      matched_budget_checkpoint=2000),
        controls=['TRAIN eta-only empirical Bernoulli prior', '512-state shuffled state-label assignment'],
        primary_metrics=['eta-centered state-effect correlation and R2', 'same-eta state-change sign accuracy',
                         'Bernoulli NLL', 'top1 empirical success and B15'],
        inference='physical-state bootstrap; model seeds are repeated fits, not independent physical evidence',
        test_policy='original16 previously opened diagnostic; fresh48 untouched until all model/checkpoint choices frozen',
        adequacy='512 is a16x independent-state expansion for a learning-curve test, not an assertion of asymptotic sufficiency',
        source_protocol_sha256=sha(SOURCE / 'protocol.json'), generator_sha256=sha(REPO / 'ring_exchange/environment.py'),
        video_policy='AGENTS.md full four-scene gate is required for batch-complete; missing real deadlock evidence stays a gap')
    freeze(OUT / 'protocol.json', protocol)
    with connect() as db, transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
            (EXPERIMENT, 'Ring physical state data expansion', str(OUT), sha(OUT / 'protocol.json'), sha(__file__),
             canonical(dict(controller_inputs=False, fresh_train=480, fresh_val=24, fresh_test=48))))
        ctl = db.execute('SELECT * FROM controller_config WHERE controller_uid=?', (protocol['controller']['controller_uid'],)).fetchone()
        assert ctl['config_json'] == canonical(protocol['controller']['payload'])
        assert ctl['compatibility_quality'] == 'EXACT_PROFILE'
        for s in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                (s['state_uid'], s['scenario_uid'], s['family'], s['content_hash'], canonical(s['physical']),
                 canonical(env['config']), canonical(dict(source_alias=s['uid'], metadata=s['metadata'])), 'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)', (s['scenario_uid'], s['uid'], s['state_uid'], EXPERIMENT))
    cases, requests = [], []
    for i, s in enumerate(states):
        sp = s['metadata']['split']
        for j, e in enumerate(protocol['eta']):
            eid = eta_identity(e)[0]
            keys = [seed_key(s, k) for k in range(protocol['trials'][sp])]
            requests.append(dict(state_uid=s['state_uid'], eta_uid=eid,
                                 controller_uid=protocol['controller']['controller_uid'], seed_keys=keys))
            for k, sk in enumerate(keys):
                c = dict(state_index=i, state_uid=s['state_uid'], eta_index=j, eta=e, eta_uid=eid,
                         controller_uid=protocol['controller']['controller_uid'], seed=k, seed_key=sk, split=sp)
                c['rollout_uid'] = uid('roll', case_identity(c))
                cases.append(c)
    freeze(OUT / 'cases.json', cases)
    freeze(OUT / 'planned_rollouts.json', dict(requests=requests))
    with (OUT / 'planner_stdout.json').open('w') as f:
        subprocess.run([sys.executable, '-m', 'shared_rollout_db.plan', '--manifest', str(OUT / 'planned_rollouts.json'),
                        '--output', str(OUT / 'cache_preflight.json')], cwd=REPO, stdout=f, check=True)
    missing, reused = [], []
    with connect(True) as db:
        for c in cases:
            r = db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',
                           tuple(c[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key'))).fetchone()
            if r is None:
                missing.append(c)
            else:
                assert not r['conflict_quarantined'] and r['compatibility_quality'] == 'EXACT_REUSE'
                reused.append(dict(case=c, record=dict(r)))
    freeze(OUT / 'reused_records.json', reused)
    # The shared legacy planner excludes numerical rows from valid reuse. They
    # are already attempted unknowns, never genuinely missing or retried here.
    for stage in ('source', 'test'):
        mm = [c for c in missing if (c['split'] == 'test') == (stage == 'test')]
        mm.sort(key=lambda c: (0 if c['split'] == 'validation' else 1, c['state_index'], c['eta_index'], c['seed']))
        freeze(OUT / f'missing_{stage}.json', mm)
    freeze(OUT / 'exact_identity_preflight.json', dict(requested=len(cases), reused=len(reused), missing=len(missing),
        cached_numerical_unknown_not_retried=sum(x['record']['numerical_failure'] for x in reused),
        protocol_sha256=sha(OUT / 'protocol.json'), cases_sha256=sha(OUT / 'cases.json'),
        missing_source_sha256=sha(OUT / 'missing_source.json'), missing_test_sha256=sha(OUT / 'missing_test.json')))
    print(json.dumps(read(OUT / 'exact_identity_preflight.json')), flush=True)


def runtime_checked():
    # Import the authoritative DB modules before the frozen runtime adds its
    # source snapshot to sys.path; never accidentally write to its copied DB.
    sys.path.insert(0, str(FIELD))
    from validation_stage2.run_rollouts import make_runtime
    from field_pipeline_v1.ring_t0_source import controller_uid
    import jax
    p = read(OUT / 'protocol.json')
    cp = p['controller']['payload']
    for rel, expected in cp['action_code_hashes'].items():
        assert sha(FIELD / rel) == expected, rel
    assert sha(cp['execution_callable']) == cp['execution_callable_sha256']
    assert sha(FIELD / 'source/ring_exchange/environment.py') == p['environment']['config']['environment_code_sha256']
    assert sha(cp['flow_checkpoint_path']) == cp['flow_checkpoint_sha256']
    rt = make_runtime('ring_exchange')
    assert controller_uid() == p['controller']['controller_uid']
    assert not jax.config.jax_enable_x64 and rt.steps == 10
    assert rt.config.to_dict() == p['environment']['config']['config']
    return rt


def run_one(rt, c, p):
    from field_pipeline_v1.ring_t0_source import rollout as original_rollout
    state = p['states'][c['state_index']]
    assert state['state_uid'] == c['state_uid']
    case = dict(id=c['rollout_uid'], state=state, eta=c['eta'], eta_index=c['eta_index'],
                seed=c['seed'], phase='real_physical_state_expansion_v1')
    capture = c['seed'] == 0 and c['eta_index'] in (0, 8, 13)
    traces = {}
    original_make = rt.make_env
    if capture:
        def observed_make(s):
            env = original_make(s)
            traces['positions'] = [env.positions.copy()]
            traces['clearance'] = []
            traces['actions'] = []
            traces['initial'] = env.augmented_state()
            original_step = env.step
            def observed_step(a):
                result = original_step(a)
                info = result[3]
                traces['positions'].append(env.positions.copy())
                traces['actions'].append(np.asarray(a).copy())
                traces['clearance'].append(min(float(np.min(info['swept_obstacle_distance'])) - env.config.collision_margin,
                    float(np.min(info['swept_outer_distance'])) - env.config.collision_margin,
                    float(np.min(info['swept_agent_distances'])) - env.config.agent_collision_margin))
                return result
            env.step = observed_step
            return env
        rt.make_env = observed_make
    try:
        raw = original_rollout(rt, case)
    finally:
        rt.make_env = original_make
    assert raw['controller_uid'] == c['controller_uid'] and raw['state_uid'] == state['uid']
    assert raw['success'] == (raw['termination'] == 'success' and not raw['collision'] and raw['error'] is None)
    record = dict(schema='explicit_ring_physical_expansion_seed_v1', experiment_uid=EXPERIMENT, case=c,
                  protocol_sha256=sha(OUT / 'protocol.json'), raw=raw,
                  success=int(raw['success']), deadlock=int(raw['termination'] == 'deadlock'),
                  timeout=int(raw['termination'] == 'timeout'), collision=int(raw['collision']),
                  numerical_failure=int(raw['error'] is not None), episode_length=raw['steps'], outcome=raw['termination'])
    if capture and traces['clearance']:
        def native(x):
            if isinstance(x, np.ndarray): return x.tolist()
            if isinstance(x, np.generic): return x.item()
            raise TypeError(type(x))
        initial = json.loads(json.dumps(traces['initial'], default=native))
        meta = dict(batch_id=EXPERIMENT, scenario='ring_exchange', rollout_id=c['rollout_uid'],
            controller=c['controller_uid'], checkpoint=p['controller']['payload']['flow_checkpoint_path'] + ':' + p['controller']['payload']['flow_checkpoint_sha256'],
            seed=c['seed_key'], source_record=str(OUT / 'raw' / (c['rollout_uid'] + '.json')),
            initial_state_sha256=hashlib.sha256(canonical(initial).encode()).hexdigest(),
            config=rt.config.to_dict(), initial_augmented_state=initial, start_step=0,
            episode_steps=len(traces['clearance']), complete=True, termination=raw['termination'],
            collision=bool(raw['collision']), safety_enabled=True,
            deadlock_detected=False, deadlock_criterion=None,
            note='Native Ring protocol has no deadlock event; timeout is never relabelled.')
        folder = OUT / 'traces'; folder.mkdir(exist_ok=True)
        path = folder / (c['rollout_uid'] + '.npz')
        np.savez_compressed(path, positions=np.asarray(traces['positions']), steps=np.arange(len(traces['positions'])),
            goals=np.asarray(state['physical']['goals']), walls=np.empty((0, 2, 2)), actions=np.asarray(traces['actions']),
            swept_clearance=np.asarray(traces['clearance']), metadata_json=np.array(json.dumps(meta, allow_nan=False)))
        record['trace_path'] = str(path)
    return record


def worker(stage, worker_id, workers, limit=None, chunk_size=None):
    pre = read(OUT / 'exact_identity_preflight.json')
    assert pre['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert pre['missing_' + stage + '_sha256'] == sha(OUT / f'missing_{stage}.json')
    if stage == 'test':
        assert (OUT / 'models_frozen.json').exists(), 'Fresh TEST before model/checkpoint freeze'
    p = read(OUT / 'protocol.json')
    all_cases = read(OUT / f'missing_{stage}.json')
    cases = (all_cases[worker_id * chunk_size:(worker_id + 1) * chunk_size] if chunk_size
             else [c for i, c in enumerate(all_cases) if i % workers == worker_id])
    if limit is not None: cases = cases[:limit]
    (OUT / 'raw').mkdir(exist_ok=True)
    rt = None
    started = time.monotonic()
    buffer, new_count, recovered = [], 0, 0
    for i, c in enumerate(cases):
        path = OUT / 'raw' / (c['rollout_uid'] + '.json')
        if path.exists():
            record = read(path)
            assert record['case'] == c and record['protocol_sha256'] == pre['protocol_sha256']
            recovered += 1
        else:
            if rt is None: rt = runtime_checked()
            record = run_one(rt, c, p)
            write(path, record)
            new_count += 1
        buffer.append(record)
        if len(buffer) >= 16 or i + 1 == len(cases):
            append_journal(buffer, EXPERIMENT, f'{stage}_{worker_id}')
            buffer.clear()
            progress = dict(stage=stage, worker=worker_id, workers=workers, completed=i+1, total=len(cases),
                            new=new_count, recovered=recovered, elapsed_seconds=time.monotonic()-started,
                            limited_pilot=limit is not None, complete=i+1 == len(cases))
            write(OUT / 'progress' / f'{stage}_{worker_id}.json', progress)
            print(json.dumps(progress), flush=True)


def merge():
    p = read(OUT / 'protocol.json')
    expected = sha(OUT / 'protocol.json')
    stats = collections.Counter()
    flags = ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure')
    with connect() as db:
        for path in sorted((DBROOT / 'journals' / EXPERIMENT).glob('*.jsonl')):
            source = uid('source', dict(path=str(path), sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?', (source,)).fetchone(): continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                    (source, EXPERIMENT, str(path), sha(path), '.jsonl', 'SEED_EXACT', len(rows)))
                inserted = duplicates = 0
                for line, r in enumerate(rows, 1):
                    assert r['experiment_uid'] == EXPERIMENT and r['protocol_sha256'] == expected
                    c, raw = r['case'], r['raw']
                    assert raw['state_uid'] == p['states'][c['state_index']]['uid']
                    assert raw['controller_uid'] == p['controller']['controller_uid'] == c['controller_uid']
                    assert eta_identity(raw['eta'])[0] == c['eta_uid']
                    assert seed_key(p['states'][c['state_index']], raw['seed']) == c['seed_key']
                    assert c['rollout_uid'] == uid('roll', case_identity(c))
                    assert not (r['success'] and (r['collision'] or r['numerical_failure']))
                    core = tuple(r[k] for k in flags)
                    old = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
                    if old:
                        assert core == tuple(old[k] for k in flags) and r['episode_length'] == old['episode_length'], 'Conflicting exact evidence'
                        duplicates += 1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (c['rollout_uid'], c['state_uid'], c['eta_uid'], c['controller_uid'], c['seed_key'], c['seed_key'],
                             *core, r['episode_length'], r['outcome'], EXPERIMENT, str(path), 'EXACT_REUSE', uid('record', r)))
                        inserted += 1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)', (c['rollout_uid'], source, line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?', (inserted, duplicates, source))
                stats['inserted'] += inserted; stats['duplicates'] += duplicates; stats['journals'] += 1
        n = db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?', (EXPERIMENT,)).fetchone()[0]
        db.execute('UPDATE experiment SET new_rollout_count=?,reused_rollout_count=? WHERE experiment_uid=?',
                   (n, len(read(OUT / 'reused_records.json')), EXPERIMENT)); db.commit()
    write(OUT / 'merge_audits' / f'{time.time_ns()}.json', dict(stats, total_new_rows=n))
    print(json.dumps(dict(stats, total_new_rows=n)), flush=True)


def collect(stage='source'):
    p = read(OUT / 'protocol.json')
    if stage == 'test': assert (OUT / 'models_frozen.json').exists()
    values = np.full((len(p['states']), 16, 16, 3), -1, np.int8)
    present = np.zeros(values.shape[:-1], bool)
    for item in read(OUT / 'reused_records.json'):
        c, r = item['case'], item['record']
        if c['split'] == 'test' and stage != 'test': continue
        index = c['state_index'], c['eta_index'], c['seed']
        values[index] = [r['success'], r['numerical_failure'], r['collision']]
        present[index] = True
    missing = []
    for c in read(OUT / 'missing_source.json') + (read(OUT / 'missing_test.json') if stage == 'test' else []):
        path = OUT / 'raw' / (c['rollout_uid'] + '.json')
        if not path.exists():
            missing.append(c['rollout_uid']); continue
        r = read(path)
        assert r['case'] == c and r['protocol_sha256'] == sha(OUT / 'protocol.json')
        index = c['state_index'], c['eta_index'], c['seed']
        assert not present[index]
        values[index] = [r['success'], r['numerical_failure'], r['collision']]
        present[index] = True
    if missing:
        print(json.dumps(dict(stage=stage, missing=len(missing), present=int(present.sum())))); return False
    valid = present & (values[..., 1] == 0)
    success = ((values[..., 0] == 1) & valid).sum(-1)
    failure = ((values[..., 0] == 0) & valid).sum(-1)
    numerical = ((values[..., 1] == 1) & present).sum(-1)
    np.savez_compressed(OUT / (stage + '_dataset.npz'), outcomes=values, present=present,
        success=success, failure=failure, numerical=numerical,
        split=np.array([s['metadata']['split'] for s in p['states']]),
        origin=np.array([s['expansion_origin'] for s in p['states']]))
    report = dict(stage=stage, requested_present=int(present.sum()), numerical=int(numerical.sum()),
        successes=int(success.sum()), failures=int(failure.sum()), collisions=int(((values[..., 2] == 1) & present).sum()),
        pure_physical_predictors=True, controller_fingerprint=p['controller']['controller_uid'],
        data_sha256=sha(OUT / (stage + '_dataset.npz')), protocol_sha256=sha(OUT / 'protocol.json'))
    if stage=='test':
        counters=collections.Counter()
        with connect(True) as db:
            for c in read(OUT/'cases.json'):
                r=db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',
                    tuple(c[k] for k in ('state_uid','eta_uid','controller_uid','seed_key'))).fetchone()
                assert r and not r['conflict_quarantined'] and r['compatibility_quality']=='EXACT_REUSE'
                i,j,k=c['state_index'],c['eta_index'],c['seed']
                assert list(values[i,j,k])==[r['success'],r['numerical_failure'],r['collision']]
                counters['exact_present']+=1;counters['numerical_unknown']+=r['numerical_failure']
        write(OUT/'final_cache_identity_audit.json',dict(counters,genuinely_missing=0,all_labels_match_authoritative_database=True))
    write(OUT / (stage + '_dataset_audit.json'), report)
    print(json.dumps(report), flush=True)
    return True


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('prepare', 'worker', 'merge', 'collect'))
    ap.add_argument('--stage', choices=('source', 'test'), default='source')
    ap.add_argument('--worker', type=int, default=0)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--chunk-size', type=int)
    args = ap.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'worker': worker(args.stage, args.worker, args.workers, args.limit, args.chunk_size)
    elif args.action == 'merge': merge()
    elif not collect(args.stage): sys.exit(2)
