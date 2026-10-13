"""Frozen FF rollout collection for a genuinely held-out Q-selection audit.

All physical states, eta candidates, random seeds, checkpoints, and analysis
rules are frozen before any critic prediction or rollout is evaluated.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
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
sys.path.insert(0, str(REPO))
import numpy as np
from shared_rollout_db.src.rollout_db import canonical, uid, eta_identity, connect, transaction
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import ROOT as DBROOT
from diagnostics.state_nuisance_rootcause_v1.real_state_expansion import FIELD

PRIOR = HERE / 'real_state_expansion'
OUT = HERE / 'heldout_adaptive_selection_v1'
EXPERIMENT = 'exp_ring_heldout_adaptive_selection_20261007_v1'
STATE_SEED_START = 947400000
N_STATES = 256
FUTURE_ROOT = 2026100403
TRIALS = 16
ETA_COUNT = 16


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze(path, payload):
    if Path(path).exists():
        assert read(path) == payload, f'Frozen artifact changed: {path}'
    else:
        write(path, payload)


def seed_key(state, index):
    return canonical(dict(future_index=index, future_root=FUTURE_ROOT,
                          rng_namespace=state['rng_namespace']))


def case_identity(case):
    return dict(state=case['state_uid'], eta=case['eta_uid'],
                controller=case['controller_uid'], seed=case['seed_key'])


def prepare():
    """Draw the entire global sample and freeze it before opening any outcome."""
    from ring_exchange.environment import sample_instance, LocalFrameConfig

    prior = read(PRIOR / 'protocol.json')
    frozen = read(PRIOR / 'models_frozen.json')
    env = prior['environment']
    cfg = LocalFrameConfig(**env['config']['config'])
    assert sha(REPO / 'ring_exchange/environment.py') == prior['generator_sha256']
    assert prior['scenario'] == 'ring_exchange'
    assert len(prior['eta']) == ETA_COUNT
    assert FUTURE_ROOT == prior['future_root']
    for name in ('physical23_eta_n512', 'physical23_eta_n32'):
        selection = frozen['selections'][name]
        assert len(selection['models']) == 3
        for model in selection['models']:
            assert sha(model['path']) == model['sha256'], model['path']
    assert sha(PRIOR / 'physical_inputs.npz') == frozen['inputs_sha256']

    states = []
    for offset in range(N_STATES):
        draw = STATE_SEED_START + offset
        instance = sample_instance(draw, 'development', cfg)
        physical = dict(positions=instance.positions.tolist(),
                        velocities=instance.velocities.tolist(),
                        goals=instance.goals.tolist(), timestep=0)
        alias = f'heldout_adaptive_selection_ring_development_{draw}'
        content = uid('content', dict(physical=physical,
                                      reset='empty monitor true-t0'))
        state = dict(uid=alias, scenario='ring_exchange',
                     scenario_uid=env['scenario_uid'], physical=physical,
                     content_hash=content,
                     state_uid=uid('state', dict(scenario=env['scenario_uid'], content=content)),
                     rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8], 16),
                     family=f'heldout_adaptive_selection_global:{draw}',
                     metadata=dict(draw_seed=draw, split='development',
                                   outcome_blind=True, source='unchanged native global development generator'))
        states.append(state)
    assert len({s['state_uid'] for s in states}) == len(states) == N_STATES
    assert len({s['rng_namespace'] for s in states}) == N_STATES
    assert len({canonical(s['physical']) for s in states}) == N_STATES
    prior_uids = {s['state_uid'] for s in prior['states']}
    assert not prior_uids.intersection(s['state_uid'] for s in states)
    if (OUT / 'protocol.json').exists():
        assert read(OUT / 'protocol.json')['states'] == states
    else:
        with connect(True) as db:
            assert all(db.execute('SELECT 1 FROM state WHERE state_uid=?', (s['state_uid'],)).fetchone() is None
                       for s in states), 'Some proposed physical states already exist in the shared database'

    plan = dict(schema='heldout_adaptive_selection_v1',
                source='all global development draws, no critic/outcome-based filtering',
                physical_state_count=N_STATES,
                state_draw_seeds=list(range(STATE_SEED_START, STATE_SEED_START + N_STATES)),
                eta_count=ETA_COUNT, rollout_seed_indices=list(range(TRIALS)),
                discovery_seed_indices=list(range(8)),
                evaluation_seed_indices=list(range(8, 16)),
                adaptive_demand_primary_gap=3 / 8,
                adaptive_demand_sensitivity_gaps=[2 / 8, 4 / 8],
                strong_pair_reversal_gap=3 / 8,
                numerical_policy='Each attempted numerical failure stays unknown. Discovery uses success lower and upper bounds; evaluation reports bounds, never treats unknown as an observed failure.',
                tie_break='Smallest eta index for every exact tie',
                discovery_fixed_policy='Highest aggregate success lower bound over all 256 states and eight discovery seeds; ties use smallest eta index',
                qeta_policy='Frozen eta-only TRAIN512 empirical prior; argmax uses smallest eta index',
                adaptive_policy='Frozen physical23_eta_n512 VAL-selected checkpoint; three optimizer seeds 17,23,41 reported separately and averaged at state level',
                diagnostic_policy='Frozen physical23_eta_n32 VAL-selected checkpoint; diagnostic only',
                all_fixed_falsification='Compare all 16 fixed eta policies on the same evaluation half; hindsight best fixed is descriptive and a bootstrap max-over-16 contrast is also reported',
                bootstrap_unit='physical state', bootstrap_replicates=5000,
                primary_cohort='all 256 naturally sampled physical states',
                subset_policy='Grouping and eta preference are based only on discovery-half rollout labels; thresholds fixed before labels open',
                checkpoint_selection='exact prior frozen mean-three-seed validation NLL; no new fitting or TEST selection')
    freeze(OUT / 'analysis_plan.json', plan)
    protocol = dict(schema='heldout_adaptive_selection_v1', experiment_uid=EXPERIMENT,
                    scenario='ring_exchange', states=states, eta=prior['eta'],
                    environment=env, controller=prior['controller'],
                    future_root=FUTURE_ROOT, trials=TRIALS,
                    generator_sha256=sha(REPO / 'ring_exchange/environment.py'),
                    source_protocol_sha256=sha(PRIOR / 'protocol.json'),
                    frozen_models_sha256=sha(PRIOR / 'models_frozen.json'),
                    frozen_inputs_sha256=sha(PRIOR / 'physical_inputs.npz'),
                    analysis_plan_sha256=sha(OUT / 'analysis_plan.json'),
                    sampling='256 independent native development draws; seeds fixed before critic or rollout analysis',
                    video_required=False)
    freeze(OUT / 'protocol.json', protocol)

    with connect() as db, transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXPERIMENT, 'Held-out adaptive Q selection audit', str(OUT),
                    sha(OUT / 'protocol.json'), sha(__file__),
                    canonical(dict(states=N_STATES, eta=ETA_COUNT, seeds=TRIALS,
                                   global_generator='development', trained_models_frozen=True))))
        ctl = db.execute('SELECT * FROM controller_config WHERE controller_uid=?',
                         (protocol['controller']['controller_uid'],)).fetchone()
        assert ctl and ctl['config_json'] == canonical(protocol['controller']['payload'])
        assert ctl['compatibility_quality'] == 'EXACT_PROFILE'
        for state in states:
            db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                       (state['state_uid'], state['scenario_uid'], state['family'],
                        state['content_hash'], canonical(state['physical']),
                        canonical(env['config']),
                        canonical(dict(source_alias=state['uid'], metadata=state['metadata'])),
                        'CONTENT_EXACT'))
            db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)',
                       (state['scenario_uid'], state['uid'], state['state_uid'], EXPERIMENT))

    cases, requests = [], []
    for i, state in enumerate(states):
        for j, eta in enumerate(protocol['eta']):
            eta_uid = eta_identity(eta)[0]
            keys = [seed_key(state, k) for k in range(TRIALS)]
            requests.append(dict(state_uid=state['state_uid'], eta_uid=eta_uid,
                                 controller_uid=protocol['controller']['controller_uid'], seed_keys=keys))
            for k, key in enumerate(keys):
                case = dict(state_index=i, state_uid=state['state_uid'], eta_index=j,
                            eta=eta, eta_uid=eta_uid, controller_uid=protocol['controller']['controller_uid'],
                            seed=k, seed_key=key)
                case['rollout_uid'] = uid('roll', case_identity(case))
                cases.append(case)
    assert len(cases) == N_STATES * ETA_COUNT * TRIALS
    freeze(OUT / 'cases.json', cases)
    freeze(OUT / 'planned_rollouts.json', dict(requests=requests))
    with (OUT / 'planner_stdout.json').open('w') as f:
        subprocess.run([sys.executable, '-m', 'shared_rollout_db.plan',
                        '--manifest', str(OUT / 'planned_rollouts.json'),
                        '--output', str(OUT / 'cache_preflight.json')],
                       cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=True)
    missing, reused = [], []
    with connect(True) as db:
        for case in cases:
            row = db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',
                             tuple(case[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key'))).fetchone()
            if row is None:
                missing.append(case)
            else:
                assert not row['conflict_quarantined'] and row['compatibility_quality'] == 'EXACT_REUSE'
                reused.append(dict(case=case, record=dict(row)))
    freeze(OUT / 'reused_records.json', reused)
    freeze(OUT / 'missing_cases.json', missing)
    freeze(OUT / 'exact_identity_preflight.json',
           dict(requested=len(cases), reused=len(reused), genuinely_missing=len(missing),
                cached_numerical_unknown_not_retried=sum(r['record']['numerical_failure'] for r in reused),
                protocol_sha256=sha(OUT / 'protocol.json'),
                analysis_plan_sha256=sha(OUT / 'analysis_plan.json'),
                missing_cases_sha256=sha(OUT / 'missing_cases.json')))
    print(json.dumps(read(OUT / 'exact_identity_preflight.json')), flush=True)


def checked_runtime(protocol):
    # Import authoritative DB modules before adding the frozen runtime path.
    sys.path.insert(0, str(FIELD))
    from validation_stage2.run_rollouts import make_runtime
    from field_pipeline_v1.ring_t0_source import controller_uid
    import jax
    controller = protocol['controller']['payload']
    for relative, expected in controller['action_code_hashes'].items():
        assert sha(FIELD / relative) == expected, relative
    assert sha(controller['execution_callable']) == controller['execution_callable_sha256']
    assert sha(FIELD / 'source/ring_exchange/environment.py') == protocol['environment']['config']['environment_code_sha256']
    assert sha(controller['flow_checkpoint_path']) == controller['flow_checkpoint_sha256']
    runtime = make_runtime('ring_exchange')
    assert controller_uid() == protocol['controller']['controller_uid']
    assert not jax.config.jax_enable_x64 and runtime.steps == 10
    assert runtime.config.to_dict() == protocol['environment']['config']['config']
    return runtime


def run_one(runtime, case, protocol):
    from field_pipeline_v1.ring_t0_source import rollout as original_rollout
    state = protocol['states'][case['state_index']]
    assert state['state_uid'] == case['state_uid']
    request = dict(id=case['rollout_uid'], state=state, eta=case['eta'],
                   eta_index=case['eta_index'], seed=case['seed'],
                   phase='heldout_adaptive_selection_v1')
    raw = original_rollout(runtime, request)
    assert raw['controller_uid'] == case['controller_uid']
    assert raw['state_uid'] == state['uid']
    assert raw['success'] == (raw['termination'] == 'success' and not raw['collision'] and raw['error'] is None)
    return dict(schema='heldout_adaptive_selection_seed_v1', experiment_uid=EXPERIMENT,
                case=case, protocol_sha256=sha(OUT / 'protocol.json'), raw=raw,
                success=int(raw['success']), deadlock=int(raw['termination'] == 'deadlock'),
                timeout=int(raw['termination'] == 'timeout'), collision=int(raw['collision']),
                numerical_failure=int(raw['error'] is not None),
                episode_length=raw['steps'], outcome=raw['termination'])


def worker(worker_id, workers, chunk_size=192, limit=None):
    pre = read(OUT / 'exact_identity_preflight.json')
    assert pre['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert pre['analysis_plan_sha256'] == sha(OUT / 'analysis_plan.json')
    assert pre['missing_cases_sha256'] == sha(OUT / 'missing_cases.json')
    protocol = read(OUT / 'protocol.json')
    all_cases = read(OUT / 'missing_cases.json')
    assert len(all_cases) == pre['genuinely_missing']
    cases = all_cases[worker_id * chunk_size:(worker_id + 1) * chunk_size]
    if limit is not None:
        cases = cases[:limit]
    (OUT / 'raw').mkdir(exist_ok=True)
    runtime = None
    began = time.monotonic()
    pending = []
    new_count = recovered = 0
    for i, case in enumerate(cases):
        path = OUT / 'raw' / (case['rollout_uid'] + '.json')
        if path.exists():
            record = read(path)
            assert record['case'] == case and record['protocol_sha256'] == pre['protocol_sha256']
            recovered += 1
        else:
            if runtime is None:
                runtime = checked_runtime(protocol)
            record = run_one(runtime, case, protocol)
            write(path, record)
            new_count += 1
        pending.append(record)
        if len(pending) >= 16 or i + 1 == len(cases):
            append_journal(pending, EXPERIMENT, f'audit_{worker_id}')
            pending.clear()
            progress = dict(worker=worker_id, workers=workers, completed=i + 1,
                            total=len(cases), new=new_count, recovered=recovered,
                            elapsed_seconds=time.monotonic() - began,
                            limited_pilot=limit is not None, complete=i + 1 == len(cases))
            write(OUT / 'progress' / f'audit_{worker_id}.json', progress)
            print(json.dumps(progress), flush=True)


def merge():
    protocol = read(OUT / 'protocol.json')
    expected = sha(OUT / 'protocol.json')
    stats = collections.Counter()
    flags = ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure')
    with connect() as db:
        for path in sorted((DBROOT / 'journals' / EXPERIMENT).glob('*.jsonl')):
            source = uid('source', dict(path=str(path), sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?', (source,)).fetchone():
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                           (source, EXPERIMENT, str(path), sha(path), '.jsonl', 'SEED_EXACT', len(rows)))
                inserted = duplicates = 0
                for line, record in enumerate(rows, 1):
                    assert record['experiment_uid'] == EXPERIMENT and record['protocol_sha256'] == expected
                    case, raw = record['case'], record['raw']
                    assert raw['state_uid'] == protocol['states'][case['state_index']]['uid']
                    assert raw['controller_uid'] == protocol['controller']['controller_uid'] == case['controller_uid']
                    assert eta_identity(raw['eta'])[0] == case['eta_uid']
                    assert seed_key(protocol['states'][case['state_index']], raw['seed']) == case['seed_key']
                    assert case['rollout_uid'] == uid('roll', case_identity(case))
                    assert not (record['success'] and (record['collision'] or record['numerical_failure']))
                    core = tuple(record[k] for k in flags)
                    old = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (case['rollout_uid'],)).fetchone()
                    if old:
                        assert core == tuple(old[k] for k in flags)
                        assert record['episode_length'] == old['episode_length']
                        duplicates += 1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (case['rollout_uid'], case['state_uid'], case['eta_uid'], case['controller_uid'],
                                    case['seed_key'], case['seed_key'], *core, record['episode_length'],
                                    record['outcome'], EXPERIMENT, str(path), 'EXACT_REUSE', uid('record', record)))
                        inserted += 1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?,?)',
                               (case['rollout_uid'], source, line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?',
                           (inserted, duplicates, source))
                stats.update(inserted=inserted, duplicates=duplicates, journals=1)
        n = db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',
                       (EXPERIMENT,)).fetchone()[0]
        db.execute('UPDATE experiment SET new_rollout_count=?,reused_rollout_count=? WHERE experiment_uid=?',
                   (n, len(read(OUT / 'reused_records.json')), EXPERIMENT))
        db.commit()
    write(OUT / 'merge_audit.json', dict(stats, total_new_rows=n))
    print(json.dumps(dict(stats, total_new_rows=n)), flush=True)


def collect():
    protocol = read(OUT / 'protocol.json')
    cases = read(OUT / 'cases.json')
    values = np.full((N_STATES, ETA_COUNT, TRIALS, 3), -1, np.int8)
    seen = np.zeros((N_STATES, ETA_COUNT, TRIALS), bool)
    for item in read(OUT / 'reused_records.json'):
        case, record = item['case'], item['record']
        index = case['state_index'], case['eta_index'], case['seed']
        assert not seen[index]
        values[index] = [record['success'], record['numerical_failure'], record['collision']]
        seen[index] = True
    missing = []
    for case in read(OUT / 'missing_cases.json'):
        path = OUT / 'raw' / (case['rollout_uid'] + '.json')
        if not path.exists():
            missing.append(case['rollout_uid'])
            continue
        record = read(path)
        assert record['case'] == case and record['protocol_sha256'] == sha(OUT / 'protocol.json')
        index = case['state_index'], case['eta_index'], case['seed']
        assert not seen[index]
        values[index] = [record['success'], record['numerical_failure'], record['collision']]
        seen[index] = True
    if missing:
        write(OUT / 'collect_missing.json', dict(count=len(missing), first=missing[:20]))
        print(json.dumps(dict(missing=len(missing), present=int(seen.sum()))), flush=True)
        return False
    assert len(cases) == int(seen.sum()) == N_STATES * ETA_COUNT * TRIALS
    assert np.all((values[..., 0] == 1) | (values[..., 0] == 0))
    assert np.all((values[..., 1] == 1) | (values[..., 1] == 0))
    assert np.all((values[..., 2] == 1) | (values[..., 2] == 0))
    assert not np.any((values[..., 0] == 1) & (values[..., 1] == 1))
    valid = seen & (values[..., 1] == 0)
    success = ((values[..., 0] == 1) & valid).sum(axis=-1)
    failure = ((values[..., 0] == 0) & valid).sum(axis=-1)
    unknown = ((values[..., 1] == 1) & seen).sum(axis=-1)
    assert np.all(success + failure + unknown == TRIALS)
    np.savez_compressed(OUT / 'fresh_rollouts.npz', outcomes=values, present=seen,
                        success=success, failure=failure, numerical_unknown=unknown,
                        eta=np.asarray(protocol['eta']),
                        state_uids=np.asarray([s['state_uid'] for s in protocol['states']]),
                        state_draw_seeds=np.asarray([s['metadata']['draw_seed'] for s in protocol['states']]))
    with connect(True) as db:
        for case in cases:
            record = db.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',
                                tuple(case[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key'))).fetchone()
            assert record and not record['conflict_quarantined'] and record['compatibility_quality'] == 'EXACT_REUSE'
            i, j, k = case['state_index'], case['eta_index'], case['seed']
            assert values[i, j, k].tolist() == [record['success'], record['numerical_failure'], record['collision']]
    audit = dict(attempted=int(seen.sum()), successes=int(success.sum()), failures=int(failure.sum()),
                 numerical_unknown=int(unknown.sum()), collisions=int((values[..., 2] == 1).sum()),
                 all_cases_match_authoritative_DB=True,
                 dataset_sha256=sha(OUT / 'fresh_rollouts.npz'),
                 protocol_sha256=sha(OUT / 'protocol.json'),
                 analysis_plan_sha256=sha(OUT / 'analysis_plan.json'))
    write(OUT / 'collection_audit.json', audit)
    print(json.dumps(audit), flush=True)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'worker', 'merge', 'collect'))
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=192)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'worker':
        worker(args.worker, args.workers, args.chunk_size, args.limit)
    elif args.action == 'merge':
        merge()
    elif not collect():
        sys.exit(2)
