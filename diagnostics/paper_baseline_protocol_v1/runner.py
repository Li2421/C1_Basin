"""Official preflight/journal/merger workflow and immutable numerical evidence."""
from core import *
import os
import sys
import time

_VERIFIED_FILES={}
_ACCELERATION_INSTALLED=False


def verify_execution(request):
    source=request['controller_payload']['execution_sources']
    manifest=Path(source['manifest']).resolve()
    assert manifest.is_relative_to(HERE/'execution_revisions')
    assert sha(manifest)==source['freeze_sha256'],'HARD_STOP: execution manifest changed'
    frozen=read(manifest)
    for relative,expected in frozen['files'].items():
        path=HERE/relative
        stamp=path.stat()
        signature=(stamp.st_mtime_ns,stamp.st_size,expected)
        if _VERIFIED_FILES.get(str(path))!=signature:
            assert sha(path)==expected,('HARD_STOP: numerical source changed',relative)
            _VERIFIED_FILES[str(path)]=signature


def register(cases, stage):
    assert os.environ.get('STB_COORDINATOR') == '1'
    unique = {}
    for request in cases:
        uid = request['rollout_uid']
        if uid in unique:
            assert request['controller_payload'] == unique[uid]['controller_payload']
            assert request['state_uid'] == unique[uid]['state_uid']
        else:
            unique[uid] = request
    cases = list(unique.values())
    with db.connect() as connection, db.transaction(connection):
        connection.execute('INSERT INTO experiment(experiment_uid,name,path,metadata_json) VALUES(?,?,?,?) ON CONFLICT(experiment_uid) DO NOTHING',
                           (EXP, EXP, str(HERE), db.canonical(dict(scope='paired_paper_baseline_protocol_v1', no_training=True))))
        for request in cases:
            physical = {k: request['state'][k] for k in ('positions', 'velocities', 'goals')}
            rows = {
                'scenario': dict(scenario_uid=request['scenario_uid'], name=request['scene'], code_config_fingerprint=request['scenario_uid'], metadata_json=db.canonical(request['scenario_payload'])),
                'state': dict(state_uid=request['state_uid'], scenario_uid=request['scenario_uid'],
                              source_group=f"{request['benchmark']}:{request['scene']}:state{request['state_index']}:solo{request['solo']}",
                              content_hash=digest(physical), physical_state_json=db.canonical(physical),
                              goals_geometry_json=db.canonical(request['scenario_payload']),
                              provenance_json=db.canonical(request['state_provenance']), identity_quality='CONTENT_EXACT'),
                'controller_config': dict(controller_uid=request['controller_uid'], scenario_uid=request['scenario_uid'],
                                          config_json=db.canonical(request['controller_payload']), compatibility_quality='EXACT_PROFILE')}
            for table, values in rows.items():
                primary = next(iter(values))
                # Human aliases and provenance are not execution identity. A
                # duplicate physical state may legitimately have another index.
                compared = {key: value for key, value in values.items()
                            if key != 'provenance_json' and not (table == 'state' and key == 'source_group')}
                existing = connection.execute(f'SELECT * FROM {table} WHERE {primary}=?', (values[primary],)).fetchone()
                if existing:
                    assert all(existing[key] == value for key, value in compared.items()), ('DB_IDENTITY_CONFLICT', table, values[primary])
                else:
                    connection.execute(f'INSERT INTO {table} (' + ','.join(values) + ') VALUES (' + ','.join('?' for _ in values) + ')', list(values.values()))
                saved = connection.execute(f'SELECT * FROM {table} WHERE {primary}=?', (values[primary],)).fetchone()
                assert saved and all(saved[key] == value for key, value in compared.items())
            eta_uid, eta, canonical_hex = db.eta_identity(request['eta_a'])
            assert eta_uid == request['eta_uid']
            connection.execute('INSERT INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?) ON CONFLICT(eta_uid) DO NOTHING',
                               (eta_uid, *eta, canonical_hex))
            assert connection.execute('SELECT canonical_hex FROM eta WHERE eta_uid=?', (eta_uid,)).fetchone()['canonical_hex'] == canonical_hex
    plan = dict(requests=[{k: r[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key')} for r in cases], exact_configs=cases)
    path = HERE / 'plans' / (stage + '_planned.json')
    if path.exists():
        assert digest(read(path)) == digest(plan), ('IMMUTABLE_STAGE_PLAN', stage)
    else:
        dump(path, plan)
    preflight = planner.preflight(path)
    dump(HERE / 'plans' / (stage + '_preflight.json'), preflight)
    assert not preflight['summary']['ambiguous'], ('DB_AMBIGUOUS', stage)
    missing = {(row['state_uid'], row['eta_uid'], row['controller_uid'], seed) for row in preflight['details'] for seed in row['missing_seeds']}
    done = {}
    with db.connect(True) as connection:
        for request in cases:
            uid = request['rollout_uid']
            local = HERE / 'raw' / (uid + '.json')
            if local.exists():
                done[uid] = read(local)
            existing = connection.execute('SELECT * FROM rollout WHERE rollout_uid=?', (uid,)).fetchone()
            if existing:
                assert not existing['conflict_quarantined'], ('QUARANTINED_RESULT', uid)
                if uid not in done:
                    source = connection.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf ON rs.source_uid=sf.source_uid WHERE rs.rollout_uid=? LIMIT 1', (uid,)).fetchone()
                    assert source, ('No readable exact source', uid)
                    with Path(source['path']).open() as stream:
                        done[uid] = next(json.loads(line) for i, line in enumerate(stream, 1) if i == source['source_line'])
                assert int(done[uid]['success']) == existing['success']
            elif uid not in done:
                assert tuple(request[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key')) in missing
    from historical_reuse import lookup
    done.update(lookup([r for r in cases if r['rollout_uid']not in done]))
    return cases, [r for r in cases if r['rollout_uid'] not in done], done


def execute_numerical(request):
    verify_execution(request)
    engine = request['controller_payload']['engine']
    global _ACCELERATION_INSTALLED
    enabled=bool(request['controller_payload'].get('exact_acceleration'))
    if engine=='paper_own_algorithm':
        assert not enabled and not _ACCELERATION_INSTALLED,'Papers require a fresh unpatched interpreter'
    elif enabled and not _ACCELERATION_INSTALLED:
        from acceleration import enable
        enable(check=False)
        _ACCELERATION_INSTALLED=True
    elif not enabled and _ACCELERATION_INSTALLED:
        raise RuntimeError('Original execution requested in an accelerated worker; use separate original/fast stages')
    if engine == 'paper_own_algorithm':
        from execution_policy import assert_method_enabled
        assert_method_enabled(request['controller_payload']['paper_method'])
        from paper_episode import run
    elif engine == 'heterogeneous_plain_original':
        import learned_setup
        learned_setup.install()
        run = RN.run
    elif engine == 'original_admission_dictionary':
        import runtime
        if not getattr(runtime, '_recording_encoding_installed', False):
            from recording_fix import install
            install()
            runtime._recording_encoding_installed = True
        from runtime import run
    elif engine in ('copied_goal_rhr', 'copied_nominal_only'):
        from historical_replay import run
    else:
        raise ValueError(('Unknown frozen engine', engine))
    return run(request)


def work(request):
    if (HERE / 'PROTECTION_VIOLATION.jsonl').exists():
        raise RuntimeError('HARD_STOP: protection violation')
    start = time.monotonic()
    result, arrays = execute_numerical(request)
    config = request['scenario_payload']['config']
    count = int(result['episode_length'])
    result = dict(result)
    result.update(completion_time=count * config['dt'] if result['success'] else None,
                  truncated_time=count * config['dt'] if result['success'] else config['max_steps'] * config['dt'],
                  horizon_time=config['max_steps'] * config['dt'])
    trace = HERE / 'traces' / (request['rollout_uid'] + '.npz')
    trace.parent.mkdir(exist_ok=True)
    temporary = trace.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(trace)
    result.update({key: request[key] for key in ('scenario_uid', 'state_uid', 'controller_uid', 'eta_uid', 'seed_key', 'rollout_uid',
                                                'scene', 'type_id', 'state_index', 'seed', 'spec', 'purpose', 'solo', 'benchmark', 'phase')})
    result.update(controller=request['controller_payload']['nominal']['type_code'], eta=request['eta_a'], future_index=request['seed'],
                  cache_identity_schema='registered_exact_uid_v1', initial_positions=request['state']['positions'],
                  initial_velocities=request['state']['velocities'], goals=request['state']['goals'], deadlock=0,
                  source='new_rollout', trace=str(trace), trace_sha256=sha(trace), worker_cpu=list(os.sched_getaffinity(0)),
                  wall_seconds=time.monotonic() - start)
    result = clean(result)
    cache_writer.append_journal([result], EXP)
    dump(HERE / 'raw' / (request['rollout_uid'] + '.json'), result)
    return result


def merge(cases):
    assert os.environ.get('STB_COORDINATOR') == '1'
    from shared_rollout_db.src import merger
    previous = list(sys.argv)
    sys.argv = ['merger', '--experiment', EXP]
    try:
        merger.main()
    finally:
        sys.argv = previous
    with db.connect(True) as connection:
        for request in cases:
            path = HERE / 'raw' / (request['rollout_uid'] + '.json')
            if not path.exists():
                continue
            result = read(path)
            row = connection.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?', (request['rollout_uid'],)).fetchone()
            assert row and row['success'] == int(result['success']) and row['numerical_failure'] == int(result['numerical_failure']) and not row['conflict_quarantined']
