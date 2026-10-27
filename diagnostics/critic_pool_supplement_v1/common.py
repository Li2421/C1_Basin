"""Experiment-local identities, immutable evidence access and official cache registration."""
import bootstrap
from experiment import *
import copy
import struct
import time
from functools import lru_cache
from shared_rollout_db.src import rollout_db as dbm, planner, cache_writer

EXP = 'exp_critic_pool_supplement_v1'
SOURCE = ROOT / 'diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1'
SEARCH = SOURCE / 'best_correction_search_v1'
V2 = ROOT / 'diagnostics/baseline_literature_scenes_distributed_v2'
TYPES = ('centralized_mppi', 'centralized_sampling', 'conflict_based_search', 'cross_entropy',
         'distributed_cross_entropy', 'distributed_shooting', 'gradient_mpc', 'prioritized_astar')
assert cache_writer.ROOT.resolve() == (ROOT / 'shared_rollout_db').resolve()


def evidence(path):
    path = Path(path).resolve()
    digest = sha(path)
    with (HERE / 'EVIDENCE_INPUTS.jsonl').open('a') as f:
        f.write(canonical(dict(path=str(path), sha256=digest)) + '\n')
    return digest


def read_evidence(path):
    evidence(path)
    return read(path)


@lru_cache(None)
def actions():
    rows = read_evidence(SEARCH / 'ACTIONS.json')['lit_swap8_free']
    keys = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(32)]
    return [rows[k] for k in keys]


def etahex(eta):
    return ':'.join(struct.pack('>d', float(v)).hex() for v in eta)


def make_case(state, version, method, scenario, controller, seed=0, eta=None, provenance=None):
    eta = [0., 0., 0.] if eta is None else list(eta)
    sc = uid('scenario', scenario)
    physical = {k: state[k] for k in ('positions', 'velocities', 'goals')}
    sid = uid('state', dict(scenario=sc, physical=physical))
    payload = dict(controller, method=method, eta_b=eta, theta=None)
    cid = uid('ctl', dict(scenario=sc, payload=payload))
    eid = dbm.eta_identity(eta)[0]
    sk = canonical(dict(future_index=seed))
    return dict(state=state, version=version, method=method, scenario_payload=scenario,
                controller_payload=payload, scenario_uid=sc, state_uid=sid, controller_uid=cid,
                eta_uid=eid, seed_key=sk, rollout_uid=uid('roll', dict(state=sid, eta=eid, controller=cid, seed=sk)),
                seed=seed, eta_a=eta, eta_b=list(eta), theta=None, state_provenance=provenance,
                safety_layer='distributed', requested_action=method)


def record(case, result):
    return dict(clean(result), **{k: case[k] for k in ('scenario_uid', 'state_uid', 'controller_uid', 'eta_uid', 'seed_key', 'rollout_uid')},
                case=case, eta=case['eta_a'], cache_identity_schema='registered_exact_uid_v1',
                initial_positions=case['state']['positions'], initial_velocities=case['state']['velocities'],
                goals=case['state']['goals'], future_index=case['seed'], deadlock=0, timestamp=utc())


def verify_inputs():
    baselines = read(HERE / 'REUSED_EVIDENCE_SHA256_START.json')['files']
    for p, h in baselines.items():
        assert sha(p) == h, ('AUDIT_INPUT_CHANGED', p)
    if (HERE / 'BACKEND_BUILD.json').exists():
        for row in read(HERE / 'BACKEND_BUILD.json')['sources']:
            assert sha(row['source']) == row['source_sha256'], row['source']
            assert sha(row['target']) == row['target_sha256'], row['target']


def register(stage, cases):
    """Single coordinator only; append new identities, hash all preexisting rows."""
    verify_inputs()
    unique = {c['rollout_uid']: c for c in cases}
    cases = list(unique.values())
    receipt = HERE / f'{stage}_REGISTRATION.receipt.json'
    if not receipt.exists():
        tables = ('experiment', 'scenario', 'state', 'controller_config', 'eta')
        inserted = []; added = {t: set() for t in tables}
        with dbm.connect() as db, dbm.transaction(db):
            keys = {t: db.execute('PRAGMA table_info(' + t + ')').fetchone()['name'] for t in tables}
            def digest(table, skip=()):
                h = hashlib.sha256(); count = 0
                for row in db.execute('SELECT * FROM ' + table + ' ORDER BY 1'):
                    if row[keys[table]] in skip:
                        continue
                    h.update(dbm.canonical(dict(row)).encode()); count += 1
                return dict(rows=count, sha256=h.hexdigest())
            before = {t: digest(t) for t in tables}; seen = {t: set() for t in tables}
            def ins(table, values):
                key = values[keys[table]]
                if key in seen[table]:
                    return
                seen[table].add(key)
                old = db.execute('SELECT * FROM ' + table + ' WHERE ' + keys[table] + '=?', (key,)).fetchone()
                if old:
                    assert all(old[k] == v for k, v in values.items()), ('UID_CONTENT_CONFLICT', table, key)
                else:
                    db.execute('INSERT OR IGNORE INTO ' + table + '(' + ','.join(values) + ') VALUES(' + ','.join('?' for _ in values) + ')', list(values.values()))
                    inserted.append(dict(table=table, row=values)); added[table].add(key)
            ins('experiment', dict(experiment_uid=EXP, name='Critic training pool supplement v1', path=str(HERE),
                protocol_hash=sha(HERE / 'PROTOCOL.md'), code_hash=sha(HERE / 'common.py'), metadata_json=canonical(dict(evidence='DEVELOPMENT'))))
            for c in cases:
                s = c['state']; physical = {k: s[k] for k in ('positions', 'velocities', 'goals')}
                ins('scenario', dict(scenario_uid=c['scenario_uid'], name=s['scene'] + '_' + c['version'],
                    code_config_fingerprint=c['scenario_uid'], metadata_json=canonical(c['scenario_payload'])))
                ins('state', dict(state_uid=c['state_uid'], scenario_uid=c['scenario_uid'], source_group=f"N{s['N']}_state{s['index']}",
                    content_hash=uid('physical', physical), physical_state_json=canonical(physical), goals_geometry_json=canonical(c['scenario_payload']),
                    provenance_json=canonical(c['state_provenance']), identity_quality='CONTENT_EXACT'))
                ins('controller_config', dict(controller_uid=c['controller_uid'], scenario_uid=c['scenario_uid'],
                    config_json=canonical(c['controller_payload']), compatibility_quality='EXACT_PROFILE'))
                eid, value, hx = dbm.eta_identity(c['eta_a']); assert eid == c['eta_uid']
                ins('eta', dict(eta_uid=eid, eta1=value[0], eta2=value[1], eta3=value[2], canonical_hex=hx))
            after = {t: digest(t, added[t]) for t in tables}; assert before == after
        write(HERE / f'{stage}_INSERTED_ROWS.json', inserted)
        write(receipt, dict(utc=utc(), original_rows_before=before, original_rows_after=after,
                           only_insert_new_metadata=True, cases_sha256=uid('cases', cases)))
    else:
        assert read(receipt)['cases_sha256'] == uid('cases', cases)
    write(HERE / f'{stage}_CASES.json', cases)
    plan = HERE / f'{stage}_PLANNED_ROLLOUTS.json'
    write(plan, dict(requests=[{k: c[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key')} for c in cases]))
    pf = planner.preflight(plan); write(HERE / f'{stage}_CACHE_PREFLIGHT.json', pf)
    assert pf['summary']['ambiguous'] == 0
    missing = {(r['state_uid'], r['eta_uid'], r['controller_uid'], sk) for r in pf['details'] for sk in r['missing_seeds']}
    todo = []; reused = []
    with dbm.connect(True) as db:
        for c in cases:
            existing = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
            if existing is not None:
                assert not existing['conflict_quarantined'], ('QUARANTINED', c['rollout_uid'])
                reused.append(dict(rollout_uid=c['rollout_uid'], cached_row=dict(existing)))
            else:
                assert (c['state_uid'], c['eta_uid'], c['controller_uid'], c['seed_key']) in missing
                todo.append(c)
    write(HERE / f'{stage}_MISSING_CASES.json', todo)
    write(HERE / f'{stage}_REUSED_ROWS.json', reused)
    write(HERE / f'{stage}_PREFLIGHT.receipt.json', dict(utc=utc(), summary=pf['summary'], execution_count=len(todo),
        reused_count=len(reused), unknowns_not_retried=True, manifest_sha256=sha(plan), cases_sha256=sha(HERE / f'{stage}_CASES.json')))
    print(stage, 'missing', len(todo), 'reused', len(reused), flush=True)
    return todo, reused
