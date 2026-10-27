"""Swap8 correction-ceiling search: new files only; frozen supplement modules are imported read-only."""
import sys
from pathlib import Path
S2 = Path(__file__).resolve().parent
PARENT = S2.parent
sys.path.insert(0, str(PARENT))
import bootstrap  # noqa: F401  parent output/cache environment and import order
from experiment import *  # noqa: F401,F403  HERE == PARENT
import numpy as np
from functools import lru_cache
import shared_rollout_db
from shared_rollout_db.src import rollout_db as dbm, planner, cache_writer
from case_factory_user_fix_20261023 import make
from registry import verify_receipt

assert HERE == PARENT and S2.parent == HERE
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT / 'shared_rollout_db')
assert (cache_writer.ROOT / 'journals').resolve() == ROOT / 'shared_rollout_db/journals'

EXP2 = 'exp_swap8_best_correction_search_v1'
SCENES = ('lit_swap8_free', 'lit_swap8_obstacles', 'lit_swap8_moving')
SHORT = {'lit_swap8_free': 'free', 'lit_swap8_obstacles': 'static', 'lit_swap8_moving': 'moving'}
BOX_H = np.array([[.5, 1.25], [-.5, .5], [0., .75]])
BOX_E = np.array([[0., 2.], [-1.5, 1.5], [0., 1.5]])
SIGMA = np.array([.2, .3, .15])
THETAS = (.25, .5, .75)
SEED = 20261023
FROZEN = ('PREREGISTRATION.receipt.json', 'IMPLEMENTATION_FROZEN.receipt.json', 'USER_FIX_IMPLEMENTATION_FROZEN_20261023.receipt.json',
          'USER_FIX_READY_20261023.receipt.json', 'WRITE_PATH_READY.receipt.json')


def write2(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(S2), path
    write(path, value)


def seal2(name, paths, **metadata):
    path = S2 / name
    assert not path.exists(), path
    write2(path, dict(utc=utc(), files={str(Path(p).resolve()): sha(p) for p in paths}, **metadata))


def integrity(label):
    """Frozen science receipts and the sealed parent COMPLETE reports must be unchanged."""
    for name in FROZEN:
        verify_receipt(name)
    complete = read(HERE / 'COMPLETE.json')
    for name, digest in complete['reports'].items():
        assert sha(HERE / name) == digest, name
    assert sha(HERE / 'COMPLETE.json') == read(S2 / 'START.json')['parent_complete_sha256'], 'parent COMPLETE changed'
    out = dict(utc=utc(), label=label, passed=True, frozen_receipts=list(FROZEN), parent_reports=len(complete['reports']))
    write2(S2 / 'INTEGRITY_CHECKS' / f'{time_ns()}_{label}.json', out)
    return out


def time_ns():
    import time
    return time.time_ns()


@lru_cache(None)
def selection():
    return read(HERE / 'SCREEN_SELECTION_PRIMARY_20261023.json')


@lru_cache(None)
def manifest():
    return {r['controller']: r for r in read(HERE / 'CONTROLLER_MANIFEST.json')}


def train_ctls(scene):
    return [c for c in selection()[scene]['training_controllers'].values() if c is not None]


def train_states(scene):
    return selection()[scene]['training_states']


def eval_states(scene):
    return selection()[scene]['evaluation_states']


def qualified(scene):
    return [manifest()[c] for c in selection()[scene]['qualified']]


@lru_cache(None)
def k16():
    # Same JSON as case_factory.candidates(), read without appending to the sealed parent's evidence log.
    return read(ROOT / 'diagnostics/controller_loco_real_flow_v1/protocol.json')['eta']


def act_safety():
    return dict(key='safety', eta_a=[0., 0., 0.], eta_b=[0., 0., 0.], theta=None)


def act_k16(a, b=None):
    eta = k16(); b = a if b is None else b
    return dict(key=f'eta{a}' if a == b else f'eta{a}_to_{b}', eta_a=eta[a], eta_b=eta[b], theta=None if a == b else .968)


def act_fixed(key, value):
    value = [float(x) for x in value]
    return dict(key=key, eta_a=value, eta_b=list(value), theta=None)


def act_schedule(key, a, b, theta):
    return dict(key=key, eta_a=[float(x) for x in a], eta_b=[float(x) for x in b], theta=float(theta))


@lru_cache(None)
def sobol_points():
    from scipy.stats import qmc
    u = qmc.Sobol(d=3, scramble=True, seed=SEED).random(64)
    return BOX_E[:, 0] + u * (BOX_E[:, 1] - BOX_E[:, 0])


def sobol_actions():
    return [act_fixed(f'sE{i:02d}', p) for i, p in enumerate(sobol_points())]


def k16_actions():
    return [act_k16(i) for i in range(16)]


def register2(stage, cases):
    """Official registration + preflight for a new stage; adapted from the sealed registry with EXP2/S2 paths."""
    for name in FROZEN:
        verify_receipt(name)
    assert not (S2 / f'{stage}_REGISTRATION.receipt.json').exists(), stage
    unique = {}
    for c in cases:
        unique.setdefault(c['rollout_uid'], c)
    cases = list(unique.values())
    tables = ('experiment', 'scenario', 'state', 'controller_config', 'eta')
    added = {t: set() for t in tables}; inserted = []
    with dbm.connect() as db, dbm.transaction(db):
        keys = {t: db.execute('PRAGMA table_info(' + t + ')').fetchone()['name'] for t in tables}

        def digest(t, skip=()):
            hh = hashlib.sha256(); count = 0
            for r in db.execute('SELECT * FROM ' + t + ' ORDER BY 1'):
                if r[keys[t]] in skip:
                    continue
                hh.update(dbm.canonical(dict(r)).encode()); count += 1
            return dict(rows=count, sha256=hh.hexdigest())
        before = {t: digest(t) for t in tables}; seen = {t: set() for t in tables}

        def ins(t, v):
            key = v[keys[t]]
            if key in seen[t]:
                return
            seen[t].add(key)
            old = db.execute('SELECT * FROM ' + t + ' WHERE ' + keys[t] + '=?', (key,)).fetchone()
            if old:
                assert all(old[k] == x for k, x in v.items()), ('UID_CONTENT_CONFLICT', t, key)
            else:
                db.execute('INSERT INTO ' + t + '(' + ','.join(v) + ') VALUES(' + ','.join('?' for _ in v) + ')', list(v.values()))
                added[t].add(key); inserted.append(dict(table=t, row=v))
        ins('experiment', dict(experiment_uid=EXP2, name='Swap8 correction ceiling search v1', path=str(S2), protocol_hash=sha(S2 / 'PROTOCOL.md'),
                               code_hash=sha(S2 / 'ceiling_common.py'), metadata_json=canonical(dict(evidence='DEVELOPMENT', parent=str(HERE)))))
        for c in cases:
            s = c['state']; physical = {k: s[k] for k in ('positions', 'velocities', 'goals')}
            ins('scenario', dict(scenario_uid=c['scenario_uid'], name=s['scene'] + '_' + c['version'], code_config_fingerprint=c['scenario_uid'], metadata_json=canonical(c['scenario_payload'])))
            ins('state', dict(state_uid=c['state_uid'], scenario_uid=c['scenario_uid'], source_group=f"N{s['N']}_state{s['index']}", content_hash=uid('physical', physical),
                              physical_state_json=canonical(physical), goals_geometry_json=canonical(c['scenario_payload']), provenance_json=canonical(c['state_provenance']), identity_quality='CONTENT_EXACT'))
            ins('controller_config', dict(controller_uid=c['controller_uid'], scenario_uid=c['scenario_uid'], config_json=canonical(c['controller_payload']), compatibility_quality='EXACT_PROFILE'))
            eid, v, hx = dbm.eta_identity(c['eta_a']); assert eid == c['eta_uid']
            ins('eta', dict(eta_uid=eid, eta1=v[0], eta2=v[1], eta3=v[2], canonical_hex=hx))
        after = {t: digest(t, added[t]) for t in tables}; assert before == after
    write2(S2 / f'{stage}_INSERTED_ROWS.json', inserted)
    seal2(f'{stage}_REGISTRATION.receipt.json', [S2 / f'{stage}_INSERTED_ROWS.json'], preexisting_rows_before=before, preexisting_rows_after=after)
    write2(S2 / f'{stage}_CASES.json', cases)
    plan = S2 / f'{stage}_PLANNED_ROLLOUTS.json'
    write2(plan, dict(requests=[{k: c[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key')} for c in cases]))
    pf = planner.preflight(plan); write2(S2 / f'{stage}_CACHE_PREFLIGHT.json', pf)
    assert not pf['summary']['ambiguous'], pf['summary']
    missing = {(r['state_uid'], r['eta_uid'], r['controller_uid'], s) for r in pf['details'] for s in r['missing_seeds']}
    todo = []; reuse = []
    with dbm.connect(True) as db:
        for c in cases:
            key = (c['state_uid'], c['eta_uid'], c['controller_uid'], c['seed_key'])
            existing = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
            if key in missing and existing is None:
                todo.append(c)
            else:
                assert existing is not None, ('missing exact evidence row', c['rollout_uid'])
                reuse.append(dict(rollout_uid=c['rollout_uid'], cached_row=dict(existing)))
    write2(S2 / f'{stage}_REUSED_ROWS.json', reuse)
    write2(S2 / f'{stage}_MISSING_CASES.json', todo)
    seal2(f'{stage}_PREFLIGHT.receipt.json', [plan, S2 / f'{stage}_CACHE_PREFLIGHT.json', S2 / f'{stage}_MISSING_CASES.json', S2 / f'{stage}_CASES.json', S2 / f'{stage}_REUSED_ROWS.json'],
          before_submission=True, unknowns_not_retried=True)
    print(stage, pf['summary'], 'executions', len(todo), 'reused', len(reuse), flush=True)
    return todo, reuse


def build_queue(stage, chunk=16):
    folder = S2 / 'QUEUES' / stage
    assert not (folder / 'QUEUE.receipt.json').exists(), stage
    todo = read(S2 / f'{stage}_MISSING_CASES.json'); groups = {}
    with dbm.connect(True) as db:
        for c in todo:
            assert db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone() is None, c['rollout_uid']
            s = c['state']; key = (s['scene'], c['controller_payload']['nominal']['controller'], s['N'], s['index'], c['seed'])
            groups.setdefault(key, []).append(c)
    index = []
    for key, rows in groups.items():
        rows.sort(key=lambda x: x['requested_action'])
        for k in range(0, len(rows), chunk):
            part = rows[k:k + chunk]; gid = uid('group', [list(key), k])
            p = folder / 'groups' / f'{gid}.json'; write2(p, part)
            nom = part[0]['controller_payload']['nominal']
            factor = {'distributed_cross_entropy': 2, 'cross_entropy': 1.5, 'distributed_shooting': 1.3}.get(nom['type_code'], 1)
            index.append(dict(group=gid, path=str(p), sha256=sha(p), count=len(part), scheduling_cost=len(part) * factor * (1.7 if key[0] != 'lit_swap8_free' else 1)))
    index.sort(key=lambda r: (-r['scheduling_cost'], r['group']))
    write2(folder / 'INDEX.json', index)
    write2(folder / 'QUEUE.receipt.json', dict(utc=utc(), stage=stage, missing=sum(r['count'] for r in index), groups=len(index),
                                               files={str(p): sha(p) for p in (S2 / f'{stage}_MISSING_CASES.json', folder / 'INDEX.json')}))
    with (S2 / 'REQUEST_LEDGER.jsonl').open('a') as f:
        f.write(canonical(read(folder / 'QUEUE.receipt.json')) + '\n')
    print(stage, 'groups', len(index), 'missing', sum(r['count'] for r in index), flush=True)
    return index


@lru_cache(None)
def parent_raw_index():
    out = {}
    for p in HERE.glob('*/raw/roll_*.json'):
        out[p.stem] = p
    for p in S2.glob('*/raw/roll_*.json'):
        out[p.stem] = p
    return out


def locate2(stage, c):
    p = S2 / stage / 'raw' / f"{c['rollout_uid']}.json"
    if p.exists():
        return p
    if c.get('reuse_source'):
        return Path(c['reuse_source'])
    p = parent_raw_index().get(c['rollout_uid'])
    assert p is not None, ('Missing immutable raw', stage, c['rollout_uid'])
    return p


def outcome_row(stage, c):
    p = locate2(stage, c); r = read(p); assert r['rollout_uid'] == c['rollout_uid']
    s = c['state']; ctl = c['controller_payload']['nominal']
    return dict(rollout_uid=c['rollout_uid'], scene=s['scene'], state=s['index'], N=s['N'], controller=ctl['controller'], type=ctl['type_code'], seed=c['seed'],
                action=c['requested_action'], eta_a=c['eta_a'], eta_b=c['eta_b'], theta=c['theta'], success=int(r['success']), upper=min(1, int(r['success']) + int(r['numerical_failure'])),
                unknown=int(r['numerical_failure']), collision=int(r['collision']), timeout=int(r['timeout']), collision_types=r.get('collision_types', {}),
                restoration_steps=r.get('restoration_steps', 0), constraint_violation_steps=r.get('constraint_violation_steps', 0), episode_length=r.get('episode_length'),
                elapsed_seconds=r.get('elapsed_seconds'), raw_path=str(p))


def stage_rows(stage):
    return [outcome_row(stage, c) for c in read(S2 / f'{stage}_CASES.json')]
