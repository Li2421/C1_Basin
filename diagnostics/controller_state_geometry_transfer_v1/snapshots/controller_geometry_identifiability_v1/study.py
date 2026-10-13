"""Controller x geometry identifiability validation: design, registration, workers, merge.

  python -m diagnostics.controller_geometry_identifiability_v1.study design --stage pilot
  python -m diagnostics.controller_geometry_identifiability_v1.study worker --stage pilot --index 0 --workers 24
  python -m diagnostics.controller_geometry_identifiability_v1.study merge  --stage pilot
  python -m diagnostics.controller_geometry_identifiability_v1.study design --stage main   # applies frozen grid rule
"""
from __future__ import annotations

import argparse, copy, dataclasses, hashlib, json, time
from collections import Counter
from pathlib import Path

import numpy as np

from shared_rollout_db.src.rollout_db import connect, transaction, uid, canonical, eta_identity
from shared_rollout_db.src.cache_writer import append_journal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCHIVE = HERE.parent / 'controller_four_scene_mechanism_v1'
EXPERIMENT = 'exp_controller_geometry_identifiability_v1'
FAMILIES = ('isotropic', 'anisotropic', 'rotate_positive', 'rotate_negative', 'coupled')
ETA_INDICES = [0, 5, 10, 15]
CANDIDATES = [round(0.36 + 0.06 * i, 2) for i in range(22)]          # 0.36 .. 1.62
PILOT_WIDTHS = CANDIDATES[::3]                                      # 0.36 .. 1.62, 8 widths
PILOT_STATE_INDICES = list(range(8))
MAIN_STATE_INDICES = list(range(16, 32))
FLAGS = ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure')

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n'); tmp.replace(p)


def freeze(p, x):
    if Path(p).exists():
        assert read(p) == x, ('immutable artifact differs', str(p))
    else:
        write(p, x)


# ----------------------------------------------------------------------------- identities
def archived():
    a = read(ARCHIVE / 'protocol.json')
    db_states = [s for s in a['states'] if s['scenario'] == 'double_bottleneck' and s['study_split'] == 'train']
    db_states = {s['study_index']: s for s in db_states}
    env = a['environments']['double_bottleneck']
    assert sha(REPO / 'double_bottleneck/environment.py') == env['config']['environment_code_sha256'], 'DB env code changed'
    return a, db_states, env


def geometry(width, env):
    from double_bottleneck.environment import Config
    base = env['config']['config']
    cfg = dataclasses.replace(Config(**base), bottleneck_width=float(width))   # native validation
    meta = dict(parent_scenario_uid=env['scenario_uid'], geometry_override=dict(bottleneck_width=float(width)),
                config=cfg.to_dict(), environment_code_sha256=env['config']['environment_code_sha256'],
                layout_code_sha256=sha(REPO / 'double_bottleneck/scenario.py'), reset=env['config']['reset'])
    digest = hashlib.sha256(canonical(meta).encode()).hexdigest()
    return dict(width=float(width), scenario_uid='scenario_' + digest, env_fingerprint='env_' + digest, meta=meta)


def controller_profile(a, geo, family):
    payload = copy.deepcopy(a['controllers']['double_bottleneck'][family]['payload'])
    payload.update(scenario_uid=geo['scenario_uid'], environment_sha256=geo['env_fingerprint'],
                   geometry_override=geo['meta']['geometry_override'],
                   execution_callable=str(HERE / 'runtime.py'), execution_callable_sha256=sha(HERE / 'runtime.py'),
                   archived_execution_callable=a['controllers']['double_bottleneck'][family]['payload']['execution_callable'],
                   conditioning='true-t0; endpoint-matched goal-reference FF operator; only DB bottleneck_width overridden')
    return dict(controller_uid=uid('ctl', payload), payload=payload)


def state_profile(s, geo):
    sid = uid('state', dict(parent_state_uid=s['state_uid'], scenario_uid=geo['scenario_uid']))
    return dict(state_uid=sid, content_hash=uid('content', dict(physical=s['physical'], scenario=geo['scenario_uid'])),
                alias=f"{s['uid']}@w{geo['width']:.2f}")


def make_case(stage, geo, role, s, sp, ctl, family, a, j, seed):
    eta = a['eta'][j]; eid = eta_identity(eta)[0]
    sk = canonical(dict(future_index=seed, future_root=2026100403, rng_namespace=s['rng_namespace']))
    return dict(stage=stage, width=geo['width'], width_role=role, scenario_uid=geo['scenario_uid'],
                state_uid=sp['state_uid'], parent_state_uid=s['state_uid'], state_alias=s['uid'], state_index=s['study_index'],
                regime=s['metadata']['regime'], controller_uid=ctl['controller_uid'], controller_name=family,
                eta_uid=eid, eta_index=j, eta=eta, seed=seed, seed_key=sk,
                rollout_uid=uid('roll', dict(state=sp['state_uid'], eta=eid, controller=ctl['controller_uid'], seed=sk)))


# ----------------------------------------------------------------------------- designs
def main_grid():
    """Frozen rule of PREREGISTRATION.md section 3, applied to merged pilot labels."""
    cases = read(HERE / 'cases_pilot.json')
    by = {}
    for c in cases:
        f = HERE / 'raw' / (c['rollout_uid'] + '.json')
        r = read(f)
        if r['numerical_failure']:
            continue
        by.setdefault(c['width'], {}).setdefault((c['controller_name'], c['eta_index']), []).append(r['success'])
    informative = []
    detail = {}
    for w, cells in sorted(by.items()):
        means = [float(np.mean(v)) for v in cells.values()]
        rng_ = max(means) - min(means)
        detail[f'{w:.2f}'] = dict(cell_range=rng_, mean=float(np.mean(means)), cells=len(means))
        if rng_ >= 0.25:
            informative.append(w)
    if not informative:
        return dict(status='NO_SIGNAL', pilot=detail)
    a_, b_ = max(0.36, min(informative) - 0.18), min(1.62, max(informative) + 0.18)
    snap = lambda x: min(CANDIDATES, key=lambda c: abs(c - x))
    grid = sorted({snap(x) for x in np.linspace(a_, b_, 14)})
    n = len(grid)
    test = sorted({grid[int(round(k * (n - 1) / 5))] for k in range(1, 5)})
    pool = [w for w in grid if w not in test]
    few = sorted({pool[0], pool[len(pool) // 2], pool[-1]})
    return dict(status='OK', pilot=detail, informative=informative, interval=[a_, b_], grid=grid,
                test=test, pool=pool, few=few)


def design(stage):
    a, db_states, env = archived()
    if stage == 'pilot':
        prereg_sha = sha(HERE / 'PREREGISTRATION.md')
        widths = {w: 'pilot' for w in PILOT_WIDTHS}
        state_idx = PILOT_STATE_INDICES
        seeds = {'pilot': [0]}
        extra = dict(preregistration_sha256=prereg_sha)
    else:
        g = main_grid(); write(HERE / 'main_grid.json', g)
        assert g['status'] == 'OK', g
        widths = {w: ('test' if w in g['test'] else ('few' if w in g['few'] else 'pool')) for w in g['grid']}
        state_idx = MAIN_STATE_INDICES
        seeds = {'pool': [0, 1], 'few': [0, 1, 2, 3], 'test': [0, 1, 2, 3]}
        extra = dict(preregistration_sha256=read(HERE / 'protocol_pilot.json')['preregistration_sha256'],
                     main_grid=g, pilot_protocol_sha256=sha(HERE / 'protocol_pilot.json'))
        assert sha(HERE / 'PREREGISTRATION.md') == extra['preregistration_sha256'], 'preregistration changed after pilot'
    geos = {w: geometry(w, env) for w in widths}
    states = [db_states[i] for i in state_idx]
    controllers = {w: {f: controller_profile(a, geos[w], f) for f in FAMILIES} for w in widths}
    sprof = {(w, s['state_uid']): state_profile(s, geos[w]) for w in widths for s in states}
    cases = []
    for w, role in sorted(widths.items()):
        for s in states:
            for f in FAMILIES:
                for j in ETA_INDICES:
                    for seed in seeds[role]:
                        cases.append(make_case(stage, geos[w], role, s, sprof[(w, s['state_uid'])], controllers[w][f], f, a, j, seed))
    assert len({c['rollout_uid'] for c in cases}) == len(cases)
    protocol = dict(schema='controller_geometry_identifiability_v1', stage=stage, experiment_uid=EXPERIMENT,
                    widths={f'{w:.2f}': r for w, r in sorted(widths.items())}, geometries={f'{w:.2f}': g for w, g in sorted(geos.items())},
                    state_indices=state_idx, parent_states=[s['state_uid'] for s in states], families=list(FAMILIES),
                    eta_indices=ETA_INDICES, eta={str(j): a['eta'][j] for j in ETA_INDICES}, seeds=seeds,
                    runtime_sha256=sha(HERE / 'runtime.py'), archived_protocol_sha256=sha(ARCHIVE / 'protocol.json'),
                    reproduction=read(HERE / 'reproduction.json')['all_equal'], n_cases=len(cases),
                    controllers={f'{w:.2f}': {f: c['controller_uid'] for f, c in v.items()} for w, v in controllers.items()},
                    video_policy='Pure Q-label/evaluation experiment: no video delivery required by AGENTS.md',
                    history_check='No archived experiment varies DB bottleneck_width (all 260 archived records use 0.5); '
                                  'archived width-0.50 goal-reference labels used only for the exact reproduction gate.',
                    **extra)
    freeze(HERE / f'protocol_{stage}.json', protocol)
    freeze(HERE / f'cases_{stage}.json', cases)
    groups = {}
    for c in cases:
        groups.setdefault((c['state_uid'], c['eta_uid'], c['controller_uid']), []).append(c['seed_key'])
    freeze(HERE / f'planned_rollouts_{stage}.json',
           dict(requests=[dict(state_uid=s, eta_uid=e, controller_uid=c, seed_keys=k) for (s, e, c), k in groups.items()]))
    register(stage, a, geos, states, sprof, controllers)
    print(json.dumps(dict(stage=stage, widths=len(widths), states=len(states), cases=len(cases))), flush=True)


def register(stage, a, geos, states, sprof, controllers):
    with connect() as db, transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) '
                   'VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXPERIMENT, 'Controller x geometry identifiability validation (DB width family)', str(HERE),
                    sha(HERE / 'PREREGISTRATION.md'), sha(HERE / 'runtime.py'), canonical(dict(max_cpu=24))))
        for w, g in geos.items():
            db.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',
                       (g['scenario_uid'], f"double_bottleneck_width_{w:.2f}", g['env_fingerprint'], canonical(g['meta'])))
            for s in states:
                sp = sprof[(w, s['state_uid'])]
                db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,'
                           'goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                           (sp['state_uid'], g['scenario_uid'], s['family'], sp['content_hash'], canonical(s['physical']),
                            canonical(g['meta']['config']), canonical(dict(parent_state_uid=s['state_uid'], source_alias=s['uid'],
                                                                           metadata=s['metadata'], experiment=EXPERIMENT)), 'CONTENT_EXACT'))
                db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)', (g['scenario_uid'], sp['alias'], sp['state_uid'], EXPERIMENT))
            for f, ctl in controllers[w].items():
                v = ctl['payload']
                prior = db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?', (ctl['controller_uid'],)).fetchone()
                assert prior is None or prior[0] == canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,'
                           'safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,'
                           'compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                           (ctl['controller_uid'], v['scenario_uid'], v['flow_checkpoint_sha256'], v['orthoflow3_sha256'],
                            uid('safe', dict(code=v['safety_projection_sha256'], cbf=v['cbf'])), str(v['horizon']), str(v['dt']),
                            v['success_semantics'], v['conditioning'], canonical(v['rng']), canonical(v), 'EXACT_PROFILE'))
        for j in ETA_INDICES:
            eid, vv, hx = eta_identity(a['eta'][j])
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)', (eid, *vv, hx))


# ----------------------------------------------------------------------------- execution
def worker(stage, index, workers):
    import jax
    from .runtime import base_runtime, controller, rollout
    p = read(HERE / f'protocol_{stage}.json'); assert p['runtime_sha256'] == sha(HERE / 'runtime.py')
    pre = read(HERE / f'cache_preflight_{stage}.json')['summary']; assert pre['ambiguous'] == 0
    a, db_states, _ = archived()
    parents = {s['state_uid']: s for s in db_states.values()}
    x64 = bool(a['controllers']['double_bottleneck']['isotropic']['payload']['jax_enable_x64'])
    cases = read(HERE / f'cases_{stage}.json')[index::workers]
    bases, rts, buffer, started, done = {}, {}, [], time.monotonic(), 0
    for c in cases:
        dest = HERE / 'raw' / (c['rollout_uid'] + '.json')
        if dest.exists():
            continue
        jax.config.update('jax_enable_x64', x64)
        w = c['width']
        if w not in bases:
            bases[w] = base_runtime(w)
            assert abs(bases[w].config.bottleneck_width - w) < 1e-12
        if (w, c['controller_name']) not in rts:
            rts[w, c['controller_name']] = controller(bases[w], c['controller_name'])
        s = parents[c['parent_state_uid']]
        r = rollout(rts[w, c['controller_name']], s, c['eta'], c['seed'])
        row = dict(schema='controller_geometry_identifiability_seed_v1', experiment_uid=EXPERIMENT,
                   protocol_sha256=sha(HERE / f'protocol_{stage}.json'), case=c,
                   cache_identity_schema='registered_exact_uid_v1',
                   **{k: c[k] for k in ('scenario_uid', 'state_uid', 'eta_uid', 'controller_uid', 'seed_key', 'eta')},
                   initial_positions=s['physical']['positions'], goals=s['physical']['goals'],
                   initial_velocities=s['physical']['velocities'],
                   flow_checkpoint_sha256=a['controllers']['double_bottleneck'][c['controller_name']]['payload']['flow_checkpoint_sha256'],
                   **r, jax_enable_x64=bool(jax.config.x64_enabled), execution_code_sha256=sha(HERE / 'runtime.py'))
        write(dest, row); buffer.append(row); done += 1
        if len(buffer) >= 8:
            append_journal(buffer, EXPERIMENT, f'{stage}_{index}'); buffer = []
    if buffer:
        append_journal(buffer, EXPERIMENT, f'{stage}_{index}')
    write(HERE / 'workers' / f'{stage}_{index}.json', dict(cases=len(cases), executed=done, seconds=time.monotonic() - started, completed=True))


def merge(stage):
    from shared_rollout_db.src.rollout_db import ROOT as DBROOT
    cases = {c['rollout_uid']: c for c in read(HERE / f'cases_{stage}.json')}
    expected = sha(HERE / f'protocol_{stage}.json')
    stats = Counter()
    with connect() as db:
        for path in sorted((DBROOT / 'journals' / EXPERIMENT).glob(f'{stage}_*.jsonl')):
            source = uid('source', dict(path=str(path), sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?', (source,)).fetchone():
                stats['journals_already_merged'] += 1; continue
            records = [json.loads(v) for v in path.read_text().splitlines() if v.strip()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                           (source, EXPERIMENT, str(path), sha(path), '.jsonl', 'SEED_EXACT', len(records)))
                added = dup = 0
                for line, r in enumerate(records, 1):
                    c = r['case']
                    assert r['protocol_sha256'] == expected and r['experiment_uid'] == EXPERIMENT and cases[c['rollout_uid']] == c
                    assert not (r['success'] and (r['collision'] or r['numerical_failure']))
                    core = tuple(r[k] for k in FLAGS)
                    old = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
                    if old:
                        assert core == tuple(old[k] for k in FLAGS) and r['episode_length'] == old['episode_length']; dup += 1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,'
                                   'deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,'
                                   'compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (c['rollout_uid'], c['state_uid'], c['eta_uid'], c['controller_uid'], c['seed_key'], c['seed_key'],
                                    *core, r['episode_length'], r['outcome'], EXPERIMENT, str(path), 'EXACT_REUSE', uid('record', r)))
                        added += 1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)', (c['rollout_uid'], source, line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?', (added, dup, source))
                stats.update(journals=1, inserted=added, duplicates=dup)
    raw = sum((HERE / 'raw' / (u + '.json')).exists() for u in cases)
    out = dict(stage=stage, **stats, raw_files=raw, planned=len(cases))
    write(HERE / 'merge_audits' / f'{stage}_{time.time_ns()}.json', out); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['design', 'worker', 'merge', 'grid'])
    ap.add_argument('--stage', choices=['pilot', 'main'], required=True)
    ap.add_argument('--index', type=int, default=0); ap.add_argument('--workers', type=int, default=24)
    x = ap.parse_args()
    if x.action == 'design': design(x.stage)
    elif x.action == 'worker': worker(x.stage, x.index, x.workers)
    elif x.action == 'merge': merge(x.stage)
    else: print(json.dumps(main_grid(), indent=2))
