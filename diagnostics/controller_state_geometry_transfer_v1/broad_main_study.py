"""State-transfer method-defect test (PREREGISTRATION.md): design, registration, workers, merge.

  python -m diagnostics.controller_state_transfer_v1.study design
  python -m diagnostics.controller_state_transfer_v1.study worker --index 0 --workers 24
  python -m diagnostics.controller_state_transfer_v1.study merge
"""
from __future__ import annotations

import argparse, copy, dataclasses, hashlib, json, time
from collections import Counter
from pathlib import Path

from shared_rollout_db.src.rollout_db import connect, transaction, uid, canonical, eta_identity
from shared_rollout_db.src.cache_writer import append_journal

HERE = Path(__file__).resolve().parent / "broad_main"
REPO = HERE.parents[2]
ARCHIVE = HERE.parents[1] / 'controller_ff_factorial_repair_v1'
EXPERIMENT = 'exp_controller_state_geometry_transfer_v1_broad_main'
CONTROLLERS = ('flow_isotropic', 'flow_anisotropic')
ETA_INDICES = [0, 2, 4, 5, 8, 10, 12, 15]
TRAIN_WIDTHS = [0.41, 0.47, 0.53, 0.59]
HELD_WIDTHS = [0.44, 0.50, 0.56, 0.62]
TRAIN_STATES = list(range(16,32)); NEW_STATES = list(range(32,44))
FLAGS = ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure')
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp'); t.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n'); t.replace(p)


def freeze(p, x):
    if Path(p).exists():
        assert read(p) == x, ('immutable artifact differs', str(p))
    else:
        write(p, x)


def archived():
    a = read(ARCHIVE / 'protocol.json')
    from .broad_states import make_state
    toy={i:make_state(i) for i in range(44)}
    env = a['environments']['toy_give_way']
    assert sha(REPO / 'single_integrator/environment.py') == env['config']['environment_code_sha256'], 'Toy env code changed'
    return a, toy, env


def geometry(width, env):
    from single_integrator.environment import Config
    cfg = dataclasses.replace(Config(**env['config']['config']), corridor_width=float(width), bay_top=.68)
    meta = dict(parent_scenario_uid=a_scenario(env), geometry_override=dict(corridor_width=float(width), bay_top=.68), config=cfg.to_dict(),
                environment_code_sha256=env['config']['environment_code_sha256'], reset=env['config']['reset'])
    d = hashlib.sha256(canonical(meta).encode()).hexdigest()
    return dict(width=float(width), scenario_uid='scenario_' + d, env_fingerprint='env_' + d, meta=meta)


def a_scenario(env):
    return env.get('scenario_uid') or env['code_config_fingerprint'].replace('env_', 'scenario_')


def design():
    assert read(HERE.parent/'broad_pilot_results.json')['passed'], 'strong-state gate must pass first'
    a, toy, env = archived()
    widths = {w: 'train' for w in TRAIN_WIDTHS} | {w: 'held' for w in HELD_WIDTHS}
    geos = {w: geometry(w, env) for w in widths}
    states = [toy[i] for i in TRAIN_STATES + NEW_STATES]
    ctl = {}
    for w, g in geos.items():
        for name in CONTROLLERS:
            pl = copy.deepcopy(a['controllers']['toy_give_way'][name]['payload'])
            pl.update(scenario_uid=g['scenario_uid'], environment_sha256=g['env_fingerprint'], geometry_override=g['meta']['geometry_override'],
                      execution_callable=str(HERE.parents[1] / 'controller_state_transfer_v1' / 'runtime.py'), execution_callable_sha256=sha(HERE.parents[1] / 'controller_state_transfer_v1' / 'runtime.py'),
                      archived_execution_callable=pl['execution_callable'],
                      conditioning='true-t0; Flow-endpoint FF operator; only Toy corridor_width overridden')
            ctl[w, name] = dict(controller_uid=uid('ctl', pl), payload=pl)
    sp = {(w, s['state_uid']): dict(state_uid=uid('state', dict(parent_state_uid=s['state_uid'], scenario_uid=g['scenario_uid'])),
                                    content_hash=uid('content', dict(physical=s['physical'], scenario=g['scenario_uid'])),
                                    alias=f"{s['uid']}@w{w:.2f}") for w, g in geos.items() for s in states}
    cases = []
    for w, role in sorted(widths.items()):
        for s in states:
            new = s['study_index'] in NEW_STATES
            if role=='train' and s['study_index']>=40: continue
            if role=='held' and s['study_index']<32: continue
            seeds=[0,1,2,3] if (32<=s['study_index']<40 or (role=='train' and w in [.41,.59])) else [0,1]
            for name in CONTROLLERS:
                for j in ETA_INDICES:
                    eid = eta_identity(a['eta'][j])[0]
                    for seed in seeds:
                        sk = canonical(dict(future_index=seed, future_root=2026100403, rng_namespace=s['rng_namespace']))
                        st = sp[(w, s['state_uid'])]['state_uid']; c = ctl[w, name]['controller_uid']
                        cases.append(dict(width=float(w), width_role=role, state_role='new' if new else 'train',
                                          scenario_uid=geos[w]['scenario_uid'], state_uid=st, parent_state_uid=s['state_uid'],
                                          state_index=s['study_index'], controller_uid=c, controller_name=name, eta_uid=eid,
                                          eta_index=j, eta=a['eta'][j], seed=seed, seed_key=sk,
                                          rollout_uid=uid('roll', dict(state=st, eta=eid, controller=c, seed=sk))))
    assert len({c['rollout_uid'] for c in cases}) == len(cases)
    protocol = dict(schema='controller_state_transfer_v1', experiment_uid=EXPERIMENT, preregistration_sha256=sha(HERE / 'PREREGISTRATION.md'),
                    reproduction=read(HERE / 'reproduction.json')['all_equal'], runtime_sha256=sha(HERE.parents[1] / 'controller_state_transfer_v1' / 'runtime.py'),
                    archived_protocol_sha256=sha(ARCHIVE / 'protocol.json'), widths={f'{w:.2f}': r for w, r in sorted(widths.items())},
                    geometries={f'{w:.2f}': g for w, g in sorted(geos.items())}, controllers=list(CONTROLLERS), eta_indices=ETA_INDICES,
                    eta={str(j): a['eta'][j] for j in ETA_INDICES}, train_states=[toy[i]['state_uid'] for i in TRAIN_STATES],
                    new_states=[toy[i]['state_uid'] for i in NEW_STATES], n_cases=len(cases),
                    video_policy='Pure Q-label/evaluation experiment: no video delivery required by AGENTS.md',
                    history_check='Prospective broader t0 states, physical state identities differ; existing controller and all execution semantics retained. Addendum explicitly follows two failed pilots.')
    freeze(HERE / 'protocol.json', protocol); freeze(HERE / 'cases.json', cases)
    groups = {}
    for c in cases:
        groups.setdefault((c['state_uid'], c['eta_uid'], c['controller_uid']), []).append(c['seed_key'])
    freeze(HERE / 'planned_rollouts.json', dict(requests=[dict(state_uid=s, eta_uid=e, controller_uid=c, seed_keys=k) for (s, e, c), k in groups.items()]))
    with connect() as db, transaction(db):
        db.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)',
                   (EXPERIMENT, 'State-transfer method-defect test (Toy corridor width family)', str(HERE), sha(HERE / 'PREREGISTRATION.md'),
                    sha(HERE.parents[1] / 'controller_state_transfer_v1' / 'runtime.py'), canonical(dict(max_cpu=24))))
        for w, g in geos.items():
            db.execute('INSERT OR IGNORE INTO scenario(scenario_uid,name,code_config_fingerprint,metadata_json) VALUES(?,?,?,?)',
                       (g['scenario_uid'], f'toy_give_way_width_{w:.2f}', g['env_fingerprint'], canonical(g['meta'])))
            for s in states:
                p_ = sp[(w, s['state_uid'])]
                db.execute('INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?)',
                           (p_['state_uid'], g['scenario_uid'], s['family'], p_['content_hash'], canonical(s['physical']), canonical(g['meta']['config']),
                            canonical(dict(parent_state_uid=s['state_uid'], source_alias=s['uid'], experiment=EXPERIMENT)), 'CONTENT_EXACT'))
                db.execute('INSERT OR IGNORE INTO state_alias VALUES(?,?,?,?)', (g['scenario_uid'], p_['alias'], p_['state_uid'], EXPERIMENT))
            for name in CONTROLLERS:
                c = ctl[w, name]; v = c['payload']
                prior = db.execute('SELECT config_json FROM controller_config WHERE controller_uid=?', (c['controller_uid'],)).fetchone()
                assert prior is None or prior[0] == canonical(v)
                db.execute('INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                           (c['controller_uid'], v['scenario_uid'], v['flow_checkpoint_sha256'], v['orthoflow3_sha256'],
                            uid('safe', dict(code=v['safety_projection_sha256'], cbf=v['cbf'])), str(v['horizon']), str(v['dt']),
                            v['success_semantics'], v['conditioning'], canonical(v['rng']), canonical(v), 'EXACT_PROFILE'))
        for j in ETA_INDICES:
            eid, vv, hx = eta_identity(a['eta'][j])
            db.execute('INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,canonical_hex) VALUES(?,?,?,?,?)', (eid, *vv, hx))
    print(json.dumps(dict(cases=len(cases), widths=len(widths), states=len(states))), flush=True)


def worker(index, workers):
    import jax
    from diagnostics.controller_state_transfer_v1.runtime import base_runtime, controller, rollout
    p = read(HERE / 'protocol.json'); assert p['runtime_sha256'] == sha(HERE.parents[1] / 'controller_state_transfer_v1' / 'runtime.py')
    pf=read(HERE / 'cache_preflight.json'); assert pf['summary']['ambiguous'] == 0
    missing={(r['state_uid'],r['eta_uid'],r['controller_uid'],sk) for r in pf['details'] for sk in r['missing_seeds']}
    a, toy, _ = archived(); parents = {s['state_uid']: s for s in toy.values()}
    x64 = bool(a['controllers']['toy_give_way']['flow_isotropic']['payload']['jax_enable_x64'])
    bases, rts, buf, t0, done = {}, {}, [], time.monotonic(), 0
    for c in read(HERE / 'cases.json')[index::workers]:
        if (c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key']) not in missing: continue
        dest = HERE / 'raw' / (c['rollout_uid'] + '.json')
        if dest.exists():
            continue
        jax.config.update('jax_enable_x64', x64)
        w = c['width']
        if w not in bases:
            bases[w] = base_runtime(w); assert abs(bases[w].config.corridor_width - w) < 1e-12
        if (w, c['controller_name']) not in rts:
            rts[w, c['controller_name']] = controller(bases[w], c['controller_name'])
        s = parents[c['parent_state_uid']]
        r = rollout(rts[w, c['controller_name']], s, c['eta'], c['seed'])
        row = dict(schema='controller_state_transfer_seed_v1', experiment_uid=EXPERIMENT, protocol_sha256=sha(HERE / 'protocol.json'), case=c,
                   **r, jax_enable_x64=bool(jax.config.x64_enabled), execution_code_sha256=sha(HERE.parents[1] / 'controller_state_transfer_v1' / 'runtime.py'))
        write(dest, row); buf.append(row); done += 1
        if len(buf) >= 8:
            append_journal(buf, EXPERIMENT, f'main_{index}'); buf = []
    if buf:
        append_journal(buf, EXPERIMENT, f'main_{index}')
    write(HERE / 'workers' / f'{index}.json', dict(executed=done, seconds=time.monotonic() - t0, completed=True))


def merge():
    from shared_rollout_db.src.rollout_db import ROOT as DBROOT
    cases = {c['rollout_uid']: c for c in read(HERE / 'cases.json')}; expected = sha(HERE / 'protocol.json'); stats = Counter()
    with connect() as db:
        for path in sorted((DBROOT / 'journals' / EXPERIMENT).glob('main_*.jsonl')):
            source = uid('source', dict(path=str(path), sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?', (source,)).fetchone():
                continue
            records = [json.loads(v) for v in path.read_text().splitlines() if v.strip()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                           (source, EXPERIMENT, str(path), sha(path), '.jsonl', 'SEED_EXACT', len(records)))
                added = dup = 0
                for line, r in enumerate(records, 1):
                    c = r['case']; assert r['protocol_sha256'] == expected and cases[c['rollout_uid']] == c
                    core = tuple(r[k] for k in FLAGS)
                    old = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
                    if old:
                        assert core == tuple(old[k] for k in FLAGS); dup += 1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (c['rollout_uid'], c['state_uid'], c['eta_uid'], c['controller_uid'], c['seed_key'], c['seed_key'], *core,
                                    r['episode_length'], r['outcome'], EXPERIMENT, str(path), 'EXACT_REUSE', uid('record', r)))
                        added += 1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)', (c['rollout_uid'], source, line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?', (added, dup, source))
                stats.update(journals=1, inserted=added, duplicates=dup)
    out = dict(**stats, raw_files=sum((HERE / 'raw' / (u + '.json')).exists() for u in cases), planned=len(cases))
    write(HERE / 'merge_audits' / f'{time.time_ns()}.json', out); print(json.dumps(out), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['design', 'worker', 'merge'])
    ap.add_argument('--index', type=int, default=0); ap.add_argument('--workers', type=int, default=24); x = ap.parse_args()
    {'design': design, 'merge': merge}.get(x.action, lambda: worker(x.index, x.workers))()
