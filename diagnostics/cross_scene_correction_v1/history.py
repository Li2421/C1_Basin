"""Index exact historical training cases; no historical entrypoint is imported."""
from common import *
import numpy as np


def key(c):
    s = c['state']
    return canonical([s['scene'], c['controller_payload']['nominal']['controller'],
                      {k: s[k] for k in ('positions', 'velocities', 'goals')}, c['seed'],
                      etahex(c['eta_a']), etahex(c['eta_b']), c['theta']])


def prepare():
    selection = read_evidence(SOURCE / 'SCREEN_SELECTION_PRIMARY_20261023.json')
    allowed = {s: {v['controller'] for v in row['training_controllers'].values() if v is not None}
               for s, row in selection.items()}
    states = {s: {canonical({k: q[k] for k in ('positions', 'velocities', 'goals')}) for q in row['training_states']}
              for s, row in selection.items()}
    etas = {etahex(a['eta_a']) for a in actions()}
    index = {}
    old_execution_index = read_evidence(SOURCE / 'HISTORY_EXECUTION_INDEX.json')
    raw_index = {}
    for base in (SOURCE, SEARCH, V2):
        for folder in base.iterdir():
            if folder.is_dir() and (folder / 'raw').is_dir():
                for p in (folder / 'raw').glob('roll_*.json'):
                    raw_index.setdefault(p.stem, p)
    sources = [SOURCE / 'old_fixed_train_CASES.json', SOURCE / 'new_fixed_train_CASES.json',
               SEARCH / 'global_CASES.json', SEARCH / 'circle_global_CASES.json']
    for path in sources:
        print('Index', path.name, flush=True)
        rows = read_evidence(path)
        for c in rows:
            s = c['state']; scene = s['scene']
            if scene not in allowed or c['seed'] != 0 or c['theta'] is not None or etahex(c['eta_a']) not in etas:
                continue
            if c['controller_payload']['nominal']['controller'] not in allowed[scene]:
                continue
            if canonical({k: s[k] for k in ('positions', 'velocities', 'goals')}) not in states[scene]:
                continue
            raw = Path(c['reuse_source']) if c.get('reuse_source') else path.parent / path.stem.replace('_CASES', '') / 'raw' / (c['rollout_uid'] + '.json')
            if not raw.exists():
                raw = raw_index.get(c['rollout_uid'], raw)
            if not raw.exists():
                semantic_key = canonical([scene, c['controller_payload']['nominal']['controller'], s['N'], s['index'],
                    c['seed'], [np.asarray(c['eta_a'], np.float64).tobytes().hex(),
                                np.asarray(c['eta_b'], np.float64).tobytes().hex(), c['theta']]])
                prior = old_execution_index.get(semantic_key)
                if prior:
                    raw = Path(prior['path'])
            assert raw.exists(), raw
            c = dict(c, reuse_source=str(raw))
            old = index.get(key(c))
            assert old is None or old['rollout_uid'] == c['rollout_uid'], ('Historical semantic identity conflict', key(c))
            index[key(c)] = c
        del rows
    write(HERE / 'HISTORY_CASES.json', list(index.values()))
    counts = {s: sum(c['state']['scene'] == s for c in index.values()) for s in selection}
    write(HERE / 'HISTORY_EXACT_COVERAGE.json', dict(counts=counts, definition_only=False, physical_identity='exact float64 JSON',
                                                 unknowns_retained=True, note='Official per-stage preflight still required.'))
    replay = []
    for scene in ('lit_circle_five', 'lit_swap8_free', 'lit_swap8_obstacles', 'lit_swap8_moving', 'lit_narrow_passage'):
        rows = [c for c in index.values() if c['state']['scene'] == scene]
        for kind, action in [('cross_entropy', 'eta13'), ('prioritized_astar', 'eta0')]:
            chosen = next(c for c in rows if c['controller_payload']['nominal']['type_code'] == kind and c['requested_action'] == action)
            raw = read_evidence(chosen['reuse_source'])
            assert sha(raw['trace_path']) == raw['trace_sha256']
            evidence(raw['trace_path'])
            replay.append(chosen)
    register('replay', replay)
    write(HERE / 'REPLAY_CASES.json', replay)
    print('Historical cases:', counts, flush=True)


def replay():
    from execution import execute
    fields = ('success', 'collision', 'timeout', 'numerical_failure', 'outcome', 'episode_length',
              'switch_step', 'minimum_swept_clearance', 'final_goal_distances',
              'paper_position_convergence', 'paper_velocity_convergence', 'collision_types',
              'constraint_violation_steps', 'restoration_steps')
    results = []
    for c in read(HERE / 'REPLAY_CASES.json'):
        target = HERE / 'replay' / (c['rollout_uid'] + '.json')
        if target.exists():
            row = read(target); assert row['passed']; results.append(row); continue
        raw = read_evidence(c['reuse_source'])
        assert sha(raw['trace_path']) == raw['trace_sha256']
        old = np.load(raw['trace_path'], allow_pickle=False)
        result, arrays = execute(c)
        checks = {}
        for name in old.files:
            if 'seconds' in name:
                continue
            new = np.asarray(arrays[name]); stored = old[name]
            checks[name] = bool(new.dtype == stored.dtype and new.shape == stored.shape and new.tobytes() == stored.tobytes())
        fc = {k: canonical(result.get(k)) == canonical(raw[k]) for k in fields if k in raw}
        row = dict(scene=c['state']['scene'], rollout_uid=c['rollout_uid'], source=c['reuse_source'],
            trace_path=raw['trace_path'], trace_sha256=raw['trace_sha256'], arrays=checks, fields=fc,
            passed=all(checks.values()) and all(fc.values()), elapsed_seconds=result['elapsed_seconds'])
        write(target, row); results.append(row)
        print('Replay', row['scene'], c['controller_payload']['nominal']['controller'], c['requested_action'], row['passed'], flush=True)
        if not row['passed']:
            write(HERE / 'FATAL_STOP.json', dict(reason='BITWISE_REPLAY_FAILED', case=row))
            raise AssertionError(row)
    write(HERE / 'REPLAY_CHECK.json', dict(utc=utc(), passed=True, cases=results, count=len(results),
        excludes_only_wallclock_timing=True, bytewise_arrays=True))


if __name__ == '__main__':
    import sys
    (prepare if sys.argv[1] == 'prepare' else replay)()
