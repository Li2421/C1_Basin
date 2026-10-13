"""Input-only measurements; preserve all cached goal-reference features."""
import argparse
import time
import numpy as np
from .study import HERE, read, write, freeze, sha
from diagnostics.controller_response_families_v1.runtime import FAMILIES
from diagnostics.controller_ff_encounter_mediation_v1.study import HERE as ENCOUNTER

INDICES = (0, 5, 10, 15)


def run(index, workers):
    import jax
    from diagnostics.controller_flow_endpoint_probe_v1.runtime import FlowEndpointMatchedField
    from diagnostics.controller_q_diversity_v1.runtime import native_runtime, controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    from diagnostics.controller_ff_encounter_mediation_v1.study import probes, valid
    from diagnostics.physical_state_critic_formal_v1.physical_entities import physical_scene
    p = read(HERE / 'protocol.json')
    bases, runtimes, audits = {}, {}, []
    started = time.monotonic()
    for state in p['states'][index::workers]:
        sc, sid = state['scenario'], state['state_uid']
        if sc not in bases:
            bases[sc] = native_runtime(sc)
        base = bases[sc]
        precision = p['controllers'][sc]['flow_isotropic']['payload']['jax_enable_x64']
        jax.config.update('jax_enable_x64', precision)

        def forbidden_step(*args, **kwargs):
            raise RuntimeError('Physical step forbidden in static controller features')
        base.env_cls.step = forbidden_step
        origin = base.make_env(state)
        poses = probes(origin, physical_scene(sc, state['physical'])['obstacles'])
        n = len(origin.positions)
        e = origin.goals - origin.positions
        e /= np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-12)
        lat = np.stack((-e[:, 1], e[:, 0]), -1)
        nominal = None
        for family in FAMILIES:
            name = 'flow_' + family
            dest = HERE / 'features' / sid / (name + '.npz')
            if dest.exists():
                continue
            if (sc, name) not in runtimes:
                runtimes[sc, name] = FlowEndpointMatchedField(base, family)
            rt = runtimes[sc, name]
            jax.config.update('jax_enable_x64', precision)
            if nominal is None:
                nominal = controller_context(rt, state)
            # Nominal endpoints are exactly identical across the five families.
            # The raw field remains controller-specific, including its matrix.
            class Interface:
                def __getattr__(self, key):
                    return getattr(rt, key)
            field = field_context(Interface(), state)
            compact = np.asarray(compact_field(np.concatenate((nominal, field), -1)))
            # Match the already frozen current/encounter feature protocol for
            # goal-reference controllers. Static keys are separate from labels.
            jax.config.update('jax_enable_x64', False)
            env = rt.make_env(state)
            current = np.zeros((16, n, 5), np.float32)
            current_raw = np.full((16, 4, n, 2), np.nan)
            current_ok = np.zeros((16, 4), bool)
            errors = []
            for j in INDICES:
                for k in range(4):
                    key = jax.random.fold_in(jax.random.PRNGKey(2026101204), k)
                    try:
                        a, _, _ = rt.action(env, key, np.asarray(p['eta'][j]), 'field')
                        assert np.isfinite(a).all()
                        current_raw[j, k] = a
                        current_ok[j, k] = True
                    except Exception as ex:
                        errors.append(dict(query='current', eta=j, key=k,
                                           error=type(ex).__name__, message=str(ex)[:240]))
                if current_ok[j].any():
                    a = current_raw[j, current_ok[j]]
                    local = np.stack(((a * e).sum(-1), (a * lat).sum(-1)), -1) / rt.config.max_speed
                    current[j, :, :2], current[j, :, 2:4] = local.mean(0), local.var(0)
                current[j, :, 4] = current_ok[j].mean()
            encounter = np.zeros((16, n, 2, 7), np.float32)
            encounter_raw = np.full((16, n, 2, 2, 2), np.nan)
            encounter_ok = np.zeros((16, n, 2, 2), bool)
            for probe in poses:
                if not probe['valid']:
                    continue
                i, t = probe['agent'], probe['type']
                env.positions[:] = probe['positions']
                env.velocities[:] = origin.velocities
                assert valid(env)
                for j in INDICES:
                    for k in range(2):
                        key = jax.random.fold_in(jax.random.PRNGKey(2026101205), k)
                        try:
                            a, _, _ = rt.action(env, key, np.asarray(p['eta'][j]), 'field')
                            assert np.isfinite(a).all()
                            encounter_raw[j, i, t, k] = a[i]
                            encounter_ok[j, i, t, k] = True
                        except Exception as ex:
                            errors.append(dict(query='encounter', agent=i, probe=t, eta=j,
                                               key=k, error=type(ex).__name__, message=str(ex)[:240]))
                    if encounter_ok[j, i, t].any():
                        a = encounter_raw[j, i, t, encounter_ok[j, i, t]]
                        local = np.stack((a @ e[i], a @ lat[i]), -1) / rt.config.max_speed
                        encounter[j, i, t, :2] = local.mean(0)
                        encounter[j, i, t, 2:4] = local.var(0)
                    encounter[j, i, t, 4] = encounter_ok[j, i, t].mean()
                    encounter[j, i, t, 5:] = probe['displacement']
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, nominal=nominal, field=field, compact=compact,
                                action_feature=np.concatenate((current, encounter.reshape(16, n, 14)), -1),
                                current_raw=current_raw, current_ok=current_ok,
                                encounter_raw=encounter_raw, encounter_ok=encounter_ok,
                                probe_valid=np.asarray([z['valid'] for z in poses]),
                                probe_positions=np.asarray([z['positions'] for z in poses]),
                                physical_steps=0)
            audits.append(dict(state_uid=sid, controller=name, errors=errors,
                               current_queries=16, encounter_queries=8 * sum(z['valid'] for z in poses)))
    write(HERE / f'features_worker{index}.json', dict(complete=True, physical_steps=0,
                                                   seconds=time.monotonic() - started, rows=audits))


def assemble():
    from shared_rollout_db.src.rollout_db import connect
    p = read(HERE / 'protocol.json')
    # Materialize each compressed array once. Keeping a slice of old[key] in
    # every row would retain a new full decompressed parent array each time.
    with np.load(ENCOUNTER / 'dataset.npz') as archive:
        old = {k: archive[k] for k in archive.files}
    cache = {(str(s), str(c)): i for i, (s, c) in enumerate(zip(old['state_uid'], old['controller']))}
    rows = [(s, name) for s in p['states'] for name in sorted(p['controllers'][s['scenario']])]
    row_ix = {(s['state_uid'], name): i for i, (s, name) in enumerate(rows)}
    n = len(rows)
    raw = {k: np.zeros((n, 16, 4), np.int8) for k in ('success', 'unknown', 'collision', 'deadlock', 'timeout', 'observed')}
    raw['termination'] = np.full((n, 16, 4), '', '<U24')
    raw['rollout_uid'] = np.full((n, 16, 4), '', '<U80')
    with connect(True) as db:
        for c in read(HERE / 'cases.json'):
            a = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
            assert a is not None and not a['conflict_quarantined'], c['rollout_uid']
            assert all(a[k] == c[k] for k in ('state_uid', 'eta_uid', 'controller_uid', 'seed_key'))
            i, j, k = row_ix[c['state_uid'], c['controller_name']], c['eta_index'], c['seed']
            assert not raw['observed'][i, j, k]
            for key in ('success', 'collision', 'deadlock', 'timeout'):
                raw[key][i, j, k] = a[key]
            raw['unknown'][i, j, k] = a['numerical_failure']
            raw['observed'][i, j, k] = 1
            raw['termination'][i, j, k] = a['outcome']
            raw['rollout_uid'][i, j, k] = c['rollout_uid']
    data = {k: [] for k in old}
    for i, (state, name) in enumerate(rows):
        ref, family = name.split('_', 1)
        old_i = cache[state['state_uid'], family]
        expected = 2 if state['study_split'] == 'train' else 4
        assert (raw['observed'][i, list(INDICES)].sum(-1) == expected).all()
        for key in data:
            if key.startswith('entity_'):
                data[key].append(old[key][old_i])
        for key, value in dict(eta=np.asarray(p['eta']), scene=state['scenario'], state_uid=state['state_uid'],
                               controller=name, kind=ref, split=state['study_split']).items():
            data[key].append(value)
        s, u, count = raw['success'][i].sum(-1), raw['unknown'][i].sum(-1), raw['observed'][i].sum(-1)
        data['success'].append(s)
        data['unknown'].append(u)
        data['failure'].append(count - s - u)
        if ref == 'goal':
            np.testing.assert_array_equal(s, old['success'][old_i])
            np.testing.assert_array_equal(u, old['unknown'][old_i])
            data['feature'].append(old['feature'][old_i])
            data['action_feature'].append(old['action_feature'][old_i])
        else:
            f = np.load(HERE / 'features' / state['state_uid'] / (name + '.npz'))
            assert int(f['physical_steps']) == 0
            nc = len(f['compact'])
            data['feature'].append(np.pad(f['compact'], ((0, 4-nc), (0, 0))))
            data['action_feature'].append(np.pad(f['action_feature'], ((0, 0), (0, 4-nc), (0, 0))))
    data = {k: np.asarray(v) for k, v in data.items()}
    physical_splits = {}
    for state in p['states']:
        key = (state['scenario'], str(state['physical']))
        assert key not in physical_splits or physical_splits[key] == state['study_split']
        physical_splits[key] = state['study_split']
    for name, artifact in [('dataset.npz', data), ('raw_labels.npz', raw)]:
        assert not (HERE / name).exists(), 'Do not silently overwrite an assembled experiment'
        np.savez_compressed(HERE / name, **artifact)
    freeze(HERE / 'dataset_audit.json', dict(dataset_sha256=sha(HERE/'dataset.npz'), raw_labels_sha256=sha(HERE/'raw_labels.npz'),
        rows=n, physical_states=len(p['states']), train_states_per_scene=32, validation_states_per_scene=8,
        label_outcomes=int(raw['observed'].sum()), new_rollouts=6400, reused_rollouts=8960,
        retained_numerical_unknown=int(raw['unknown'].sum()), collision=int(raw['collision'].sum()),
        physical_train_val_overlap=0, old_goal_labels_and_features_unchanged=True,
        feature_source_sha256=sha(__file__), no_physical_rollouts_in_feature_queries=True))
    print(read(HERE/'dataset_audit.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['features', 'assemble'])
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args()
    run(a.index, a.workers) if a.stage == 'features' else assemble()
