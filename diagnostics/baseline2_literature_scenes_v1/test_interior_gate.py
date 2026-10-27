"""Prove the geometry shortcut, then replay 16 saved five-agent episodes."""
from experiment import *
import numpy as np
import time
from types import SimpleNamespace
from controllers import wall_penalties as original_cost
from controllers_interior import wall_penalties as interior_cost


def static():
    rng = np.random.default_rng(2026102007)
    rows = []
    for scene in NAMES:
        env = SimpleNamespace(config=SimpleNamespace(
            scene=scene, agent_radius=2. if scene == 'lit_circle_five' else .15,
            wall_collision_margin=.005, wall_radius=0.))
        for n in (1, 2, 3, 5, 14):
            for scale in (0., 1., 7., 8.995, 9.1, 12.):
                p = rng.normal(size=(16, 12, n, 2)) * scale
                a = original_cost(p, env)
                b = interior_cost(p, env)
                assert all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                           for x, y in zip(a, b)), (scene, n, scale)
                rows.append(dict(scene=scene, n=n, scale=scale))
        if scene == 'lit_circle_five':
            angles = np.arange(4096) * 2 * np.pi / 4096
            for offset in (-1e-8, -1e-10, -1e-12, 0., 1e-12, 1e-10, 1e-8):
                p = (8.995 + offset) * np.stack([np.cos(angles), np.sin(angles)], -1)
                a = original_cost(p, env)
                b = interior_cost(p, env)
                assert all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                           for x, y in zip(a, b)), offset
    # Components of NumPy's computed normals are finite; their norm error is
    # below 4 eps. Two-component dot/norm rounding is far below the 1e-10 m
    # guard for these <=9 m points. The guard only omits exact-zero costs.
    normals = np.stack([np.cos(np.arange(128)*2*np.pi/128),
                        np.sin(np.arange(128)*2*np.pi/128)], -1)
    assert np.max(np.linalg.norm(normals, axis=-1)) <= 1 + 4*np.finfo(float).eps
    write(HERE/'CONTROLLER_INTERIOR_STATIC.json', dict(
        cases=rows, boundary_angles=4096, boundary_offsets=7,
        cost_dtypes_and_bytes_identical=True,
        guard_margin_m=1e-10, estimated_conservative_dot_error_bound_m=1e-12,
        normal_max_norm=float(np.max(np.linalg.norm(normals, axis=-1)))))


def replay():
    from flow_runtime_interior import run
    groups = {}
    for p in (HERE/'safety_lit_circle_five/raw').glob('*.json'):
        r = read(p)
        if r['numerical_failure'] or r['case']['seed'] != 0:
            continue
        typ = r['case']['controller_payload']['nominal']['type_code']
        groups.setdefault(typ, []).append((float(r['elapsed_seconds']), p, r))
    assert len(groups) == 8
    selected = [x for typ in sorted(groups) for x in sorted(groups[typ])[:2]]
    assert len(selected) == 16
    outputs = []
    for _, path, old in selected:
        before_raw = sha(path)
        before_trace = sha(old['trace_path'])
        assert before_trace == old['trace_sha256']
        start = time.monotonic()
        result, arrays = run(old['case'])
        with np.load(old['trace_path']) as z:
            for key in ('positions', 'actions', 'velocities', 'nominal', 'safe',
                        'cone_residual', 'projection_rewrite', 'swept_clearance'):
                x, y = z[key], arrays[key]
                assert x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes(), (path, key)
            assert z['actions'][0].tobytes() == arrays['actions'][0].tobytes()
        for key in ('success', 'collision', 'timeout', 'numerical_failure', 'outcome', 'episode_length'):
            assert result[key] == old[key], (path, key)
        assert sha(path) == before_raw and sha(old['trace_path']) == before_trace
        outputs.append(dict(source=str(path), source_sha256=before_raw,
                            trace=old['trace_path'], trace_sha256=before_trace,
                            type=old['case']['controller_payload']['nominal']['type_code'],
                            old_seconds=old['elapsed_seconds'], replay_seconds=time.monotonic()-start,
                            first_action_bytes=True, full_trace_bytes=True,
                            terminal_and_steps_identical=True))
        print(outputs[-1]['type'], round(outputs[-1]['replay_seconds'], 3), 'exact', flush=True)
    write(HERE/'CONTROLLER_INTERIOR_REPLAY.json', outputs)
    seal('CONTROLLER_INTERIOR_EQUIVALENCE.receipt.json', [
        HERE/'controllers.py', HERE/'controllers_interior.py',
        HERE/'interior_nominal.py', HERE/'flow_runtime_interior.py',
        HERE/'CONTROLLER_INTERIOR.diff', HERE/'CONTROLLER_INTERIOR_PROVENANCE.json',
        HERE/'CONTROLLER_INTERIOR_STATIC.json', HERE/'CONTROLLER_INTERIOR_REPLAY.json',
        HERE/'test_interior_gate.py'],
        same_canonical_semantic_identity=True, historical_replays=16,
        new_scientific_labels_generated=0, original_sources_unchanged=True)


if __name__ == '__main__':
    static()
    replay()
