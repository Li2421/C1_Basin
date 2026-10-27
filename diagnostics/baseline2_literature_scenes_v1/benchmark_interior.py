"""Paired nominal-query timing on saved poses; no environment steps or labels."""
from experiment import *
import numpy as np
import time
import jax
from flow_scene import LiteratureEnv
from states import LiteratureConfig
from controllers import Nominal as Old
from controllers_interior import Nominal as New
jax.config.update('jax_enable_x64', False)


def main():
    candidates = {}
    for path in (HERE/'safety_lit_circle_five/raw').glob('*.json'):
        r = read(path)
        kind = r['case']['controller_payload']['nominal']['type_code']
        if kind not in ('cross_entropy', 'distributed_cross_entropy',
                        'centralized_mppi', 'centralized_sampling'):
            continue
        if r['numerical_failure'] or r['case']['seed'] != 0:
            continue
        if kind not in candidates or r['elapsed_seconds'] > candidates[kind][1]['elapsed_seconds']:
            candidates[kind] = (path, r)
    rows = []
    for kind, (path, r) in sorted(candidates.items()):
        c = r['case']
        env = LiteratureEnv(c['state'], LiteratureConfig(**c['scenario_payload']['config']))
        old, new = Old(c['controller_payload']['nominal']), New(c['controller_payload']['nominal'])
        with np.load(r['trace_path']) as z:
            points, velocities = z['positions'], z['velocities']
        sums = [0., 0.]
        for i, t in enumerate(np.linspace(0, len(points)-2, 32, dtype=int)):
            env.positions, env.velocities, env.step_count = points[t].copy(), velocities[t].copy(), int(t)
            key = jax.random.fold_in(jax.random.PRNGKey(2026102008), i)
            actions = [None, None]
            for j in ((0, 1) if i % 2 == 0 else (1, 0)):
                start = time.perf_counter()
                actions[j] = (old if j == 0 else new).action(env, key)
                sums[j] += time.perf_counter()-start
            assert actions[0].dtype == actions[1].dtype and actions[0].tobytes() == actions[1].tobytes()
        rows.append(dict(type=kind, source=str(path), source_sha256=sha(path),
                         canonical_seconds=sums[0], interior_seconds=sums[1],
                         speed_ratio=sums[0]/sums[1], queries=32, all_actions_byte_identical=True))
        print(kind, sums, sums[0]/sums[1], flush=True)
    write(HERE/'CONTROLLER_INTERIOR_PAIRED_TIMING.json', dict(
        utc=utc(), rows=rows, new_physical_steps=0,
        alternating_query_order=True, caveat='Wall-clock development measurement on saved poses; not a research endpoint'))


if __name__ == '__main__':
    main()
