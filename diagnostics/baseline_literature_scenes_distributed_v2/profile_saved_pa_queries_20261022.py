"""Profile deterministic PA replans from saved traces, without environment steps."""
from experiment import *
import cProfile
import io
import pstats
import time


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    os.nice(10)
    from registry import verify_receipt
    verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
    import numpy as np
    import jax
    from states import LiteratureConfig
    from flow_scene import LiteratureEnv
    from controllers import bound
    from reservation_certificate_v2_start_signature_v2_20261022 import ReservationNominal
    from result_inventory import find_raw
    jax.config.update('jax_enable_x64', False)
    stage = 'distributed_revised_fixed_lit_circle_five'
    cases = read(HERE/(stage+'_CASES.json'))
    rows = []
    sources = []
    for ctl in ('prioritized_astar0', 'prioritized_astar1'):
        c = next(c for c in cases if c['controller_payload']['nominal']['controller'] == ctl
                 and c['state']['index'] == 13 and c['method'] == 'fixed_eta10' and c['seed'] == 0)
        path = find_raw(c['rollout_uid'], stage, c['state']['scene'])
        raw = read(path)
        assert sha(raw['trace_path']) == raw['trace_sha256']
        sources.append(dict(raw_path=str(path), raw_sha256=sha(path),
                            trace_sha256=raw['trace_sha256'], controller=ctl))
        interval = json.loads(c['controller_payload']['nominal']['parameters'])[1]
        with np.load(raw['trace_path']) as z:
            for target in (0, 600, 1200, 1800):
                step = int(target//interval*interval)
                if step >= len(z['nominal']):
                    continue
                env = LiteratureEnv(c['state'], LiteratureConfig(**c['scenario_payload']['config']))
                env.positions = z['positions'][step].copy()
                env.velocities = z['velocities'][step].copy()
                env.step_count = step
                actor = ReservationNominal(c['controller_payload']['nominal'])
                profiler = cProfile.Profile()
                started = time.monotonic()
                value = profiler.runcall(actor.action, env, jax.random.PRNGKey(0))
                value = bound(value, env.config.max_speed).astype(np.float64)
                value[np.linalg.norm(env.goals-env.positions, axis=-1) <= env.config.goal_tolerance] = 0
                elapsed = time.monotonic()-started
                exact = value.tobytes() == z['nominal'][step].tobytes()
                assert exact, (ctl, step)
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats('cumtime').print_stats(30)
                rows.append(dict(controller=ctl, step=step, elapsed_seconds=elapsed,
                                 nominal_bitwise_equal=exact, profile=stream.getvalue()))
                print(ctl, step, round(elapsed, 5), 'exact', exact, flush=True)
    target = HERE/'PA_SAVED_QUERY_PROFILE_20261022.json'
    assert not target.exists()
    write(target, dict(utc=utc(), historical_profile_gap='Existing query profiles cover CBS, not PA',
                       sources=sources, queries=rows, environment_steps=0, new_rollouts=0,
                       new_labels=0, database_writes=0, local_cpu_affinity=sorted(os.sched_getaffinity(0))))
    seal('PA_SAVED_QUERY_PROFILE_20261022.receipt.json',
         [target, Path(__file__), HERE/'reservation_certificate_v2_start_signature_v2_20261022.py',
          HERE/'reservation_certificate_start_signature_v2_20261022.py',
          HERE/'tiled_planning_start_signature_v2_20261022.py'], environment_steps=0, new_rollouts=0)


if __name__ == '__main__':
    main()
