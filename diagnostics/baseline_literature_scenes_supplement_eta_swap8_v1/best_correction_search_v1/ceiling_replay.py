"""Bitwise replay of stored parent rollouts through the search execution path; no DB or journal writes."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_worker import execute
from collections import defaultdict

FIELDS = ('success', 'collision', 'timeout', 'numerical_failure', 'outcome', 'episode_length', 'switch_step', 'minimum_swept_clearance',
          'final_goal_distances', 'collision_types', 'constraint_violation_steps', 'sampled_speed_violation_steps', 'restoration_steps')
ARRAYS = ('positions', 'velocities', 'nominal', 'safe', 'actions', 'projection_rewrite', 'swept_clearance', 'clearance_types')


def pick():
    chosen = []
    plan = [('new_fixed_train', 'lit_swap8_free', 3), ('new_pair_train', 'lit_swap8_free', 1), ('new_fixed_train', 'lit_swap8_obstacles', 3),
            ('swap_main', 'lit_swap8_moving', 3)]
    for stage, scene, k in plan:
        cases = [c for c in read(HERE / f'{stage}_MISSING_CASES.json') if c['state']['scene'] == scene and c['seed'] == 0]
        by_type = defaultdict(list)
        for c in cases:
            by_type[c['controller_payload']['nominal']['type_code']].append(c)
        types = sorted(by_type)
        rng = np.random.default_rng(SEED)
        for t in rng.choice(types, size=k, replace=False):
            group = by_type[t]
            if scene == 'lit_swap8_moving':
                # Include episodes that end in collision to exercise the restoration path.
                collided = [c for c in group if read(HERE / stage / 'raw' / f"{c['rollout_uid']}.json")['collision']]
                group = collided or group
            chosen.append((stage, group[int(rng.integers(len(group)))]))
    return chosen


def main():
    out = []
    for stage, c in pick():
        stored = read(HERE / stage / 'raw' / f"{c['rollout_uid']}.json")
        trace = np.load(stored['trace_path'])
        result, arrays = execute(c)
        result = clean(result); arrays_ok = {k: bool(np.array_equal(np.asarray(arrays[k]), trace[k])) for k in ARRAYS}
        fields_ok = {k: canonical(result.get(k)) == canonical(stored.get(k)) for k in FIELDS}
        out.append(dict(stage=stage, rollout_uid=c['rollout_uid'], scene=c['state']['scene'], controller=c['controller_payload']['nominal']['controller'],
                        state=c['state']['index'], action=c['requested_action'], outcome=result['outcome'], stored_outcome=stored['outcome'],
                        fields=fields_ok, arrays=arrays_ok, passed=all(fields_ok.values()) and all(arrays_ok.values())))
        print(out[-1]['scene'], out[-1]['controller'], out[-1]['action'], out[-1]['outcome'], 'PASS' if out[-1]['passed'] else 'FAIL', flush=True)
    write2(S2 / 'REPLAY_CHECK.json', dict(utc=utc(), passed=all(r['passed'] for r in out), cases=out))
    assert all(r['passed'] for r in out), 'Replay mismatch'


if __name__ == '__main__':
    main()
