"""Empirical check that gradient MPC, prioritized A* and conflict-based search ignore the seed: run seed 1 of stored
seed-0 search rollouts and compare every saved array bitwise (no DB/journal writes)."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_circle import CIRCLE
from ceiling_worker import execute
import copy

DETERMINISTIC = ('gradient_mpc', 'prioritized_astar', 'conflict_based_search')


def seed1(c):
    from registry import case
    d = copy.deepcopy(c)
    out = case(d['state'], d['version'], d['method'], d['scenario_payload'], d['controller_payload'], 1, d['eta_a'], d['eta_b'], d['theta'], d['state_provenance'])
    for k in ('safety_layer', 'requested_action', 'scene_revision', 'rng_state_uid'):
        if k in d:
            out[k] = d[k]
    return out


def main():
    rng = np.random.default_rng(SEED); out = []
    for stage, scenes in (('global', SCENES), ('circle_global', (CIRCLE,))):
        cases = read(S2 / f'{stage}_MISSING_CASES.json')
        for scene in scenes:
            for t in DETERMINISTIC:
                group = [c for c in cases if c['state']['scene'] == scene and c['controller_payload']['nominal']['type_code'] == t and (S2 / stage / 'raw' / f"{c['rollout_uid']}.json").exists()]
                c0 = group[int(rng.integers(len(group)))]
                stored = read(S2 / stage / 'raw' / f"{c0['rollout_uid']}.json"); trace = np.load(stored['trace_path'])
                c1 = seed1(c0); assert c1['rollout_uid'] != c0['rollout_uid'] and c1['seed'] == 1
                result, arrays = execute(c1)
                keys = [k for k in trace.files if 'seconds' not in k]
                same = {k: bool(np.array_equal(np.asarray(arrays[k]), trace[k])) for k in keys}
                row = dict(scene=scene, type=t, action=c0['requested_action'], state=c0['state']['index'], seed0_outcome=stored['outcome'],
                           seed1_outcome=result['outcome'], identical=all(same.values()) and result['outcome'] == stored['outcome'], arrays=same)
                out.append(row); print(scene, t, row['action'], row['seed0_outcome'], row['seed1_outcome'], 'IDENTICAL' if row['identical'] else 'DIFFERENT', flush=True)
    write2(S2 / 'DETERMINISM_CHECK.json', dict(utc=utc(), passed=all(r['identical'] for r in out), cases=out))


if __name__ == '__main__':
    main()
