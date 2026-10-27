"""Circle-five: identity check of reconstructed cases and bitwise replay of stored parent rollouts (no DB/journal writes)."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_circle import make_circle, CIRCLE
from ceiling_worker import execute

FIELDS = ('success', 'collision', 'timeout', 'numerical_failure', 'outcome', 'episode_length', 'switch_step', 'minimum_swept_clearance',
          'final_goal_distances', 'paper_position_convergence', 'paper_velocity_convergence')


def identity():
    out = {}
    fixed = {c['rollout_uid'] for c in read(HERE / 'old_fixed_train_CASES.json') if c['state']['scene'] == CIRCLE}
    pairs = {c['rollout_uid'] for c in read(HERE / 'old_pair_train_CASES.json') if c['state']['scene'] == CIRCLE}
    evals = {c['rollout_uid'] for c in read(HERE / 'old_eval_CASES.json') if c['state']['scene'] == CIRCLE}
    mine = [make_circle(ctl, s, act_k16(i), 0) for ctl in train_ctls(CIRCLE) for s in train_states(CIRCLE) for i in range(16)]
    out['history16_training'] = [sum(c['rollout_uid'] in fixed for c in mine), len(mine)]
    ch = read(HERE / 'OLD_FROZEN_CHOICES.json')['scenes'][CIRCLE]
    acts = [act_k16(*map(int, r['action'].replace('eta', '').split('_to_'))) for r in ch['pair_ranking']]
    mine = [make_circle(ctl, s, a, 0) for ctl in train_ctls(CIRCLE) for s in train_states(CIRCLE) for a in acts]
    out['screen_pairs_training'] = [sum(c['rollout_uid'] in pairs for c in mine), len(mine)]
    acts = [act_k16(9), act_k16(9, 3), act_k16(13), act_k16(13, 10)]
    mine = [make_circle(ctl, s, a, 0) for ctl in qualified(CIRCLE) for s in eval_states(CIRCLE) for a in acts]
    out['evaluation_arms'] = [sum(c['rollout_uid'] in evals for c in mine), len(mine)]
    return out


def main():
    ids = identity(); print(ids, flush=True)
    rng = np.random.default_rng(SEED); out = []
    plan = [('old_fixed_train', ('cross_entropy', 'gradient_mpc', 'distributed_shooting')), ('old_pair_train', ('centralized_sampling',))]
    for stage, types in plan:
        cases = [c for c in read(HERE / f'{stage}_MISSING_CASES.json') if c['state']['scene'] == CIRCLE]
        for t in types:
            group = [c for c in cases if c['controller_payload']['nominal']['type_code'] == t]
            c = group[int(rng.integers(len(group)))]
            stored = read(HERE / stage / 'raw' / f"{c['rollout_uid']}.json"); trace = np.load(stored['trace_path'])
            result, arrays = execute(c); result = clean(result)
            keys = [k for k in trace.files if 'seconds' not in k]
            arrays_ok = {k: bool(np.array_equal(np.asarray(arrays[k]), trace[k])) for k in keys}
            fields_ok = {k: canonical(result.get(k)) == canonical(stored.get(k)) for k in FIELDS if k in stored}
            row = dict(stage=stage, rollout_uid=c['rollout_uid'], controller=c['controller_payload']['nominal']['controller'], state=c['state']['index'],
                       action=c['requested_action'], outcome=result['outcome'], stored_outcome=stored['outcome'], fields=fields_ok, arrays=arrays_ok,
                       passed=all(fields_ok.values()) and all(arrays_ok.values()))
            out.append(row); print(row['controller'], row['action'], row['outcome'], 'PASS' if row['passed'] else 'FAIL', flush=True)
    passed = all(r['passed'] for r in out) and all(a == b for a, b in ids.values())
    write2(S2 / 'REPLAY_CHECK_CIRCLE.json', dict(utc=utc(), passed=passed, identity=ids, cases=out))
    assert passed, 'circle identity or replay mismatch'


if __name__ == '__main__':
    main()
