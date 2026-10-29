"""User-authorized computational triage; never a mathematical failure claim."""
from pathlib import Path
import datetime
import json

HERE = Path(__file__).resolve().parent


def review(state):
    path = HERE / 'USER_EARLY_STOP_POLICY.json'
    if not path.exists() or state['scope'] != 'screen' or not state['code'].startswith(('level3_', 'level2_')):
        return None
    policy = json.loads(path.read_text())
    reference = json.loads((HERE / 'SEARCH_four_2_screen.json').read_text())
    incumbent = {tuple(g['identity']): g for g in reference['groups']}
    groups = state['groups']
    if state.get('user_early_stop'):
        return state['user_early_stop']
    # Complete frontiers only: successful cases have intentionally stopped.
    available = min((g['searched'] for g in groups if not g['success']), default=64)
    if available >= 16:
        return None  # Keep the original complete16 screening decision.
    if available < 8:
        return None
    count = lambda rows, k: sum(g['success'] and g['first_success'] <= k for g in rows)
    gains = [g['identity'] for g in groups if g['success'] and not incumbent[tuple(g['identity'])]['success']]
    # Retain every observed improvement on an incumbent failure, including the
    # warehouse cases on which all completed families are weak.
    decision = dict(time=datetime.datetime.now().astimezone().isoformat(), code=state['code'],
                    incumbent='four_2', available_complete_budget=available,
                    coverage8=count(groups, 8), incumbent8=count(reference['groups'], 8),
                    new_successes_on_incumbent_failures=gains,
                    policy_file='USER_EARLY_STOP_POLICY.json', prune=False)
    if available >= 12:
        decision.update(coverage12=count(groups, 12), incumbent12=count(reference['groups'], 12))
    if not gains:
        lag8 = decision['incumbent8'] - decision['coverage8']
        if available >= 12:
            lag12 = decision['incumbent12'] - decision['coverage12']
            decision['prune'] = lag8 >= policy['lag_both_8_and_12'] and lag12 >= policy['lag_both_8_and_12']
        else:
            decision['prune'] = lag8 >= policy['lag_at_8']
    folder = HERE / 'value_reviews'
    folder.mkdir(exist_ok=True)
    (folder / (state['code'] + f'_k{available:02d}.json')).write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n')
    if not decision['prune']:
        return None
    decision.update(reason='用户授权：同预算明显落后且没有对当前最好族失败案例的新增成功；提前停止以节省计算。',
                    unfinished_candidates_are_not_failures=True)
    return decision
