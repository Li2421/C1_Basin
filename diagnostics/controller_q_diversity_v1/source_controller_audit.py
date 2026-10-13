"""TRAIN-only crossed-controller evidence; no new simulation or target labels."""
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import json
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    requests = json.loads((HERE / 'batches/source_q1/cases.json').read_text())
    requests = [r for r in requests if r['split'] == 'train' and
                r['state_index'] < 64 and r['controller_name'].startswith('analytic')]
    lookup = {}
    with sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for r in requests:
            old = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (r['rollout_uid'],)).fetchone()
            if old is not None:
                assert not old['conflict_quarantined']
                lookup[r['rollout_uid']] = dict(old)
            else:
                path = HERE / 'batches/source_q1/raw' / (r['rollout_uid'] + '.json')
                record = json.loads(path.read_text())
                assert record['execution_semantics_version'] == 'scene_precision_guard_v1'
                assert record['jax_enable_x64'] == protocol['controllers'][r['scene']][r['controller_name']]['payload']['jax_enable_x64']
                lookup[r['rollout_uid']] = record
        table = {(r['scene'], r['state_index'], r['controller_name'], r['eta_index']):
                 lookup[r['rollout_uid']] for r in requests}
        summaries = []
        for scene in sorted({r['scene'] for r in requests}):
            controllers = {}
            for ci in range(4):
                name = f'analytic{ci}'
                groups = [[table[scene, s, name, j] for s in range(64)] for j in range(16)]
                succ = [sum(r['success'] for r in g) for g in groups]
                unknown = [sum(r['numerical_failure'] for r in g) for g in groups]
                controllers[name] = dict(physical_states=64, attempts=1024,
                    verified_success=sum(succ), numerical=sum(unknown),
                    collision=sum(r['collision'] for g in groups for r in g),
                    eta_success_counts=succ, eta_unknown_counts=unknown,
                    best_eta_lowest_index_tie=max(range(16), key=lambda j: succ[j]),
                    known_eta_varying_states=sum(len({table[scene, s, name, j]['success'] for j in range(16)
                        if not table[scene, s, name, j]['numerical_failure']}) > 1 for s in range(64)),
                    oracle_verified_states=sum(any(table[scene, s, name, j]['success'] for j in range(16)) for s in range(64)),
                    best_fixed_verified_states=max(succ))
            contrasts = []
            for a, b in combinations(range(4), 2):
                counts = Counter(); reversals = []
                for s in range(64):
                    signs = []
                    for j in range(16):
                        x = table[scene, s, f'analytic{a}', j]; y = table[scene, s, f'analytic{b}', j]
                        known = not (x['numerical_failure'] or y['numerical_failure'])
                        counts['known_matched_state_eta'] += known
                        counts['opposite_outcomes'] += known and x['success'] != y['success']
                    for i, j in combinations(range(16), 2):
                        x = [table[scene, s, f'analytic{c}', k] for c, k in [(a, i), (a, j), (b, i), (b, j)]]
                        if not any(r['numerical_failure'] for r in x):
                            d1 = x[0]['success'] - x[1]['success']; d2 = x[2]['success'] - x[3]['success']
                            counts['known_state_eta_pair_cases'] += 1
                            if d1 * d2 == -1:
                                counts['controller_induced_eta_ranking_reversals_Q1'] += 1
                                signs.append((i, j))
                                if len(reversals) < 3:
                                    reversals.append(dict(state_index=s, eta_a=i, eta_b=j,
                                        outcomes=[int(r['success']) for r in x]))
                    counts['physical_states_with_controller_induced_reversal'] += bool(signs)
                contrasts.append(dict(controllers=[a, b], **counts, examples=reversals))
            errors=[];lower=[];ambiguous=0
            for s in range(64):
                for j in range(16):
                    values=[table[scene,s,f'analytic{ci}',j] for ci in range(4)]
                    ys=[r['success'] for r in values if not r['numerical_failure']]
                    k=len(ys);u=4-k;ones=sum(ys)
                    ambiguous+=len(set(ys))>1
                    if not u:errors.append((ones/4)*(1-ones/4))
                    lower.append(min(((ones+z)/4)*(1-(ones+z)/4) for z in range(u+1)))
            summaries.append(dict(scene=scene, controllers=controllers, contrasts=contrasts,
                same_input_conflicting_known_labels=ambiguous,
                empirical_seed0_state_eta_only_MSE_floor_complete_cells=sum(errors)/len(errors) if errors else None,
                empirical_seed0_MSE_floor_unknown_worst=sum(lower)/len(lower),
                floor_caveat='Finite TRAIN seed0 contradiction bound for four equally weighted controllers; not a population Bayes-error estimate.'))
        # Independent seed1 evidence exists only in preserved early Q2 TRAIN.
        replication = defaultdict(Counter)
        reqmap = {(r['scene'], r['state_index'], r['controller_name'], r['eta_index']): r for r in requests}
        seed1 = {}
        for key, r in reqmap.items():
            sk = json.loads(r['seed_key']); sk['future_index'] = 1
            value = db.execute('SELECT success,numerical_failure FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=? AND conflict_quarantined=0',
                (r['state_uid'], r['eta_uid'], r['controller_uid'], json.dumps(sk, sort_keys=True, separators=(',', ':')))).fetchone()
            if value is not None: seed1[key] = dict(value)
        for scene in sorted({r['scene'] for r in requests}):
            for s in range(64):
                for a, b in combinations(range(4), 2):
                    for j in range(16):
                        keys = [(scene, s, f'analytic{c}', j) for c in (a, b)]
                        if not all(k in seed1 for k in keys): continue
                        x = [table[k] for k in keys]; y = [seed1[k] for k in keys]
                        if any(r['numerical_failure'] for r in x+y): continue
                        if x[0]['success'] != x[1]['success']:
                            replication[scene]['seed0_controller_outcome_changes_with_independent_seed1'] += 1
                            replication[scene]['same_outcome_change_on_seed1'] += all(x[k]['success'] == y[k]['success'] for k in (0, 1))
                    for i, j in combinations(range(16), 2):
                        keys = [(scene, s, f'analytic{c}', k) for c, k in [(a, i), (a, j), (b, i), (b, j)]]
                        if not all(k in seed1 for k in keys): continue
                        x = [table[k] for k in keys]; y = [seed1[k] for k in keys]
                        if any(r['numerical_failure'] for r in x + y): continue
                        d0 = [x[0]['success'] - x[1]['success'], x[2]['success'] - x[3]['success']]
                        d1 = [y[0]['success'] - y[1]['success'], y[2]['success'] - y[3]['success']]
                        if d0[0] * d0[1] == -1:
                            replication[scene]['seed0_reversal_cases_with_independent_seed1'] += 1
                            replication[scene]['same_reversal_on_seed1'] += d0 == d1
    # Exact unprojected FF counterexample: same nominal terminal response,
    # different intervention gain. This is algebra, not rollout evidence.
    counterexample = []
    for lam in (2., 4.):
        n = 10; contraction = (1-lam/n)**n
        target = .05/(1-contraction); eta_gain = (1-contraction)/lam
        h = 0.; largest_safety_target = 0.
        for k in range(n):
            field = lam*(target-h)+.02
            safety_target = h+(1-k/n)*field
            largest_safety_target = max(largest_safety_target,abs(safety_target))
            assert abs(safety_target) < .5
            h += field/n
        assert abs(h-(.05+.02*eta_gain)) < 1e-12
        counterexample.append(dict(relaxation=lam, target=target,
            nominal_terminal=.05, correction=.02, correction_gain=eta_gain,
            intervened_terminal=h,max_abs_projection_target=largest_safety_target,
            speed_only_safety_projection_inactive=True))
    result = dict(time_local=datetime.now().astimezone().isoformat(),
        source='First64 originally frozen TRAIN physical states per scene; all four TRAIN analytic controllers; fixed seed0',
        no_final_or_validation_labels=True, new_rollouts=0, summaries=summaries,
        seed1_reversal_replication=dict(replication),
        limitations='TRAIN evidence of label structure, not critic generalization. Q1 reversal counts are descriptive; independent Q1 replication uses only available prior Q2 TRAIN seeds. Controller pairs are dependent.',
        nominal_alias_counterexample=counterexample,
        counterexample_scope='Linear deterministic FF without active safety projection, ten native pseudo-time steps; proves endpoint alone is not a sufficient universal FF descriptor, not insufficiency of all seven current probes.')
    (HERE/'source_controller_structure.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
