"""EXPLORATORY (zero rollouts): can three source scenes determine the held scene's eta regime?

For each LOSO fold and each physically meaningful scene descriptor, a 1-D threshold rule is fitted to the
three source scenes' regimes (best K4 eta for Flow-reference programs, from controller_ff_factorial_repair_v1).
Counts descriptors whose consistent rule predicts the held scene correctly / wrongly / ambiguously
(held value inside the source gap, so the threshold location decides). Output: scene_multiplicity.json.
"""
import itertools, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
FACT = HERE.parent / 'controller_ff_factorial_repair_v1'
S = ['toy_give_way', 'double_bottleneck', 'four_way_intersection', 'ring_exchange']
REGIME = {'toy_give_way': 10, 'double_bottleneck': 15, 'four_way_intersection': 15, 'ring_exchange': 10}


def descriptors():
    p = json.loads((FACT / 'protocol.json').read_text()); z = np.load(FACT / 'dataset.npz', allow_pickle=True)
    cfg = {sc: (e['config']['config'] if 'config' in e['config'] else e['config']) for sc, e in p['environments'].items()}
    out = {}
    for sc in S:
        st = [s for s in p['states'] if s['scenario'] == sc]; c = cfg[sc]
        n = len(st[0]['physical']['positions']); T = c['max_steps'] * c['dt']
        L = float(np.mean([np.linalg.norm(np.array(s['physical']['goals']) - np.array(s['physical']['positions']), axis=1).mean() for s in st]))
        pd = float(np.mean([np.mean([np.linalg.norm(np.array(s['physical']['positions'])[i] - np.array(s['physical']['positions'])[j])
                                     for i, j in itertools.combinations(range(n), 2)]) for s in st]))
        nobs = int(z['entity_obstacle_mask'][z['scene'] == sc][0].sum())
        out[sc] = dict(n_agents=n, agent_radius=c['agent_radius'], max_speed=c['max_speed'], horizon_s=T, max_travel=T * c['max_speed'],
                       start_goal_dist=L, time_slack=T * c['max_speed'] / L, mean_pair_dist=pd, pair_dist_over_diam=pd / (2 * c['agent_radius']),
                       n_obstacle_entities=nobs, deadlock_monitor=float('progress_window_seconds' in c), goal_tolerance=c['goal_tolerance'],
                       loop_topology=float(sc in ('ring_exchange', 'four_way_intersection')), agents_per_obstacle=n / max(nobs, 1),
                       speed_over_radius=c['max_speed'] / c['agent_radius'])
    return out


def main():
    D = descriptors(); res = {}
    for h in S:
        src = [s for s in S if s != h]; row = dict(correct=[], wrong=[], ambiguous=[], sources_not_separable=[])
        for k in D[h]:
            a = [D[s][k] for s in src if REGIME[s] == 10]; b = [D[s][k] for s in src if REGIME[s] == 15]; x = D[h][k]
            if max(a) < min(b): lo, hi, gap = 10, 15, (max(a), min(b))
            elif max(b) < min(a): lo, hi, gap = 15, 10, (max(b), min(a))
            else: row['sources_not_separable'].append(k); continue
            if gap[0] < x < gap[1]: row['ambiguous'].append(k); continue
            row['correct' if (lo if x <= gap[0] else hi) == REGIME[h] else 'wrong'].append(k)
        res[h] = row
    always = [k for k in D[S[0]] if all(k in res[h]['correct'] for h in S)]
    out = dict(label='EXPLORATORY', descriptors=D, regime=REGIME, folds=res, correct_in_all_folds=always,
               totals={t: sum(len(res[h][t]) for h in S) for t in ('correct', 'wrong', 'ambiguous', 'sources_not_separable')})
    (HERE / 'scene_multiplicity.json').write_text(json.dumps(out, indent=2) + '\n'); print(json.dumps(out['totals']), always)


if __name__ == '__main__':
    main()
