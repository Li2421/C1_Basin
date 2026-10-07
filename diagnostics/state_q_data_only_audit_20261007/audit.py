"""Read-only, data-only state/input versus continuation-Q diagnostics.

No predictor, learned embedding, checkpoint, training, or rollout is used.
This is retrospective analysis of previously opened matched Q16 cohorts.
"""
from pathlib import Path
from itertools import permutations, combinations
import hashlib
import json
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'orthoflow3_tt_ff_matched_learnability_v1'
CHAINS = ('TT', 'FF')
PERMUTATIONS = 2000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_distances(entities, indices):
    """Physical-unit feature distance, minimized over agent permutations.

    Each feature block has equal weight. Fixed within-scene obstacle identities
    are retained; all agent and obstacle padding is removed. The same agent
    permutation is applied to own, pair, and obstacle features.
    """
    idx = np.asarray(indices)
    am = entities['agent_mask'][idx]
    om = entities['obstacle_mask'][idx]
    assert np.all(am == am[0]) and np.all(om == om[0])
    na, no = int(am[0].sum()), int(om[0].sum())
    a = entities['agents'][idx, :na].astype(float)
    p = entities['pairs'][idx, :na, :na].astype(float)
    o = entities['obstacles'][idx, :na, :no].astype(float)
    g = entities['globals'][idx].astype(float)
    pm = ~np.eye(na, dtype=bool)
    out = np.zeros((len(idx), len(idx)))
    for i, j in combinations(range(len(idx)), 2):
        values = []
        for perm in permutations(range(na)):
            perm = np.asarray(perm)
            pp = p[j][perm][:, perm]
            blocks = [np.mean((a[i] - a[j][perm]) ** 2),
                      np.mean((p[i][pm] - pp[pm]) ** 2),
                      np.mean((g[i] - g[j]) ** 2)]
            if no:
                blocks.append(np.mean((o[i] - o[j][perm]) ** 2))
            values.append(np.mean(blocks))
        out[i, j] = out[j, i] = np.sqrt(min(values))
    return out


def q_distances(q):
    """Compare matched eta columns, separating level from curve shape.

    Subtracting the mean pairwise Q difference removes pure state difficulty.
    A shared eta preference cancels automatically in the state difference.
    """
    n = len(q)
    raw = np.zeros((n, n))
    shape = np.zeros((n, n))
    common = np.zeros((n, n), dtype=int)
    for i, j in combinations(range(n), 2):
        ok = np.isfinite(q[i]) & np.isfinite(q[j])
        d = q[i, ok] - q[j, ok]
        assert len(d) >= 2
        raw[i, j] = raw[j, i] = np.sqrt(np.mean(d ** 2))
        shape[i, j] = shape[j, i] = np.std(d)
        common[i, j] = common[j, i] = len(d)
    return raw, shape, common


def normalized_rank(x):
    r = rankdata(x)
    r -= r.mean()
    norm = np.linalg.norm(r)
    return r / norm if norm else np.zeros_like(r)


def association(dx, dy, state_permutations):
    n = len(dx)
    tri = np.triu_indices(n, 1)
    rx = normalized_rank(dx[tri])
    rho = float(rx @ normalized_rank(dy[tri]))
    null = np.array([rx @ normalized_rank(dy[p][:, p][tri])
                     for p in state_permutations])
    near = dx.copy()
    np.fill_diagonal(near, np.inf)
    nn = near.argmin(axis=1)
    nearest = float(dy[np.arange(n), nn].mean())
    random = float(dy[tri].mean())
    return dict(spearman=rho,
                state_permutation_p_two_sided=float((1 + (np.abs(null) >= abs(rho)).sum()) / (len(null) + 1)),
                nearest_state_mean_Q_curve_shape_distance=nearest,
                random_other_state_expected_Q_curve_shape_distance=random,
                nearest_vs_random_reduction=1 - nearest / random if random else None,
                nearest_state_indices=nn.tolist())


def residuals(q, ok):
    rows, cols = np.nonzero(ok)
    x = np.concatenate((np.eye(q.shape[0])[rows], np.eye(q.shape[1])[cols]), axis=1)
    y = q[ok]
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def reliability(outcomes):
    ok = ~(outcomes[..., 1].astype(bool).any(axis=-1))
    halves = [outcomes[:, :, a:b, 0].mean(axis=-1) for a, b in ((0, 8), (8, 16))]
    r = [residuals(q, ok) for q in halves]
    covariance = float(np.mean(r[0] * r[1]))
    mean_residual_var = float(np.mean(((r[0] + r[1]) / 2) ** 2))
    return dict(complete_cells=int(ok.sum()),
                split_half_interaction_residual_correlation=float(np.corrcoef(*r)[0, 1]),
                cross_half_interaction_covariance=covariance,
                reliable_fraction_estimate=covariance / mean_residual_var if mean_residual_var else None,
                note='Two disjoint eight-seed halves; state and eta main effects removed descriptively. This measures label repeatability, not input sufficiency.')


def reversals(q, uids, eta):
    strong = 0
    pairs = set()
    example = None
    for i, j in combinations(range(len(q)), 2):
        for a, b in combinations(range(q.shape[1]), 2):
            values = np.array([q[i, a], q[i, b], q[j, a], q[j, b]])
            if not np.isfinite(values).all():
                continue
            d1, d2 = values[0] - values[1], values[2] - values[3]
            if d1 * d2 < 0 and min(abs(d1), abs(d2)) >= .25:
                strong += 1
                pairs.add((i, j))
            robust_reverse = ((values[0] >= 15/16 and values[1] <= .5 and values[2] <= .5 and values[3] >= 15/16)
                              or (values[1] >= 15/16 and values[0] <= .5 and values[3] <= .5 and values[2] >= 15/16))
            if example is None and robust_reverse:
                example = dict(states=[uids[i], uids[j]], eta_indices=[a, b],
                               eta=[eta[a], eta[b]], Q16=values.reshape(2, 2).tolist())
    return dict(strong_gap_025_quadruples=strong, state_pairs_with_strong_reversal=len(pairs),
                total_state_pairs=len(q) * (len(q) - 1) // 2,
                example_first_lexicographic_robust_vs_low_Q=example,
                note='Quadruples share states and eta; they are not independent trials. Q16 is empirical success frequency.')


def main():
    data = np.load(SOURCE / 'dataset.npz')
    entities = np.load(SOURCE / 'entities.npz')
    truth = np.load(SOURCE / 'test_truth.npz')
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    output = dict(method='No-training retrospective matched state-input/Q audit',
                  protocol=dict(states='All already-opened TEST states in the matched TT/FF panel, 16 per scene',
                                eta='Same exact 16-candidate bank for every state and formulation',
                                label='Empirical success fraction over 16 future seeds; numerical failures stay unknown',
                                comparisons='Within scene and fixed controller/formulation; exact same eta columns',
                                input_h='Unlearned physical-unit entity tensors; minimum over agent permutations',
                                input_context='Existing 76 response channels, frozen TRAIN normalization; same eta columns',
                                Q_shape='RMS of state-pair Q differences after removing their mean across common exact eta',
                                permutation='Permute whole physical-state Q profiles, never individual cells',
                                permutation_count=PERMUTATIONS,
                                caveat='Exploratory metric-specific association; no claim of generic learnability or extrapolation'),
                  source_hashes={name: sha(SOURCE / name) for name in ['dataset.npz', 'entities.npz', 'test_truth.npz', 'protocol.json']},
                  script_sha256=sha(Path(__file__)), cohorts={})
    for scene in ('toy_give_way', 'ring_exchange'):
        loc = np.flatnonzero(data['scene'][truth['indices']] == scene)
        indices = truth['indices'][loc]
        assert len(indices) == 16 and np.all(data['split'][indices] == 'test')
        uids = [protocol['states'][int(i)]['uid'] for i in indices]
        families = [protocol['states'][int(i)]['family'] for i in indices]
        assert len(set(families)) == len(indices)
        dh = physical_distances(entities, indices)
        rng = np.random.default_rng(202610071)
        perms = [rng.permutation(len(indices)) for _ in range(PERMUTATIONS)]
        for ci, chain in enumerate(CHAINS):
            outcome = truth['outcomes'][ci, loc]
            q = outcome[..., 0].mean(axis=-1).astype(float)
            unknown = outcome[..., 1].sum(axis=-1)
            q[unknown > 0] = np.nan
            raw, shape, common = q_distances(q)
            context = data['context'][ci, indices].astype(float)
            dc = np.sqrt(np.mean((context[:, None] - context[None, :]) ** 2, axis=(2, 3)))
            key = f'{scene}:{chain}'
            output['cohorts'][key] = dict(
                states=len(indices), eta=len(protocol['eta']), exact_Q16_cells=int(np.isfinite(q).sum()),
                numerical_unresolved_seed_count=int(unknown.sum()),
                common_exact_eta_per_state_pair_min=int(common[np.triu_indices(len(indices), 1)].min()),
                state_uids=uids, Q16=[[None if not np.isfinite(x) else float(x) for x in row] for row in q],
                input_h_vs_Q_shape=association(dh, shape, perms),
                response_context_vs_Q_shape=association(dc, shape, perms),
                label_repeatability=reliability(outcome),
                direct_ranking_reversals=reversals(q, uids, protocol['eta']))
    # Holm correction for the eight pre-defined input-distance association tests.
    tests = [c[k] for c in output['cohorts'].values() for k in ['input_h_vs_Q_shape', 'response_context_vs_Q_shape']]
    order = sorted(range(len(tests)), key=lambda i: tests[i]['state_permutation_p_two_sided'])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(tests)-rank) * tests[i]['state_permutation_p_two_sided']))
        tests[i]['holm_p_eight_association_tests'] = running
    (HERE / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for key, c in output['cohorts'].items():
        print(key)
        for k in ['input_h_vs_Q_shape', 'response_context_vs_Q_shape']:
            print(k, {a: b for a, b in c[k].items() if a != 'nearest_state_indices'})
        print('repeatability', c['label_repeatability'])
        print('reversals', c['direct_ranking_reversals'])


if __name__ == '__main__':
    main()
