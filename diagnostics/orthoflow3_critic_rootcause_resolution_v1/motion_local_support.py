"""Non-neural causal support check using only cached, compatible count pairs.

This is a known-controller diagnostic, not a transferable selector. k/metric are
selected on source VAL only. The already-opened alias cases are a posthoc
regression set and are never used to select k. No new simulator calls or labels.
"""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'motion_local_support'
KS = (1, 3, 5, 10, 20)
CONTROLLERS = (0, 14)


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]));writer.writeheader();writer.writerows(rows)


def features():
    d = dict(np.load(ROOT/'motion_factorial_support/dataset.npz'))
    x = dict(np.load(ROOT/'controller_function_support/frozen_model_entities.npz'))
    tx = dict(np.load(ROOT/'motion_counterexample_regression/entities.npz'))
    tr, va = [np.flatnonzero(d['split']==v) for v in ('train', 'validation')]
    states = np.unique(d['state_index'][tr])
    source_h, target_h = [], []
    # Work in the exact fixed physical coordinates consumed by the critic.
    # Group dimension normalization prevents high-dimensional pair geometry
    # from automatically dominating globals/agent information.
    for key in ('agents', 'pairs', 'obstacles', 'globals'):
        a = x[key].reshape(len(x[key]), -1)
        b = tx[key].reshape(len(tx[key]), -1)
        c, scale = a[states].mean(0), np.maximum(a[states].std(0), .05)
        source_h.append((a-c)/scale/np.sqrt(a.shape[1]))
        target_h.append((b-c)/scale/np.sqrt(a.shape[1]))
    hs, ht = np.concatenate(source_h, -1)/2, np.concatenate(target_h, -1)/2
    folder = ROOT/'motion_counterexample_regression'
    pp = read(folder/'pairs.json')
    measured = [dict(np.load(folder/f'inputs_{p["name"]}.npz')) for p in read(folder/'protocol.json')['profiles']]
    source_c, target_c = [], []
    for key in ('context', 'agent_response', 'goal_response', 'goal_motion_response'):
        values = d[key].reshape(16, 160, -1)
        c, scale = values[:12, tr].mean((0, 1)), np.maximum(values[:12, tr].std((0, 1)), .05)
        source_c.append((values[list(CONTROLLERS)]-c)/scale/np.sqrt(values.shape[-1]))
        a = measured[0][key].reshape(16, -1)
        b = measured[1 if key=='goal_motion_response' else 0][key].reshape(16, -1)
        target_c.append((np.stack((a, b))-c)/scale/np.sqrt(values.shape[-1]))
    cs = np.concatenate(source_c, -1).transpose(1, 0, 2).reshape(160, -1)/np.sqrt(8)
    ct = np.concatenate(target_c, -1).transpose(1, 0, 2).reshape(16, -1)/np.sqrt(8)
    source_state = hs[d['state_index']]
    target_state = ht[np.array([p['state_index'] for p in pp])]
    feature_pairs = {
        'h_only': (source_state, target_state),
        'h_plus_response': (np.concatenate((source_state, cs), -1)/np.sqrt(2),
                            np.concatenate((target_state, ct), -1)/np.sqrt(2))}
    return d, tr, va, pp, feature_pairs


def predict(source, query, train_indices, query_eta, train_eta, s, f, k):
    predictions, neighbors, distances = [], [], []
    for row, eta in zip(query, query_eta):
        candidates = train_indices[train_eta[train_indices] == eta]
        distance = np.linalg.norm(source[candidates]-row, axis=-1)
        order = np.argsort(distance, kind='stable')[:k]
        ii = candidates[order]
        # Jeffreys half-count prevents infinite local NLL with Q4 observations.
        predictions.append((s[:, ii].sum(1)+.5)/(s[:, ii].sum(1)+f[:, ii].sum(1)+1))
        neighbors.append(ii); distances.append(distance[order])
    return np.array(predictions).T, neighbors, distances


def metrics(p, s, f):
    q = s/(s+f)
    dp, dq = p[1]-p[0], q[1]-q[0]
    return dict(NLL=float((-q*np.log(p)-(1-q)*np.log1p(-p)).mean()),
                MAE=float(abs(p-q).mean()), causal_MAE=float(abs(dp-dq).mean()),
                causal_correlation=float(np.corrcoef(dp, dq)[0, 1]) if dp.std()>1e-10 else None,
                strong_cases=int((abs(dq)>=.25).sum()),
                strong_sign_correct=int(((np.sign(dp)==np.sign(dq)) & (abs(dq)>=.25)).sum()))


def main():
    OUT.mkdir(exist_ok=True)
    d, tr, va, target_pairs, spaces = features()
    s, f = [d[k][list(CONTROLLERS)] for k in ('success', 'failure')]
    assert np.all(s[:, tr]+f[:, tr]>0)
    target_eta = np.array([p['eta_index'] for p in target_pairs])
    source_pairs = read(ROOT/'controller_function_support/pairs.json')
    source_uids = {source_pairs[i]['state_uid'] for i in tr}
    assert not source_uids & {p['state_uid'] for p in target_pairs}
    counts = {}
    with (ROOT/'motion_counterexample_regression/per_pair_predictions.csv').open() as handle:
        for row in csv.DictReader(handle):
            counts[row['state_uid'], row['eta_uid']] = [int(row['parent_success']), int(row['motion_success'])]
    ts = np.array([counts[p['state_uid'], p['eta_uid']] for p in target_pairs]).T
    tf = 16-ts
    val_rows, regression_rows, neighbor_rows = [], [], []
    for space, (source, target) in spaces.items():
        for k in KS:
            p, _, _ = predict(source, source[va], tr, d['eta_index'][va], d['eta_index'], s, f, k)
            val_rows.append(dict(space=space, k=k, **metrics(p, s[:, va], f[:, va])))
    chosen = min(val_rows, key=lambda r:r['NLL'])
    for space, (source, target) in spaces.items():
        for k in KS:
            p, nn, dist = predict(source, target, tr, target_eta, d['eta_index'], s, f, k)
            regression_rows.append(dict(space=space, k=k, source_selected=space==chosen['space'] and k==chosen['k'],
                **metrics(p, ts, tf), example_parent_prediction=float(p[0, 1]), example_motion_prediction=float(p[1, 1]),
                example_true_delta=.625, example_predicted_delta=float(p[1, 1]-p[0, 1])))
            if k != max(KS):
                continue
            for i, (ids, ds) in enumerate(zip(nn, dist)):
                for rank, (j, distance) in enumerate(zip(ids, ds), 1):
                    neighbor_rows.append(dict(space=space, query_state_uid=target_pairs[i]['state_uid'], query_eta_uid=target_pairs[i]['eta_uid'],
                        example=(i==1), rank=rank, TRAIN_state_uid=source_pairs[j]['state_uid'], normalized_group_distance=float(distance),
                        parent_s=int(s[0, j]), parent_f=int(f[0, j]), motion_s=int(s[1, j]), motion_f=int(f[1, j]),
                        observed_delta=float(s[1, j]/(s[1, j]+f[1, j])-s[0, j]/(s[0, j]+f[0, j]))))
    save_csv(OUT/'source_VAL.csv', val_rows)
    save_csv(OUT/'posthoc_regression.csv', regression_rows)
    save_csv(OUT/'nearest_TRAIN.csv', neighbor_rows)
    write(OUT/'audit.json', dict(selected_by_source_VAL=chosen, new_rollouts=0, new_neural_fits=0,
        TRAIN_families=64, source_VAL_families=16, regression_families=8, state_uid_overlap=0,
        scope='Known-controller causal kNN diagnostic; not zero-shot, unseen-eta, or independent confirmation. Regression labels previously opened.',
        normalization='Source TRAIN features only;per feature std floor0.05;group dimension normalized;state/context equally weighted.',
        estimation='Pooled observed Bernoulli counts within k same-exact-eta TRAIN neighbors;Jeffreys half-count smoothing;no missing-seed imputation.',
        limitations='Nearby feature disagreement alone does not prove aliasing or insufficiency. Q4 neighbor effects can be noisy. No hyperparameter selected on regression labels.',
        exact_controller_indices=list(CONTROLLERS)))
    print(json.dumps(dict(source_selected=chosen, regression=regression_rows), indent=2))


if __name__ == '__main__':
    main()
