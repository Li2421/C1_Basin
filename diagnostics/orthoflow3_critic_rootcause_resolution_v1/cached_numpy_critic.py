"""Small, read-only CPU forward check of existing frozen Flax checkpoints.

No training and no simulator. This reproduces the existing entity/pooling MLP,
not a new architecture. Must pass numerical agreement with saved GPU predictions
before using the cached regression results. Single-thread NumPy avoids waiting
for a GPU allocation to evaluate 16 cached feature rows.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import msgpack
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unpack(code, data):
    # Frozen float32 arrays use Flax's public on-disk ndarray extension encoding.
    if code == 1:
        shape, dtype, buffer = msgpack.unpackb(data, raw=True)
        if dtype != b'float32':
            raise ValueError(f'Unsupported checkpoint dtype: {dtype!r}')
        return np.frombuffer(buffer, dtype=np.float32).reshape(shape)
    raise ValueError(f'Unexpected checkpoint extension: {code}')


def load(path):
    return msgpack.unpackb(Path(path).read_bytes(), ext_hook=unpack, raw=False)['params']['core']


def dense(x, params):
    return x @ params['kernel'] + params['bias']


def silu(x):
    return x * expit(x)


def pool(x, mask, axis):
    mask = mask[..., None]
    total = mask.sum(axis)
    mean = (x*mask).sum(axis)/np.maximum(total, np.float32(1))
    maximum = np.where(mask > 0, x, np.float32(-1e9)).max(axis)
    maximum = np.where(total > 0, maximum, np.float32(0))
    return np.concatenate((mean, maximum), -1)


def physical(x, par):
    am = x['agent_mask']
    n = am.shape[-1]
    pm = am[:, :, None]*am[:, None, :]*(1-np.eye(n, dtype=np.float32)[None])
    pair = silu(dense(silu(dense(x['pairs'], par['pair1'])), par['pair2']))
    message = pool(pair, pm, 2)
    om = am[:, :, None]*x['obstacle_mask'][:, None, :]
    obstacles = silu(dense(silu(dense(x['obstacles'], par['obstacle1'])), par['obstacle2']))
    geometry = pool(obstacles, om, 2)
    own = silu(dense(x['agents'], par['agent1']))
    agents = silu(dense(np.concatenate((own, message, geometry), -1), par['agent2']))
    scene = pool(agents, am, 1)
    count = np.stack((np.log1p(am.sum(-1)), np.log1p(x['obstacle_mask'].sum(-1))), -1)
    return silu(dense(np.concatenate((scene, x['globals'], count), -1), par['scene']))


def forward(params, x, eta, context, kind):
    x = {key: np.asarray(value, dtype=np.float32) for key, value in x.items()}
    eta, context = np.asarray(eta, np.float32), np.asarray(context, np.float32)
    a = context[:, 24:88].reshape(len(eta), x['agents'].shape[1], 16)
    a = np.broadcast_to(a.mean(1, keepdims=True), a.shape)
    xx = {**x, 'agents': np.concatenate((x['agents'], a), -1)}
    rest = np.zeros_like(context[:, 88:104]) if kind == 'motion_only' else context[:, 88:104]
    motion = np.zeros_like(context[:, 104:]) if kind == 'rest_only' else context[:, 104:]
    cc = np.concatenate((context[:, :24], rest, motion), -1)
    h = physical(xx, params['physical_encoder'])
    e = silu(dense(eta, params['eta_encoder']))
    c = silu(dense(cc, params['context_encoder']))
    cid = silu(dense(np.zeros((len(eta), 3), np.float32), params['id_encoder']))
    z = silu(dense(np.concatenate((h, e, c, cid), -1), params['trunk1']))
    z = silu(dense(z, params['trunk2']))
    return dense(z, params['out'])[..., 0]


def regression_predictions(folder, entry):
    proto = read(folder/'protocol.json')
    assert sha(entry['checkpoint']) == entry['checkpoint_sha256']
    assert sha(entry['normalization']) == entry['normalization_sha256']
    par, norm = load(entry['checkpoint']), read(Path(entry['normalization']))
    pairs = read(folder/'pairs.json')
    with np.load(folder/'entities.npz') as values:
        indices = [p['state_index'] for p in pairs]
        x = {k: values[k][indices] for k in values}
    eta = (np.array([p['eta'] for p in pairs], np.float32)-norm['eta_center'])/norm['eta_scale']
    measured = [dict(np.load(folder/f'inputs_{p["name"]}.npz')) for p in proto['profiles']]
    results = []
    for condition in ('parent', 'correct_motion', 'wrong_motion'):
        a, b = measured[0], measured[1] if condition == 'correct_motion' else measured[0]
        c = np.concatenate(((a['context']-norm['context_center'])/norm['context_scale'],
                            ((a['agent_response']-norm['agent_center'])/norm['agent_scale']).reshape(len(pairs), -1),
                            (a['goal_response']-norm['goal_center'])/norm['goal_scale'],
                            (b['goal_motion_response']-norm['goal_motion_center'])/norm['goal_motion_scale']), -1)
        results.append(forward(par, x, eta, c, entry['kind']))
    z = np.array(results)
    assert np.array_equal(z[0], z[2])
    if entry['kind'] == 'rest_only':
        assert np.array_equal(z[0], z[1])
    return z


def run():
    start = perf_counter()
    old = ROOT/'motion_counterexample_regression'
    dest = ROOT/'motion_drop_rest_regression'
    output = dest/'numpy_check'
    output.mkdir(exist_ok=True)
    expected = np.load(old/'predictions.npz')
    checks = []
    for entry in read(old/'protocol.json')['models']:
        z = regression_predictions(old, entry)
        key = f"{entry['arm']}__fold{entry['fold']}__seed{entry['seed']}"
        delta = abs(z-expected[key])
        assert np.allclose(z, expected[key], atol=2e-5, rtol=2e-5), (key, delta.max())
        assert np.array_equal(z.reshape(3, 8, 2).argmax(-1), expected[key].reshape(3, 8, 2).argmax(-1))
        checks.append(dict(model=key, max_abs_logit_error=float(delta.max()), top1_exact=True))
    # Cover the motion-only mask too, using all cached source-VAL controllers
    # and the original GPU predictions of the exact source-selected checkpoint.
    matrix = dict(np.load(ROOT/'motion_factorial_support/dataset.npz'))
    source_entities = dict(np.load(ROOT/'controller_function_support/frozen_model_entities.npz'))
    val = np.flatnonzero(matrix['split'] == 'validation')
    source_x = {k: v[matrix['state_index'][val]] for k, v in source_entities.items()}
    checked_source_predictions = 0
    for entry in read(dest/'protocol.json')['models']:
        par, norm = load(entry['checkpoint']), read(Path(entry['normalization']))
        eta = (matrix['eta'][val]-norm['eta_center'])/norm['eta_scale']
        target = np.load(Path(entry['path'])/'predictions.npz')[f"step{entry['step']}"]
        predicted = []
        for controller in range(16):
            context = np.concatenate(((matrix['context'][controller, val]-norm['context_center'])/norm['context_scale'],
                         ((matrix['agent_response'][controller, val]-norm['agent_center'])/norm['agent_scale']).reshape(len(val), -1),
                         (matrix['goal_response'][controller, val]-norm['goal_center'])/norm['goal_scale'],
                         (matrix['goal_motion_response'][controller, val]-norm['goal_motion_center'])/norm['goal_motion_scale']), -1)
            predicted.append(forward(par, source_x, eta, context, 'motion_only'))
        predicted = np.array(predicted)
        delta = abs(predicted-target)
        assert np.allclose(predicted, target, atol=2e-5, rtol=2e-5), (entry, delta.max())
        assert np.array_equal(predicted.reshape(16, 16, 2).argmax(-1), target.reshape(16, 16, 2).argmax(-1))
        checked_source_predictions += target.size
        checks.append(dict(model=f"source_motion_only_fold{entry['fold']}_seed{entry['seed']}",
                           max_abs_logit_error=float(delta.max()), top1_exact=True))
    # Frozen labels already read by original regression. No new DB access needed.
    with (old/'per_pair_predictions.csv').open() as handle:
        counts = {}
        for row in csv.DictReader(handle):
            counts[row['state_uid'], row['eta_uid']] = [int(row['parent_success']), int(row['motion_success'])]
    pairs = read(dest/'pairs.json')
    success = np.array([counts[p['state_uid'], p['eta_uid']] for p in pairs]).T
    q, good = success/16, (success >= 15).reshape(2, 8, 2)
    ci, hi = np.arange(2)[:, None], np.arange(8)[None, :]
    rows, predictions = [], {}
    for entry in read(dest/'protocol.json')['models']:
        z = regression_predictions(dest, entry)
        p = expit(z)
        chosen = z[:2].reshape(2, 8, 2).argmax(-1)
        key = f"drop_rest__fold{entry['fold']}__seed{entry['seed']}"
        predictions[key] = z
        dq, dp = q[1]-q[0], p[1]-p[0]
        rows.append(dict(fold=entry['fold'], seed=entry['seed'], constituents_seen=entry['fold']!=2,
            NLL=float((q*np.logaddexp(0, -z[:2])+(1-q)*np.logaddexp(0, z[:2])).mean()),
            B15=int(good[ci, hi, chosen].sum()), oracle_B15=int(good.any(-1).sum()), cases=16,
            causal_delta_MAE=float(abs(dp-dq).mean()), causal_delta_correlation=float(np.corrcoef(dp, dq)[0, 1]),
            example_true_parent=float(q[0, 1]), example_true_motion=float(q[1, 1]),
            example_pred_parent=float(p[0, 1]), example_pred_motion=float(p[1, 1]), example_pred_delta=float(dp[1]),
            correct_motion_NLL=float((q[1]*np.logaddexp(0, -z[1])+(1-q[1])*np.logaddexp(0, z[1])).mean()),
            wrong_motion_NLL=float((q[1]*np.logaddexp(0, -z[2])+(1-q[1])*np.logaddexp(0, z[2])).mean())))
    with (output/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output/'predictions.npz', **predictions)
    audit = dict(implementation='Single-thread float32 NumPy read-only forward; no JAX/GPU/training',
                 GPU_reference_models=len(checks), GPU_reference_predictions=sum(v.size for v in expected.values())+checked_source_predictions,
                 max_logit_error=max(c['max_abs_logit_error'] for c in checks), all_top1_equal=True,
                 checks=checks, new_rollouts=0, new_fits=0, prior_model_selection_unchanged=True,
                 scope='Cached posthoc regression only. All three input-mask variants verified against saved GPU predictions. No new checkpoint or feature selection.',
                 seconds=perf_counter()-start, code_sha256=sha(__file__))
    (output/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(seconds=audit['seconds'], max_logit_error=audit['max_logit_error'], results=rows), indent=2))


if __name__ == '__main__':
    run()
