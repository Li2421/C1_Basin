"""Formal Ring FF Q(s,eta) / Q(eta) fits on identical frozen Q4 labels.

Only the physically encoded state and eta enter the state critic. This file
reads the isolated TRAIN/VAL artifact; no old or future TEST labels are loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'ring_ff_train_plan.json'
DATA = ROOT / 'ring_ff_trainval.npz'
SEEDS = (17, 23, 41)
STEPS = tuple(range(200, 12001, 200))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class StateCritic(nn.Module):
    @nn.compact
    def __call__(self, state, eta):
        z = jnp.concatenate((state, eta), axis=-1)
        z = nn.silu(nn.Dense(64, name='dense1')(z))
        z = nn.silu(nn.Dense(64, name='dense2')(z))
        return nn.Dense(1, name='out')(z)[..., 0]


class EtaCritic(nn.Module):
    @nn.compact
    def __call__(self, eta):
        z = nn.silu(nn.Dense(64, name='dense1')(eta))
        z = nn.silu(nn.Dense(64, name='dense2')(z))
        return nn.Dense(1, name='out')(z)[..., 0]


def metrics(logits, val_s, val_f, val_u):
    probabilities = np.asarray(jax.nn.sigmoid(logits), float)
    lower = val_s / 16
    upper = (val_s + val_u) / 16
    selected = np.argmax(logits, axis=1)
    oracle = lower.max(1)
    regret = float(np.mean(oracle - lower[np.arange(len(selected)), selected]))
    robust = int(np.sum(val_s[np.arange(len(selected)), selected] >= 15))
    a, b = np.triu_indices(16, 1)
    direction = np.where(lower[:, a] - upper[:, b] >= .25, 1,
                         np.where(lower[:, b] - upper[:, a] >= .25, -1, 0))
    valid = direction != 0
    predicted = np.sign(logits[:, a] - logits[:, b])
    pairwise = float(np.mean((predicted[valid] == direction[valid]).astype(float)
                             + .5 * (predicted[valid] == 0))) if valid.any() else None
    nll = float((val_s * np.logaddexp(0., -logits)
                 + val_f * np.logaddexp(0., logits)).sum() / (val_s + val_f).sum())
    return dict(selection_regret=regret, pairwise_accuracy=pairwise,
                validation_nll=nll, validation_B15=robust,
                mean_selected_success=float(np.mean(lower[np.arange(len(selected)), selected])),
                decisive_pair_cases=int(valid.sum()),
                validation_unknown=int(val_u.sum()),
                selected_eta=selected.tolist(),
                predicted_probability=probabilities.tolist())


def select_key(row, tolerance):
    # Discrete selection regret: candidates within one predeclared 1/512 bin
    # are resolved by decisive pair ranking, then NLL, then chronological step.
    bucket = int(np.floor(row['selection_regret'] / tolerance + 1e-9))
    return (bucket, -(row['pairwise_accuracy'] if row['pairwise_accuracy'] is not None else -.5),
            row['validation_nll'], row['step'])


def fit(kind, seed):
    assert kind in ('state', 'eta_only') and seed in SEEDS
    plan = json.loads(PLAN.read_text())
    assert sha(DATA) == json.loads((ROOT / 'ring_ff_trainval_manifest.json').read_text())['trainval_artifact_sha256']
    assert plan['test_labels_forbidden_until_models_frozen']
    d = np.load(DATA)
    assert d['physical23'].shape == (544, 23) and d['eta'].shape == (16, 3)
    assert np.all(d['split'][:512] == 'train') and np.all(d['split'][512:] == 'validation')
    assert np.all(d['success'][:512] + d['failure'][:512] + d['numerical'][:512] == 4)
    assert np.all(d['success'][512:] + d['failure'][512:] + d['numerical'][512:] == 16)
    folder = ROOT / 'ring_ff_fits' / kind / f'seed{seed}'
    if (folder / 'complete.json').is_file():
        row = json.loads((folder / 'complete.json').read_text())
        assert row['trainval_sha256'] == sha(DATA) and sha(folder / 'checkpoint.msgpack') == row['checkpoint_sha256']
        return row
    folder.mkdir(parents=True, exist_ok=True)
    x = jnp.asarray(d['physical23'], dtype=jnp.float32)
    e = jnp.asarray(d['eta'], dtype=jnp.float32)
    s = jnp.asarray(d['success'], dtype=jnp.float32)
    f = jnp.asarray(d['failure'], dtype=jnp.float32)
    val_s = np.asarray(d['success'][512:], int)
    val_f = np.asarray(d['failure'][512:], int)
    val_u = np.asarray(d['numerical'][512:], int)
    net = StateCritic() if kind == 'state' else EtaCritic()
    if kind == 'state':
        params = net.init(jax.random.PRNGKey(seed), x[:1], e[:1])
    else:
        params = net.init(jax.random.PRNGKey(seed), e[:1])
    optimizer = optax.chain(optax.clip_by_global_norm(5.),
                            optax.adamw(.0008, weight_decay=.0001))
    optim = optimizer.init(params)
    valid_train = np.argwhere(np.asarray(d['success'][:512] + d['failure'][:512]) > 0)
    assert len(valid_train) == 512 * 16
    mean_exposure = float(np.mean(np.asarray(d['success'][:512] + d['failure'][:512])))

    @jax.jit
    def block(pp, oo, draw):
        def one(carry, batch):
            weights, state = carry
            ii, jj = batch[:, 0], batch[:, 1]
            def loss(p):
                z = net.apply(p, x[ii], e[jj]) if kind == 'state' else net.apply(p, e[jj])
                return jnp.mean(s[ii, jj] * jax.nn.softplus(-z)
                                + f[ii, jj] * jax.nn.softplus(z)) / mean_exposure
            value, grads = jax.value_and_grad(loss)(weights)
            updates, state = optimizer.update(grads, state, weights)
            return (optax.apply_updates(weights, updates), state), value
        return jax.lax.scan(one, (pp, oo), draw)

    @jax.jit
    def predict(pp):
        ii = jnp.repeat(jnp.arange(512, 544), 16)
        jj = jnp.tile(jnp.arange(16), 32)
        z = net.apply(pp, x[ii], e[jj]) if kind == 'state' else net.apply(pp, e[jj])
        return z.reshape(32, 16)

    rng = np.random.default_rng(seed)
    started = time.monotonic()
    history = []
    best = None
    best_bytes = None
    previous = 0
    for step in STEPS:
        indices = rng.integers(len(valid_train), size=(step - previous, 128))
        draws = jnp.asarray(valid_train[indices], dtype=jnp.int32)
        (params, optim), _ = block(params, optim, draws)
        row = dict(step=step, **metrics(np.asarray(predict(params)), val_s, val_f, val_u))
        history.append(row)
        if best is None or select_key(row, plan['checkpoint_selection']['near_tie_tolerance']) < select_key(best, plan['checkpoint_selection']['near_tie_tolerance']):
            best, best_bytes = row, serialization.to_bytes(params)
        previous = step
        if step % 1000 == 0:
            print(json.dumps(dict(kind=kind, seed=seed, step=step,
                                  regret=row['selection_regret'],
                                  best_step=best['step'], seconds=time.monotonic()-started)), flush=True)
    (folder / 'checkpoint.msgpack').write_bytes(best_bytes)
    dump(folder / 'history.json', history)
    result = dict(kind=kind, seed=seed, selected_step=best['step'],
                  selection_regret=best['selection_regret'],
                  pairwise_accuracy=best['pairwise_accuracy'],
                  validation_nll=best['validation_nll'],
                  validation_B15=best['validation_B15'],
                  checkpoint_sha256=sha(folder / 'checkpoint.msgpack'),
                  trainval_sha256=sha(DATA), plan_sha256=sha(PLAN),
                  script_sha256=sha(__file__),
                  elapsed_seconds=time.monotonic()-started,
                  backend=jax.default_backend())
    dump(folder / 'complete.json', result)
    return result


def freeze_models():
    plan = json.loads(PLAN.read_text())
    all_rows = {kind: [json.loads((ROOT / 'ring_ff_fits' / kind / f'seed{seed}' / 'complete.json').read_text())
                       for seed in SEEDS] for kind in ('state', 'eta_only')}
    chosen = {}
    for kind, rows in all_rows.items():
        chosen[kind] = min(rows, key=lambda r: (
            int(np.floor(r['selection_regret'] / plan['checkpoint_selection']['near_tie_tolerance'] + 1e-9)),
            -(r['pairwise_accuracy'] if r['pairwise_accuracy'] is not None else -.5),
            r['validation_nll'], r['selected_step'], r['seed']))
    output = dict(all_runs=all_rows, selected=chosen, plan_sha256=sha(PLAN),
                  trainval_sha256=sha(DATA), test_labels_used=False,
                  input_definition='physical23 + eta3 only; eta-only eta3 only')
    target = ROOT / 'ring_ff_models_frozen.json'
    if target.exists():
        assert json.loads(target.read_text()) == output
    else:
        dump(target, output)
    print(json.dumps({k:(v['seed'],v['selected_step']) for k,v in chosen.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('state', 'eta_only', 'freeze'))
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    print(json.dumps(fit(args.kind, args.seed) if args.kind != 'freeze' else freeze_models()), flush=True)
