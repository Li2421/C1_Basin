"""Source-only factorial repair, with selection-first frozen checkpoint rule."""
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax
from diagnostics.controller_q_diversity_v1.train import Critic
from .study import HERE, read, write, freeze, sha
from .features import INDICES

SCENES = ('toy_give_way', 'double_bottleneck', 'four_way_intersection', 'ring_exchange')
FOLDS = {'joint': None, 'held_toy': SCENES[0], 'held_db': SCENES[1], 'held_four': SCENES[2], 'held_ring': SCENES[3]}
CONDITIONS = ('eta_only_same_all_labels', 'field_same_all_labels', 'encounter_same_all_labels',
              'encounter_flow_eta_prior', 'encounter_goal_only')
SEEDS = (17, 23, 41)
JOBS = [dict(fold=f, condition=c, seed=s) for f in FOLDS for c in CONDITIONS for s in SEEDS]


def load():
    d = dict(np.load(HERE / 'dataset.npz'))
    for k in ('eta', 'success', 'failure', 'unknown', 'action_feature'):
        d[k] = d[k][:, list(INDICES)]
    return d


def centered(x):
    return x - x.mean(0, keepdims=True) - x.mean(-1, keepdims=True) + x.mean((0, -1), keepdims=True)


def metrics(q, d, ix, z):
    s, f, u = (d[k][ix] for k in ('success', 'failure', 'unknown'))
    valid = s + f > 0
    y = s / np.maximum(s + f, 1)
    n = s + f + u
    lo, hi = s / n, (s + u) / n
    selected = q.argmax(-1)
    # Tight worst-case regret over unresolved numerical outcomes. When the
    # selected eta is itself the oracle it cannot incur self-regret.
    regret = hi - lo[np.arange(len(ix)), selected, None]
    regret[np.arange(len(ix)), selected] = 0.
    regret = float(np.maximum(regret.max(-1), 0).mean())
    a, b = np.triu_indices(4, 1)
    pos = lo[:, a] - hi[:, b] >= .5
    neg = lo[:, b] - hi[:, a] >= .5
    strong = pos | neg
    direction = q[:, a] - q[:, b]
    ranking = np.where(direction == 0, .5, np.where(pos, direction > 0, direction < 0))
    accuracy = float(ranking[strong].mean()) if strong.any() else .5
    state_errors, controller_errors = [], []
    for sc in np.unique(d['scene'][ix]):
        for c in np.unique(d['controller'][ix]):
            ii = np.flatnonzero((d['scene'][ix] == sc) & (d['controller'][ix] == c))
            if len(ii) > 1:
                state_errors.append(float(np.mean((centered(q[ii]) - centered(lo[ii])) ** 2)))
    for sid in np.unique(d['state_uid'][ix]):
        ii = np.flatnonzero(d['state_uid'][ix] == sid)
        a, b = np.triu_indices(len(ii), 1)
        good = valid[ii[a]] & valid[ii[b]]
        if good.any():
            controller_errors.append(float(np.mean(((q[ii[a]] - q[ii[b]]) - (y[ii[a]] - y[ii[b]])) ** 2, where=good)))
    return dict(selection_regret_upper=regret, strong_pair_accuracy=accuracy,
                strong_pair_cases=int(strong.sum()), state_eta_centered_MSE=float(np.mean(state_errors)),
                Q_MSE=float(np.mean((q - y) ** 2, where=valid)),
                controller_contrast_MSE=float(np.mean(controller_errors)),
                nll=float(np.sum(s * np.logaddexp(0, -z) + f * np.logaddexp(0, z)) / np.sum(s + f)))


def fit(index):
    job = JOBS[index]
    rule = read(HERE / 'analysis_protocol.json')
    p = rule['optimization']
    audit = read(HERE / 'dataset_audit.json')
    assert sha(HERE / 'dataset.npz') == audit['dataset_sha256']
    d = load()
    held = FOLDS[job['fold']]
    source = d['scene'] != held if held else np.ones(len(d['scene']), bool)
    tr = (d['split'] == 'train') & source
    va = np.flatnonzero((d['split'] == 'validation') & source)
    if job['condition'] == 'encounter_goal_only':
        tr &= d['kind'] == 'goal'
    assert not np.intersect1d(d['state_uid'][tr], d['state_uid'][va]).size
    if held:
        assert held not in d['scene'][tr] and held not in d['scene'][va]
    out = HERE / 'fits' / job['fold'] / job['condition'] / f"seed{job['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    freeze(out / 'job.json', dict(**job, dataset_sha256=audit['dataset_sha256'],
        rule_sha256=sha(HERE/'analysis_protocol.json'), code_sha256=sha(__file__),
        source_train_state_ids=sorted(set(d['state_uid'][tr])), source_val_state_ids=sorted(set(d['state_uid'][va])),
        held_labels_used_for_fit_normalization_or_selection=False))
    mask = d['entity_agent_mask'].astype(bool)
    feature = np.concatenate((np.broadcast_to(d['feature'][:, None], d['action_feature'].shape[:-1] + (64,)), d['action_feature']), -1)
    v = feature[tr][np.broadcast_to(mask[tr, None, :], feature[tr].shape[:-1])]
    mean, scale = v.mean(0), np.maximum(v.std(0), .05)
    freeze(out / 'normalization.json', dict(mean=mean.tolist(), scale=scale.tolist(), source_train_only=True, clip=10.))
    values = np.clip((feature - mean) / scale, -10, 10) * mask[:, None, :, None]
    if job['condition'] == 'field_same_all_labels':
        values[..., 64:] = 0
    c = jnp.asarray(values, jnp.float32)
    x = {k[7:]: jnp.asarray(d[k], jnp.float32) for k in d if k.startswith('entity_')}
    e = jnp.asarray(d['eta'], jnp.float32)
    s, f = d['success'].astype(float).copy(), d['failure'].astype(float).copy()
    eligible = s + f > 0
    control_audit = []
    if job['condition'] == 'encounter_flow_eta_prior':
        for sc in SCENES:
            for name in np.unique(d['controller'][d['kind'] == 'flow']):
                keep = tr & (d['scene'] == sc) & (d['controller'] == name)
                if not keep.any():
                    continue
                trial = s[keep] + f[keep]
                observed = trial > 0
                state_mean = np.sum(s[keep] / np.maximum(trial, 1), 0) / np.maximum(observed.sum(0), 1)
                s[keep], f[keep] = trial * state_mean, trial * (1 - state_mean)
                control_audit.append(dict(scene=sc, controller=name, independent_states=int(keep.sum()),
                                           mean=state_mean.tolist(), resolved_states=observed.sum(0).tolist()))
    write(out / 'label_control.json', dict(source_train_only=True, rows=control_audit))
    s, f = jnp.asarray(s, jnp.float32), jnp.asarray(f, jnp.float32)
    kind = 'eta_only' if job['condition'] == 'eta_only_same_all_labels' else 'state_field_compact'
    net = Critic(kind)
    gather = lambda ii: {k: v[ii] for k, v in x.items()}
    params = net.init(jax.random.PRNGKey(job['seed']), gather(jnp.array([0])), c[:1, 0], e[:1, 0])
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(p['learning_rate'], weight_decay=p['weight_decay']))
    opt = optimizer.init(params)

    @jax.jit
    def block(params, opt, draws):
        def step(carry, draw):
            pp, oo = carry
            ii, jj = draw[:, 0], draw[:, 1]
            def loss(par):
                z = net.apply(par, gather(ii), c[ii, jj], e[ii, jj])
                ss, ff = s[ii, jj], f[ii, jj]
                return jnp.mean((ss * jax.nn.softplus(-z) + ff * jax.nn.softplus(z)) / jnp.maximum(ss + ff, 1))
            loss0, grad = jax.value_and_grad(loss)(pp)
            updates, oo = optimizer.update(grad, oo, pp)
            return (optax.apply_updates(pp, updates), oo), loss0
        return jax.lax.scan(step, (params, opt), draws)

    @jax.jit
    def predict(params, ii):
        return jax.vmap(lambda j: net.apply(params, gather(ii), c[ii, j], e[ii, j]), out_axes=1)(jnp.arange(4))

    def infer(params, ii):
        return np.concatenate([np.asarray(predict(params, jnp.asarray(ii[k:k+32]))) for k in range(0, len(ii), 32)])

    pools = {}
    for sc in SCENES:
        for name in np.unique(d['controller']):
            keep = tr & (d['scene'] == sc) & (d['controller'] == name)
            rr, jj = np.where(eligible & keep[:, None])
            if len(rr):
                pools[sc, name] = np.column_stack((rr, jj))
    scenes = sorted({key[0] for key in pools})
    batch = p['batch_size'] // len(scenes)
    assert batch * len(scenes) == p['batch_size']
    rng = np.random.default_rng(job['seed'])
    history, best = [], None
    started = time.monotonic()
    for step in range(p['validation_every'], p['steps'] + 1, p['validation_every']):
        chunks = []
        for sc in scenes:
            entries = [pool for (scene, _), pool in pools.items() if scene == sc]
            which = rng.integers(len(entries), size=(p['validation_every'], batch))
            draws = np.empty(which.shape + (2,), np.int32)
            for j, pool in enumerate(entries):
                selected = which == j
                draws[selected] = pool[rng.integers(len(pool), size=selected.sum())]
            chunks.append(draws)
        (params, opt), loss = block(params, opt, jnp.asarray(np.concatenate(chunks, 1)))
        z = infer(params, va).astype(float)
        q = 1 / (1 + np.exp(-np.clip(z, -100, 100)))
        score = metrics(q, d, va, z)
        row = dict(step=step, training_nll=float(np.mean(loss)), **score)
        assert all(np.isfinite(v) for v in row.values()), row
        history.append(row)
        key = (score['selection_regret_upper'], -score['strong_pair_accuracy'], score['state_eta_centered_MSE'],
               score['Q_MSE'], score['controller_contrast_MSE'], score['nll'], step)
        if best is None or key < best:
            best, chosen, best_bytes = key, row, serialization.to_bytes(params)
        if step % 1000 == 0:
            write(out / 'history.json', history)
            print(dict(**job, step=step, regret=score['selection_regret_upper'],
                       best_step=chosen['step'], seconds=round(time.monotonic()-started, 1)), flush=True)
    (out / 'checkpoint.msgpack').write_bytes(best_bytes)
    best_params = serialization.from_bytes(params, best_bytes)
    z = infer(best_params, np.arange(len(d['scene']))).astype(float)
    np.savez_compressed(out / 'predictions.npz', logits=z, probability=1 / (1 + np.exp(-np.clip(z, -100, 100))),
                        eta_indices=np.asarray(INDICES))
    write(out / 'history.json', history)
    write(out / 'complete.json', dict(**job, selected=chosen, seconds=time.monotonic()-started,
        source_states=len(np.unique(d['state_uid'][tr])), train_cells=int((eligible & tr[:, None]).sum())))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['fit'])
    p.add_argument('--index', type=int, required=True)
    a = p.parse_args()
    fit(a.index)
