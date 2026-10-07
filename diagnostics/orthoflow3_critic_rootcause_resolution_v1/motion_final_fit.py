"""Fit preselected matched models on source TRAIN; freeze before new targets."""
import argparse
import hashlib
import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from .motion_factorial_train import OUT as SOURCE, load, model, metrics, read, write, sha, SEEDS
from .motion_eta_control import model as eta_model

ROOT = SOURCE.parent
OUT = ROOT/'motion_final_source_models'
RULE = ROOT/'motion_final_fit_protocol.json'
ARMS = ('full_update', 'trunk_only', 'eta_only')
STEPS = (25, 75, 400)


def prepare():
    protocol = read(RULE)
    assert sha(SOURCE/'dataset.npz') == protocol['source_dataset_sha256']
    selections = {
        'full_update': read(ROOT/'motion_freeze_control/source_selection.json')['selection']['full_update']['step'],
        'trunk_only': read(ROOT/'motion_freeze_control/source_selection.json')['selection']['trunk_only']['step'],
        'eta_only': read(ROOT/'motion_eta_control/source_selection.json')['neural']['step'],
    }
    assert selections == dict(full_update=25, trunk_only=400, eta_only=75)
    value = dict(rule_sha256=sha(RULE), dataset_sha256=sha(SOURCE/'dataset.npz'),
                 selected_steps=selections, source_selection_only=True,
                 code_sha256=sha(__file__), model_code_sha256=sha(ROOT/'goal_velocity_train.py'),
                 eta_code_sha256=sha(ROOT/'motion_eta_control.py'),
                 target_labels_read=False, new_rollouts=0)
    dest = OUT/'selection_frozen.json'
    if dest.exists():
        assert read(dest) == value, 'Do not silently change the frozen fitting protocol'
    else:
        write(dest, value)
    print(value, flush=True)


def train_seed(index):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization, traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend() == 'gpu'
    seed = SEEDS[index]
    frozen = read(OUT/'selection_frozen.json')
    assert sha(RULE) == frozen['rule_sha256'] and sha(__file__) == frozen['code_sha256']
    assert sha(SOURCE/'dataset.npz') == frozen['dataset_sha256']
    assert sha(ROOT/'goal_velocity_train.py') == frozen['model_code_sha256']
    assert sha(ROOT/'motion_eta_control.py') == frozen['eta_code_sha256']
    d, x, tr, va, norm = load(list(range(12)))
    si, eta, context = d['state_index'], d['normalized_eta'], d['input_context']
    assert len(tr) == 128 and len(va) == 32
    for arm in ARMS:
        dest = OUT/arm/f'seed{seed}'
        if (dest/'complete.json').exists():
            previous = read(dest/'complete.json')
            assert previous['selection_frozen_sha256'] == sha(OUT/'selection_frozen.json')
            continue
        m = eta_model() if arm == 'eta_only' else model('rest_motion')
        p = m.init(jax.random.PRNGKey(seed), gather(x, si[:1]), jnp.zeros((1, 3)), jnp.zeros((1, 120)))
        initial_hash = hashlib.sha256(serialization.to_bytes(p)).hexdigest()
        flat = traverse_util.flatten_dict(p)
        masks = {k: k[-2] in {'trunk1', 'trunk2', 'out'} for k in flat}
        mask = traverse_util.unflatten_dict(masks)
        opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(.0008, weight_decay=.0001))
        state = opt.init(p)

        def make_step(freeze):
            @jax.jit
            def step(pp, oo, xx, e, c, success, failure):
                def loss(params):
                    z = m.apply(params, xx, e, c)
                    q = success/jnp.maximum(success+failure, 1)
                    return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
                value, grad = jax.value_and_grad(loss)(pp)
                if freeze:
                    grad = jax.tree.map(lambda a, b: a if b else jnp.zeros_like(a), grad, mask)
                update, oo = opt.update(grad, oo, pp)
                if freeze:
                    update = jax.tree.map(lambda a, b: a if b else jnp.zeros_like(a), update, mask)
                return optax.apply_updates(pp, update), oo, value
            return step

        warm, fixed = make_step(False), make_step(True)
        predict = jax.jit(lambda pp, xx, e, c: m.apply(pp, xx, e, c))
        rng = np.random.default_rng(seed)
        order = hashlib.sha256()
        history, predictions = [], {}
        dest.mkdir(parents=True, exist_ok=True)
        saved_encoders = None
        for it in range(1, 401):
            draw = rng.choice(tr, 32)
            order.update(draw.tobytes())
            ii, ci = np.tile(draw, 16), np.repeat(np.arange(16), 32)
            update = fixed if arm == 'trunk_only' and it > 25 else warm
            p, state, loss = update(p, state, gather(x, si[ii]), jnp.asarray(eta[ii], jnp.float32),
                                   jnp.asarray(context[ci, ii], jnp.float32),
                                   jnp.asarray(d['success'][ci, ii]), jnp.asarray(d['failure'][ci, ii]))
            if it == 25:
                saved_encoders = {k: np.array(v) for k, v in traverse_util.flatten_dict(p).items() if not masks[k]}
                if arm == 'trunk_only':
                    parent = serialization.msgpack_restore((OUT/'full_update'/f'seed{seed}'/'step25.msgpack').read_bytes())
                    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(parent)))
            if it not in STEPS:
                continue
            (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
            z = np.array([np.asarray(predict(p, gather(x, si[va]), jnp.asarray(eta[va], jnp.float32),
                                            jnp.asarray(context[c, va], jnp.float32))) for c in range(16)])
            predictions[f'step{it}'] = z
            history.append(dict(step=it, last_batch_TRAIN_NLL=float(loss),
                                native_seen_controller_VAL=metrics(z, d, va, list(range(12))),
                                program_VAL=metrics(z, d, va, [12, 13, 14, 15]),
                                used_for_checkpoint_selection=False))
        if arm == 'trunk_only':
            assert all(np.array_equal(v, traverse_util.flatten_dict(p)[k]) for k, v in saved_encoders.items())
        write(dest/'normalization.json', norm)
        write(dest/'history.json', history)
        np.savez_compressed(dest/'predictions.npz', indices=va, **predictions)
        write(dest/'complete.json', dict(seed=seed, arm=arm, selected_step=frozen['selected_steps'][arm],
              total_steps=400, train_families=64, train_controllers=16, TRAIN_pairs=2048,
              initial_parameters_sha256=initial_hash, pair_order_sha256=order.hexdigest(),
              parameter_count=int(sum(a.size for a in flat.values())),
              selection_frozen_sha256=sha(OUT/'selection_frozen.json'),
              first25_full_update_exact=arm=='trunk_only', frozen_parameters_unchanged=arm=='trunk_only',
              new_rollouts=0, target_labels_used=False))
        print(dict(seed=seed, arm=arm, complete=True, chosen=frozen['selected_steps'][arm]), flush=True)


def freeze():
    specification = read(OUT/'selection_frozen.json')
    entries = []
    for seed in SEEDS:
        records = [read(OUT/arm/f'seed{seed}'/'complete.json') for arm in ARMS]
        for key in ('initial_parameters_sha256', 'pair_order_sha256', 'parameter_count'):
            assert len({r[key] for r in records}) == 1, (seed, key)
        for arm in ARMS:
            folder = OUT/arm/f'seed{seed}'
            step = specification['selected_steps'][arm]
            for label, checkpoint in [(arm, step)] + ([('full_update_matched400', 400)] if arm=='full_update' else []):
                entries.append(dict(variant=label, seed=seed, path=str(folder), checkpoint=f'step{checkpoint}.msgpack',
                                    checkpoint_sha256=sha(folder/f'step{checkpoint}.msgpack'),
                                    normalization_sha256=sha(folder/'normalization.json'), steps=checkpoint,
                                    diagnostic_only=label=='full_update_matched400'))
    result = dict(models=entries, rule_sha256=sha(RULE), selection_frozen_sha256=sha(OUT/'selection_frozen.json'),
                  source_dataset_sha256=sha(SOURCE/'dataset.npz'), target_labels_used=False, new_rollouts=0,
                  independent_confirmation_complete=False,
                  scope='Frozen source-only models. No claim of generalization until independent evaluation.')
    path = OUT/'models_frozen.json'
    if path.exists():
        assert read(path) == result
    else:
        write(path, result)
    print(dict(frozen_models=len(entries), sha256=sha(path), new_rollouts=0), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'train', 'freeze'))
    p.add_argument('--index', type=int, default=0)
    a = p.parse_args()
    train_seed(a.index) if a.action=='train' else globals()[a.action]()
