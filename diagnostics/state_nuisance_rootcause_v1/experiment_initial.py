"""Focused, paired Ring FF nuisance-input intervention; no rollout code."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import hashlib
import json
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'orthoflow3_tt_ff_matched_learnability_v1'
import sys
sys.path.insert(0, str(ROOT))


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def setup():
    import jax
    import jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_critic_rootcause_resolution_v1.db_transfer_train import model
    data = dict(np.load(SOURCE / 'dataset.npz'))
    entities = dict(np.load(SOURCE / 'entities.npz'))
    frozen = read(SOURCE / 'models_frozen.json')
    manifest = read(SOURCE / 'dataset_frozen.json')
    assert sha(SOURCE / 'dataset.npz') == manifest['dataset_sha256']
    assert sha(SOURCE / 'entities.npz') == manifest['entities_sha256']
    return jax, jnp, serialization, model, data, entities, frozen


def replay():
    jax, jnp, serialization, model, data, entities, frozen = setup()
    cached = np.load(SOURCE / 'frozen_predictions.npz')
    funcs = {k: jax.jit(model(k).apply) for k in ('eta_only', 'full_context')}
    result = {}
    parity = []
    for run in frozen['runs']:
        scene, chain, kind, seed = [run[k] for k in ('scene', 'chain', 'kind', 'seed')]
        ci = ('TT', 'FF').index(chain)
        p = serialization.msgpack_restore(Path(run['checkpoint']).read_bytes())
        assert sha(run['checkpoint']) == run['checkpoint_sha256']
        for split in ('train', 'validation', 'test'):
            ix = np.flatnonzero((data['scene'] == scene) & (data['split'] == split))
            ii = np.repeat(ix, 16)
            jj = np.tile(np.arange(16), len(ix))
            chunks = []
            for start in range(0, len(ii), 128):
                si, ej = ii[start:start+128], jj[start:start+128]
                z = funcs[kind](p, {k: jnp.asarray(v[si]) for k, v in entities.items()},
                                jnp.asarray(data['eta'][si, ej]), jnp.asarray(data['context'][ci, si, ej]))
                chunks.append(np.asarray(z))
            z = np.concatenate(chunks).reshape(len(ix), 16)
            key = f'{scene}__{chain}__{kind}__{seed}__{split}'
            result[key] = z
            if split != 'train':
                reference = cached[key + '__correct']
                error = float(np.max(np.abs(z - reference)))
                from scipy.special import expit
                pdiff = abs(expit(z)-expit(reference))
                parity.append(dict(key=key, max_abs_logit_error=error,
                                   max_abs_probability_error=float(pdiff.max()),
                                   mean_abs_probability_error=float(pdiff.mean()),
                                   changed_argmax=int((z.argmax(1)!=reference.argmax(1)).sum())))
                # Current GPU kernels differ slightly from the historical run.
                # Audit on probability scale; all historical heldout comparisons
                # retain the ORIGINAL frozen scores, rather than replacing them.
                assert pdiff.max() < .002 and pdiff.mean() < .0003, (key, error, pdiff.max())
                result[key] = reference.copy()
        print('replayed', scene, chain, kind, seed, flush=True)
    np.savez_compressed(HERE / 'replayed_logits.npz', **result)
    write(HERE / 'replay_integrity.json', dict(backend=jax.default_backend(), checks=parity,
          heldout_predictions='Original frozen scores retained; only TRAIN scores newly replayed',
          protocol_sha256=sha(HERE / 'protocol.json'), dataset_sha256=sha(SOURCE / 'dataset.npz'),
          original_model_freeze_sha256=sha(SOURCE / 'models_frozen.json')))


def fit():
    jax, jnp, serialization, model, data, entities, frozen = setup()
    import optax
    from flax import traverse_util
    scene, chain, ci = 'ring_exchange', 'FF', 1
    indices = {split: np.flatnonzero((data['scene'] == scene) & (data['split'] == split))
               for split in ('train', 'validation', 'test')}
    tr, va = indices['train'], indices['validation']
    means = {k: np.mean(v[tr], axis=0) for k, v in entities.items()}
    context_mean = np.mean(data['context'][ci, tr], axis=0)
    pairs = np.array([(i, j) for i in tr for j in range(16)
                      if data['success'][ci, i, j] + data['failure'][ci, i, j] > 0])
    nmean = float(np.mean([data['success'][ci, i, j] + data['failure'][ci, i, j] for i, j in pairs]))
    net = model('full_context')
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(.0008, weight_decay=.0001))
    mask = None

    def gather(arm, ii, jj):
        xx = {k: jnp.asarray(v[ii] if arm == 'full_replay' else np.broadcast_to(means[k], (len(ii), *means[k].shape)))
              for k, v in entities.items()}
        cc = context_mean[jj] if arm == 'constant_all' else data['context'][ci, ii, jj]
        return xx, jnp.asarray(data['eta'][ii, jj]), jnp.asarray(cc)

    def make_step(warm):
        @jax.jit
        def step(p, o, xx, ee, cc, ss, ff):
            def loss(q):
                z = net.apply(q, xx, ee, cc)
                return jnp.mean(ss * jax.nn.softplus(-z) + ff * jax.nn.softplus(z)) / nmean
            val, grad = jax.value_and_grad(loss)(p)
            if warm:
                grad = jax.tree.map(lambda a, b: a if b else jnp.zeros_like(a), grad, mask)
            updates, o = optimizer.update(grad, o, p)
            if warm:
                updates = jax.tree.map(lambda a, b: a if b else jnp.zeros_like(a), updates, mask)
            return optax.apply_updates(p, updates), o, val
        return step

    step, warmstep = make_step(False), make_step(True)
    predict = jax.jit(net.apply)

    def predict_split(p, arm, split):
        ix = indices[split]
        ii, jj = np.repeat(ix, 16), np.tile(np.arange(16), len(ix))
        z = []
        for start in range(0, len(ii), 128):
            z.append(np.asarray(predict(p, *gather(arm, ii[start:start+128], jj[start:start+128]))))
        return np.concatenate(z).reshape(len(ix), 16)

    def nll(z, split):
        ix = indices[split]
        s, f = data['success'][ci, ix], data['failure'][ci, ix]
        return float(np.sum(s * np.logaddexp(0, -z) + f * np.logaddexp(0, z)) / np.sum(s + f))

    all_runs = []
    for arm in ('full_replay', 'constant_h', 'constant_all'):
        for seed in (17, 23, 41):
            dest = HERE / 'fits' / arm / f'seed{seed}'
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'complete.json').exists():
                all_runs.append(read(dest / 'complete.json'))
                continue
            start = time.perf_counter()
            pp = net.init(jax.random.PRNGKey(seed), {k: jnp.asarray(v[tr[:1]]) for k, v in entities.items()},
                          jnp.zeros((1, 3)), jnp.zeros((1, 76)))
            initial = hashlib.sha256(serialization.to_bytes(pp)).hexdigest()
            old = next(r for r in frozen['runs'] if (r['scene'], r['chain'], r['kind'], r['seed']) == (scene, chain, 'full_context', seed))
            assert initial == old['initial_sha256'], (arm, seed, 'initial mismatch')
            assert hashlib.sha256(pairs.tobytes()).hexdigest() == old['pair_order_sha256']
            mask = traverse_util.unflatten_dict({k: k[-2] != 'raw_context_skip' for k in traverse_util.flatten_dict(pp)})
            oo = optimizer.init(pp)
            rng = np.random.default_rng(seed)
            drawhash = hashlib.sha256()
            history = []
            for it in range(1, 2001):
                draw = rng.integers(len(pairs), size=128)
                drawhash.update(draw.tobytes())
                ii, jj = pairs[draw].T
                pp, oo, loss = (warmstep if it <= 25 else step)(pp, oo, *gather(arm, ii, jj),
                    jnp.asarray(data['success'][ci, ii, jj]), jnp.asarray(data['failure'][ci, ii, jj]))
                if it % 100:
                    continue
                zz = {split: predict_split(pp, arm, split) for split in ('train', 'validation')}
                history.append(dict(step=it, TRAIN_NLL=nll(zz['train'], 'train'), VAL_NLL=nll(zz['validation'], 'validation')))
                (dest / f'step{it}.msgpack').write_bytes(serialization.to_bytes(pp))
                np.savez_compressed(dest / f'step{it}_source.npz', **zz)
                if it % 500 == 0:
                    print('progress', arm, seed, it, round(time.perf_counter()-start, 1), history[-1], flush=True)
            assert drawhash.hexdigest() == old['batch_draw_sha256']
            write(dest / 'history.json', history)
            complete = dict(arm=arm, seed=seed, backend=jax.default_backend(),
                            seconds=time.perf_counter()-start, initial_sha256=initial,
                            batch_draw_sha256=drawhash.hexdigest(), original_full_step=old['selected_step'],
                            protocol_sha256=sha(HERE / 'protocol.json'), script_sha256=sha(__file__),
                            test_labels_read=False)
            write(dest / 'complete.json', complete)
            all_runs.append(complete)
            print('completed', arm, seed, round(complete['seconds'], 1), flush=True)
    selections = {}
    output = {}
    for arm in ('full_replay', 'constant_h', 'constant_all'):
        histories = [read(HERE / 'fits' / arm / f'seed{s}' / 'history.json') for s in (17, 23, 41)]
        val = np.mean([[h['VAL_NLL'] for h in history] for history in histories], axis=0)
        beststep = histories[0][int(np.argmin(val))]['step']
        original = next(r for r in all_runs if r['arm'] == arm)['original_full_step']
        selections[arm] = dict(selected_step=beststep, original_full_step=original,
                               mean_VAL_NLL_by_step={str((j+1)*100): float(x) for j, x in enumerate(val)})
        for seed in (17, 23, 41):
            for label, it in [('selected', beststep), ('original_step', original), ('final', 2000)]:
                path = HERE / 'fits' / arm / f'seed{seed}' / f'step{it}.msgpack'
                pp = serialization.msgpack_restore(path.read_bytes())
                for split in ('train', 'validation', 'test'):
                    output[f'{arm}__{seed}__{label}__{split}'] = predict_split(pp, arm, split)
    write(HERE / 'selection.json', dict(arms=selections, test_labels_read=False, protocol_sha256=sha(HERE / 'protocol.json')))
    np.savez_compressed(HERE / 'intervention_logits.npz', **output)
    write(HERE / 'training_complete.json', dict(runs=all_runs, selected_on='mean three-seed source VAL NLL',
          data_sha256=sha(SOURCE / 'dataset.npz'), selection_sha256=sha(HERE / 'selection.json')))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('replay', 'fit'))
    globals()[parser.parse_args().action]()
