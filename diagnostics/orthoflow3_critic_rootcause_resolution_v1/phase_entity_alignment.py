"""One source-only matched input association test after crossed supervision.

Unlike earlier two-controller controls, this uses the frozen twelve-function
plus crossed-phase evidence. No new measurements, task rollouts, parameters,
target labels, or loss terms. Existing agent-response associations are retained.
"""
import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from .phase_factorial_train import load, metrics, causal, folds, SEEDS, STEPS
from .phase_factorial_support import OUT as DATA, read, write, sha
from .goal_response_cv import model as mean_model, csvwrite
OUT = DATA.parent / 'phase_entity_alignment'


def model():
    import flax.linen as nn
    import jax.numpy as jnp
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    class AlignedModel(nn.Module):
        @nn.compact
        def __call__(self, x, eta, context):
            a = context[:, 24:88].reshape(-1, x['agents'].shape[1], 16)
            xx = {**x, 'agents': jnp.concatenate([x['agents'], a], -1)}
            c = jnp.concatenate([context[:, :24], context[:, 88:104]], -1)
            return Critic(True, True, False, name='core')(xx, eta, c, jnp.zeros((len(eta), 3)))
    return AlignedModel()


def prepare():
    assert not (OUT / 'protocol.json').exists()
    write(OUT / 'protocol.json', dict(
        question='Does preserving entity-response association permit state-specific learning under richer crossed-controller supervision?',
        compared_to=str(DATA), dataset_sha256=sha(DATA / 'dataset.npz'),
        source_families='same64TRAIN16VAL', source_controller_folds=folds(),
        change='Remove mean-over-agents broadcast of already existing16D per-agent H20 response. All40 global context dimensions retained.',
        unchanged=['dataset', 'normalization', 'initialization', 'parameter count', 'minibatch draw order', '1500steps', 'optimizer', 'pureW1NLL', 'source-only checkpoint selection'],
        seeds=list(SEEDS), selection='PooledheldnativecontrollerVALNLL across3CVfolds and3seeds',
        old_negative='Earlier2/4-controller H20 association controls were not stably beneficial. This test is justified only by new goal information and crossed supervision.',
        no_automatic_target_promotion=True, new_rollouts=0, target_labels_used=False,
        code_sha256=sha(__file__)))


def train(index):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend() == 'gpu'
    protocol = read(OUT / 'protocol.json')
    assert protocol['dataset_sha256'] == sha(DATA / 'dataset.npz')
    fold, seed = index // 3, SEEDS[index % 3]
    dest = OUT / 'cv' / f'fold{fold}' / f'seed{seed}'
    if (dest / 'complete.json').exists():
        return
    held = folds()[fold]
    fitc = [i for i in range(12) if i not in held]
    trainc = fitc + ([12, 13] if 0 in fitc and 1 in fitc else [])
    d, x, tr, va, norm = load(fitc)
    si, e, c = d['state_index'], d['normalized_eta'], d['input_context']
    m = model()
    args = (gather(x, si[:1]), jnp.zeros((1, 3)), jnp.zeros((1, 104)))
    p = m.init(jax.random.PRNGKey(seed), *args)
    baseline_init = mean_model('H20_goal').init(jax.random.PRNGKey(seed), *args)
    assert jax.tree.structure(p) == jax.tree.structure(baseline_init)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(baseline_init)))
    opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(.0008, weight_decay=.0001))
    o = opt.init(p)
    @jax.jit
    def step(p, o, xx, e, c, s, f):
        def loss(pp):
            z = m.apply(pp, xx, e, c)
            q = s / jnp.maximum(s + f, 1)
            return jnp.mean(q * jax.nn.softplus(-z) + (1-q) * jax.nn.softplus(z))
        v, g = jax.value_and_grad(loss)(p)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o, v
    predict = jax.jit(lambda p, xx, e, c: m.apply(p, xx, e, c))
    def evaluate(p, ii, condition='correct'):
        shifted = np.roll(ii.reshape(-1, 2), -1, axis=0).ravel()
        vals = []
        for ci in range(14):
            cc = c[ci, ii].copy()
            xx = gather(x, si[ii])
            ee = e[ii]
            if condition == 'wrong_agent_response':
                cc[:, 24:88] = np.roll(cc[:, 24:88].reshape(-1, 4, 16), 1, axis=1).reshape(-1, 64)
            if condition == 'joint_state_context_shuffle':
                xx, cc = gather(x, si[shifted]), c[ci, shifted]
            if condition == 'state_shuffle':
                xx = gather(x, si[shifted])
            if condition == 'goal_wrong_controller':
                cc[:, 88:] = c[(ci+1) % 12, ii, 88:]
            if condition == 'eta_shuffle':
                ee = e[ii.reshape(-1, 2)[:, ::-1].ravel()]
            vals.append(np.asarray(predict(p, xx, jnp.asarray(ee, jnp.float32), jnp.asarray(cc, jnp.float32))))
        return np.array(vals)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    history, scores = [], {}
    for it in range(1, 1501):
        draw = rng.choice(tr, 32)
        ii, ci = np.tile(draw, len(trainc)), np.repeat(trainc, 32)
        p, o, value = step(p, o, gather(x, si[ii]), jnp.asarray(e[ii], jnp.float32),
                          jnp.asarray(c[ci, ii], jnp.float32), jnp.asarray(d['success'][ci, ii]), jnp.asarray(d['failure'][ci, ii]))
        if it not in STEPS:
            continue
        z = evaluate(p, va)
        scores[f'step{it}'] = z
        (dest / f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        row = dict(step=it, held_native_VAL=metrics(z, d, va, held), seen_native_VAL=metrics(z, d, va, fitc),
                   phase_VAL=metrics(z, d, va, [12, 13]), causal_phase_VAL=causal(z, d, va))
        if it in (75, 300, 700, 1500):
            tz = evaluate(p, tr)
            row['TRAIN'] = metrics(tz, d, tr, trainc)
            scores[f'train_step{it}'] = tz
        history.append(row)
    # Save perturbation predictions at all observed checkpoints; select solely
    # by correct held-native-controller VAL NLL in the summary stage.
    for it in STEPS:
        pp = serialization.msgpack_restore((dest / f'step{it}.msgpack').read_bytes())
        for condition in ('wrong_agent_response', 'joint_state_context_shuffle', 'state_shuffle', 'goal_wrong_controller', 'eta_shuffle'):
            scores[f'step{it}__{condition}'] = evaluate(pp, va, condition)
    write(dest / 'history.json', history)
    write(dest / 'normalization.json', norm)
    np.savez_compressed(dest / 'predictions.npz', train_indices=tr, validation_indices=va, **scores)
    write(dest / 'complete.json', dict(fold=fold, seed=seed, held=held, fit=trainc,
        matched_initialization=True, parameter_count=int(sum(a.size for a in jax.tree.leaves(p))),
        new_rollouts=0, target_labels_used=False, data_sha256=sha(DATA/'dataset.npz'), code_sha256=sha(__file__)))
    print(dict(fold=fold, seed=seed, done=True), flush=True)


def summarize():
    rows, candidates = [], []
    for step in STEPS:
        group = []
        for fold in range(3):
            for seed in SEEDS:
                path = OUT / 'cv' / f'fold{fold}' / f'seed{seed}'
                read(path / 'complete.json')
                r = next(r for r in read(path / 'history.json') if r['step'] == step)
                group.append(dict(fold=fold, seed=seed, **r['held_native_VAL']))
                rows.append(dict(step=step, fold=fold, seed=seed, **r['held_native_VAL']))
        candidates.append(dict(step=step, NLL=float(np.mean([r['NLL'] for r in group])),
                               B15_by_seed=[sum(r['B15'] for r in group if r['seed'] == s) for s in SEEDS]))
    best = min(candidates, key=lambda r: r['NLL'])
    controls = []
    for fold in range(3):
        d, x, tr, va, norm = load([i for i in range(12) if i not in folds()[fold]])
        for seed in SEEDS:
            with np.load(OUT/'cv'/f'fold{fold}'/f'seed{seed}'/'predictions.npz') as ps:
                for condition in ('correct', 'wrong_agent_response', 'joint_state_context_shuffle', 'state_shuffle', 'goal_wrong_controller', 'eta_shuffle'):
                    key = f"step{best['step']}" + ('' if condition == 'correct' else f'__{condition}')
                    z = ps[key]
                    for label, cs in (('held_native', folds()[fold]), ('phase', [12, 13])):
                        controls.append(dict(fold=fold, seed=seed, condition=condition, evaluation=label,
                                             **metrics(z, d, va, cs), **causal(z, d, va)))
    csvwrite(OUT/'source_trajectories.csv', rows)
    csvwrite(OUT/'selected_input_controls.csv', controls)
    write(OUT/'source_selection.json', dict(selection=best, candidates=candidates, target_labels_used=False,
        independent_confirmation=False, existing_mean_reference=read(DATA/'source_selection.json')['selection']['crossed_phase']['H20_goal']))
    print(best, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'train', 'summarize'))
    p.add_argument('--index', type=int, default=0)
    a = p.parse_args()
    if a.action == 'train':
        train(a.index)
    else:
        globals()[a.action]()
