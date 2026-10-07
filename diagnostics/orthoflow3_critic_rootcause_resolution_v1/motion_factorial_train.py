"""Matched input x supervision test of the demonstrated motion-response alias.

Source-only model selection; the independent alias-confirmation families are
excluded. No checkpoint is selected using the intervention outcomes or targets.
All input arms have the same parameter shapes, initialization and batch order.
"""
import argparse
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
from scipy.special import expit
from .motion_factorial_support import OUT, SOURCE, RULE, read, write, sha
from .goal_velocity_train import model, load as previous_load, OUT as PREVIOUS
from .goal_response_cv import csvwrite, folds, SEEDS, STEPS

ARMS = ('native_repeated', 'motion_intervention')
KINDS = ('rest_only', 'rest_motion')
CONTROLS = ('correct', 'motion_parent_replacement', 'motion_wrong_controller',
            'motion_wrong_state', 'rest_wrong_controller', 'context_wrong_state',
            'joint_state_context_shuffle', 'state_shuffle', 'eta_shuffle')


def load(fitc):
    d = dict(np.load(OUT/'dataset.npz'))
    x = dict(np.load(SOURCE/'frozen_model_entities.npz'))
    tr = np.flatnonzero(d['split'] == 'train')
    va = np.flatnonzero(d['split'] == 'validation')
    assert len(tr) == 128 and len(va) == 32 and len(d['success']) == 16
    assert not set(d['state_index'][tr]) & set(d['state_index'][va])
    norm = {'fitting_native_controllers': fitc, 'normalization_intervention_data_used': False}
    ec = d['eta'][tr].mean(0)
    es = np.maximum(d['eta'][tr].std(0), .1)
    norm.update(eta_center=ec.tolist(), eta_scale=es.tolist())
    d['normalized_eta'] = (d['eta']-ec)/es
    features = []
    for name, prefix, axes, minimum in (
        ('context', 'context', (0, 1), .05),
        ('agent_response', 'agent', (0, 1, 2), .01),
        ('goal_response', 'goal', (0, 1), .05),
        ('goal_motion_response', 'goal_motion', (0, 1), .05),
    ):
        values = d[name]
        center = values[fitc][:, tr].mean(axes)
        scale = np.maximum(values[fitc][:, tr].std(axes), minimum)
        norm[prefix+'_center'] = center.tolist()
        norm[prefix+'_scale'] = scale.tolist()
        features.append(((values-center)/scale).reshape(16, 160, -1))
    d['input_context'] = np.concatenate(features, -1).astype(np.float32)
    assert d['input_context'].shape == (16, 160, 120)
    assert np.array_equal(d['input_context'][[14, 15], :, :104], d['input_context'][[0, 1], :, :104])
    assert np.array_equal(d['input_context'][[14, 15], :, 104:], d['input_context'][[1, 0], :, 104:])
    assert np.isfinite(d['input_context']).all()
    return d, x, tr, va, norm


def correlation(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if np.std(a)>1e-9 and np.std(b)>1e-9 else None


def metrics(z, d, ii, controllers):
    s = d['success'][controllers][:, ii]
    f = d['failure'][controllers][:, ii]
    q = s/(s+f)
    zz = z[controllers]
    p = expit(zz)
    shape = (len(controllers), -1, 2)
    ss, ff, qq, pp = [a.reshape(shape) for a in (s, f, q, p)]
    good, bad = ss>=15, ff>=2
    ch = pp.argmax(-1)
    ci, hi = np.arange(len(controllers))[:, None], np.arange(qq.shape[1])[None, :]
    dt, dp = qq[..., 0]-qq[..., 1], pp[..., 0]-pp[..., 1]
    dt = dt-dt.mean(1, keepdims=True)
    dp = dp-dp.mean(1, keepdims=True)
    selected_q, selected_p = qq[ci, hi, ch], pp[ci, hi, ch]
    return dict(
        NLL=float((q*np.logaddexp(0, -zz)+(1-q)*np.logaddexp(0, zz)).mean()),
        MAE=float(abs(p-q).mean()), B15=int(good[ci, hi, ch].sum()),
        non_B15=int(bad[ci, hi, ch].sum()),
        unresolved_selected=int((~(good|bad))[ci, hi, ch].sum()),
        oracle_B15=int(good.any(-1).sum()), cases=int(ch.size),
        selected_Q=float(selected_q.mean()), regret=float((qq.max(-1)-selected_q).mean()),
        severe_false_positive=int(((selected_p>.9)&(selected_q<=.5)).sum()),
        centered_contrast_correlation=correlation(dt, dp),
        centered_contrast_skill=1-float(np.mean((dt-dp)**2))/max(float(np.mean(dt**2)), 1e-12),
        predicted_centered_contrast_std=float(dp.std()), true_centered_contrast_std=float(dt.std()),
    )


def causal(z, d, ii):
    q = d['success'][:, ii]/(d['success'][:, ii]+d['failure'][:, ii])
    p = expit(z)
    dq = q[[14, 15]]-q[[0, 1]]
    dp = p[[14, 15]]-p[[0, 1]]
    td = np.diff(dq.reshape(2, -1, 2), axis=-1)[..., 0]
    pd = np.diff(dp.reshape(2, -1, 2), axis=-1)[..., 0]
    td -= td.mean(1, keepdims=True)
    pd -= pd.mean(1, keepdims=True)
    strong = abs(dq)>=.25
    return dict(
        delta_MAE=float(abs(dq-dp).mean()), delta_correlation=correlation(dq, dp),
        mean_true_abs_delta=float(abs(dq).mean()), mean_predicted_abs_delta=float(abs(dp).mean()),
        strong_cells=int(strong.sum()), strong_sign_correct=int(((np.sign(dp)==np.sign(dq))&strong).sum()),
        centered_causal_interaction_correlation=correlation(td, pd),
        centered_causal_interaction_skill=1-float(np.mean((td-pd)**2))/max(float(np.mean(td**2)), 1e-12),
        predicted_centered_causal_std=float(pd.std()), true_centered_causal_std=float(td.std()),
    )


def train(index):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend() == 'gpu'
    fold, arm, kind, seed = index//12, ARMS[index%12//6], KINDS[index%6//3], SEEDS[index%3]
    assert fold in range(3)
    dest = OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}'
    if (dest/'complete.json').exists():
        return
    held = folds()[fold]
    fitc = [i for i in range(12) if i not in held]
    have = 0 in fitc and 1 in fitc
    trainc = fitc + ([12, 13]+([0, 1] if arm=='native_repeated' else [14, 15]) if have else [])
    d, x, tr, va, norm = load(fitc)
    si, e, c = d['state_index'], d['normalized_eta'], d['input_context']
    m = model(kind)
    p = m.init(jax.random.PRNGKey(seed), gather(x, si[:1]), jnp.zeros((1, 3)), jnp.zeros((1, 120)))
    init_hash = __import__('hashlib').sha256(serialization.to_bytes(p)).hexdigest()
    opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(.0008, weight_decay=.0001))
    o = opt.init(p)
    @jax.jit
    def step(p, o, xx, e, c, s, f):
        def loss(pp):
            z = m.apply(pp, xx, e, c)
            q = s/jnp.maximum(s+f, 1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v, g = jax.value_and_grad(loss)(p)
        u, o = opt.update(g, o, p)
        return optax.apply_updates(p, u), o, v
    predict = jax.jit(lambda pp, xx, ee, cc: m.apply(pp, xx, ee, cc))
    reuse_path = None
    if not have:
        od, ox, ot, ov, on = previous_load(fitc)
        assert np.array_equal(tr, ot) and np.array_equal(va, ov)
        assert all(np.array_equal(x[k], ox[k]) for k in x)
        for key in ('success', 'failure', 'input_context'):
            assert np.array_equal(d[key][:14], od[key])
        assert np.array_equal(d['normalized_eta'], od['normalized_eta'])
        assert norm == on
        reuse_path = PREVIOUS/'cv'/f'fold{fold}'/kind/f'seed{seed}'
        done = read(reuse_path/'complete.json')
        assert done['fit'] == trainc and done['held'] == held
    def evaluate(pp, ii, condition='correct'):
        shift = np.roll(ii.reshape(-1, 2), -1, axis=0).ravel()
        scores = []
        for ci in range(16):
            cc, ss, ee = c[ci, ii].copy(), si[ii], e[ii]
            if condition=='motion_parent_replacement' and ci in (14, 15):
                cc[:, 104:] = c[ci-14, ii, 104:]
            if condition=='motion_wrong_controller':
                cc[:, 104:] = c[(ci+1)%12, ii, 104:]
            if condition=='motion_wrong_state':
                cc[:, 104:] = c[ci, shift, 104:]
            if condition=='rest_wrong_controller':
                cc[:, 88:104] = c[(ci+1)%12, ii, 88:104]
            if condition=='context_wrong_state':
                cc = c[ci, shift]
            if condition=='joint_state_context_shuffle':
                ss, cc = si[shift], c[ci, shift]
            if condition=='state_shuffle':
                ss = si[shift]
            if condition=='eta_shuffle':
                ee = e[ii.reshape(-1, 2)[:, ::-1].ravel()]
            scores.append(np.asarray(predict(pp, gather(x, ss), jnp.asarray(ee, jnp.float32), jnp.asarray(cc, jnp.float32))))
        return np.array(scores)
    rng = np.random.default_rng(seed)
    history, predictions = [], {}
    dest.mkdir(parents=True, exist_ok=True)
    batch_hasher = __import__('hashlib').sha256()
    for it in range(1, 1501):
        draw = rng.choice(tr, 32)
        batch_hasher.update(draw.tobytes())
        if reuse_path is None:
            ii, ci = np.tile(draw, len(trainc)), np.repeat(trainc, 32)
            p, o, v = step(p, o, gather(x, si[ii]), jnp.asarray(e[ii], jnp.float32), jnp.asarray(c[ci, ii], jnp.float32), jnp.asarray(d['success'][ci, ii]), jnp.asarray(d['failure'][ci, ii]))
        if it not in STEPS:
            continue
        if reuse_path is not None:
            p = serialization.msgpack_restore((reuse_path/f'step{it}.msgpack').read_bytes())
        z = evaluate(p, va)
        if kind=='rest_only':
            assert np.array_equal(z[[14, 15]], z[[0, 1]]), 'Old-input alias must remain exact'
        predictions[f'step{it}'] = z
        (dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
        train_z = evaluate(p, tr)
        ts, tf = d['success'][trainc][:, tr], d['failure'][trainc][:, tr]
        tq, tz = ts/(ts+tf), train_z[trainc]
        history.append(dict(step=it, held_native_VAL=metrics(z, d, va, held),
            seen_native_VAL=metrics(z, d, va, fitc), phase_VAL=metrics(z, d, va, [12, 13]),
            motion_VAL=metrics(z, d, va, [14, 15]), causal_motion_VAL=causal(z, d, va),
            TRAIN_NLL=float((tq*np.logaddexp(0, -tz)+(1-tq)*np.logaddexp(0, tz)).mean())))
    for it in STEPS:
        pp = serialization.msgpack_restore((dest/f'step{it}.msgpack').read_bytes())
        for condition in CONTROLS[1:]:
            predictions[f'step{it}__{condition}'] = evaluate(pp, va, condition)
    write(dest/'normalization.json', norm)
    write(dest/'history.json', history)
    np.savez_compressed(dest/'predictions.npz', indices=va, **predictions)
    write(dest/'complete.json', dict(fold=fold, arm=arm, kind=kind, seed=seed, fit=trainc, held=held,
        data_sha256=sha(OUT/'dataset.npz'), protocol_sha256=sha(RULE), code_sha256=sha(__file__),
        shared_model_implementation_sha256=sha(__import__('sys').modules[model.__module__].__file__),
        initial_parameters_sha256=init_hash, minibatch_pair_order_sha256=batch_hasher.hexdigest(),
        parameter_count=int(sum(a.size for a in jax.tree.leaves(p))),
        checkpoint_reuse=str(reuse_path) if reuse_path else None,
        motion_labels_used=have and arm=='motion_intervention', target_labels_used=False, new_rollouts=0))
    print(dict(fold=fold, arm=arm, kind=kind, seed=seed, done=True, reused=reuse_path is not None), flush=True)


def summarize():
    rows, controls, selected, audit = [], [], {}, []
    for arm in ARMS:
        selected[arm] = {}
        for kind in KINDS:
            candidates = []
            for step in STEPS:
                rr = []
                for fold in range(3):
                    for seed in SEEDS:
                        path = OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}'
                        done = read(path/'complete.json')
                        if step==STEPS[0]:
                            audit.append(done)
                        h = next(a for a in read(path/'history.json') if a['step']==step)
                        rr.append(dict(arm=arm, kind=kind, fold=fold, seed=seed, step=step, **h['held_native_VAL']))
                rows += rr
                candidates.append(dict(step=step, NLL=float(np.mean([r['NLL'] for r in rr])),
                    B15_by_seed=[sum(r['B15'] for r in rr if r['seed']==s) for s in SEEDS], cases_per_seed=192))
            best = min(candidates, key=lambda r: r['NLL'])
            selected[arm][kind] = best
            for fold in range(3):
                held = folds()[fold]
                d, x, tr, va, norm = load([i for i in range(12) if i not in held])
                for seed in SEEDS:
                    pp = np.load(OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}'/'predictions.npz')
                    for condition in CONTROLS:
                        key = f"step{best['step']}"+('' if condition=='correct' else f'__{condition}')
                        z = pp[key]
                        for ev, cs in (('held_native', held), ('phase', [12, 13]), ('motion', [14, 15])):
                            controls.append(dict(arm=arm, kind=kind, fold=fold, seed=seed, step=best['step'],
                                condition=condition, evaluation=ev, program_constituents_seen=fold!=2,
                                **metrics(z, d, va, cs), **causal(z, d, va)))
    for fold in range(3):
        for seed in SEEDS:
            rr = [r for r in audit if r['fold']==fold and r['seed']==seed]
            for key in ('initial_parameters_sha256', 'minibatch_pair_order_sha256', 'parameter_count'):
                assert len({r[key] for r in rr})==1, (fold, seed, key)
    csvwrite(OUT/'source_trajectories.csv', rows)
    csvwrite(OUT/'selected_input_controls.csv', controls)
    write(OUT/'matched_training_audit.json', dict(runs=audit, matched_initialization=True,
        matched_minibatch_pair_order=True, matched_parameter_count=True, independent_targets_used=False))
    write(OUT/'source_selection.json', dict(selection=selected, criterion='pooled source-held-native-controller VAL NLL',
        new_training_rollouts=0, target_labels_used=False, independent_confirmation=False,
        caution='These source CV results select checkpoints, not final confirmation. Fold2 excludes the intervention constituents and reuses exact native-only checkpoints.'))
    print(selected, flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'summarize'))
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    train(args.index) if args.action=='train' else summarize()
