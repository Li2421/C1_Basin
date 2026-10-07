"""Frozen independent confirmation: score without labels, then audit outcomes.

Inference is single-thread NumPy and must reproduce cached GPU source logits.
No training, controller search, checkpoint selection, or rollout execution.
"""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from . import cached_numpy_critic as cpu

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'motion_final_source_models'
RULE = ROOT/'motion_independent_confirmation_protocol.json'
CONDITIONS = ('correct', 'wrong_controller_alt', 'joint_state_context_shuffle',
              'state_shuffle', 'all_context_wrong_state')
read, sha = cpu.read, cpu.sha


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csvwrite(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def forward(params, x, eta, context, variant):
    if variant != 'eta_only':
        return cpu.forward(params, x, eta, context, 'rest_motion')
    eta = np.asarray(eta, np.float32)
    size = len(eta)
    h = np.zeros((size, len(params['physical_encoder']['scene']['bias'])), np.float32)
    e = cpu.silu(cpu.dense(eta, params['eta_encoder']))
    c = cpu.silu(cpu.dense(np.zeros((size, 56), np.float32), params['context_encoder']))
    cid = cpu.silu(cpu.dense(np.zeros((size, 3), np.float32), params['id_encoder']))
    z = cpu.silu(cpu.dense(np.concatenate((h, e, c, cid), -1), params['trunk1']))
    return cpu.dense(cpu.silu(cpu.dense(z, params['trunk2'])), params['out'])[..., 0]


def normalized_context(initial, response, norm):
    blocks = []
    for values, prefix in ((initial['context'], 'context'), (initial['agent_response'], 'agent'),
                           (response['goal_response'], 'goal'),
                           (response['goal_motion_response'], 'goal_motion')):
        block = (values-np.array(norm[prefix+'_center']))/np.array(norm[prefix+'_scale'])
        blocks.append(block.reshape(len(block), -1))
    result = np.concatenate(blocks, -1).astype(np.float32)
    assert result.shape[1] == 120 and np.isfinite(result).all()
    return result


def load_entry(entry):
    path = Path(entry['path'])
    assert sha(path/entry['checkpoint']) == entry['checkpoint_sha256']
    assert sha(path/'normalization.json') == entry['normalization_sha256']
    return cpu.load(path/entry['checkpoint']), read(path/'normalization.json')


def source_parity(entries):
    """All arms, all source controllers; no held-target labels or JAX execution."""
    d = dict(np.load(ROOT/'motion_factorial_support/dataset.npz'))
    entities = dict(np.load(ROOT/'controller_function_support/frozen_model_entities.npz'))
    val = np.flatnonzero(d['split'] == 'validation')
    x = {k: v[d['state_index'][val]] for k, v in entities.items()}
    checks = []
    for entry in entries:
        par, norm = load_entry(entry)
        e = (d['eta'][val]-norm['eta_center'])/norm['eta_scale']
        prediction = []
        for ci in range(16):
            values = {k: d[k][ci, val] for k in ('context', 'agent_response', 'goal_response', 'goal_motion_response')}
            context = normalized_context(values, values, norm)
            prediction.append(forward(par, x, e, context, entry['variant']))
        prediction = np.array(prediction)
        expected = np.load(Path(entry['path'])/'predictions.npz')[f"step{entry['steps']}"]
        assert np.allclose(prediction, expected, atol=2e-5, rtol=2e-5)
        assert np.array_equal(prediction.reshape(16, 16, 2).argmax(-1), expected.reshape(16, 16, 2).argmax(-1))
        checks.append(dict(variant=entry['variant'], seed=entry['seed'],
                           max_abs_logit_error=float(abs(prediction-expected).max()), top1_exact=True,
                           predictions=int(prediction.size)))
    return checks


def folder(replica):
    assert replica in (0, 1)
    rule = read(RULE)['target_rule'][replica]
    return ROOT/f"motion_independent_confirmation_{rule['controller_seed']}"


def predict(replica):
    out = folder(replica)
    destination = out/'prediction_freeze.json'
    assert not destination.exists(), 'Target predictions already frozen'
    frozen = read(SOURCE/'models_frozen.json')
    assert read(out/'protocol.json')['models_frozen_sha256'] == sha(SOURCE/'models_frozen.json')
    checks = source_parity(frozen['models'])
    pairs, states = read(out/'pairs.json'), read(out/'states.json')
    assert len(states) == 64 and len(pairs) == 128
    si = np.array([p['state_index'] for p in pairs])
    assert np.array_equal(si, np.repeat(np.arange(64), 2))
    eta = np.array([p['eta'] for p in pairs], np.float32)
    assert np.array_equal(eta.reshape(64, 2, 3), np.broadcast_to(eta[:2], (64, 2, 3)))
    entities = dict(np.load(out/'entities.npz'))
    inp = {name: dict(np.load(out/f'inputs_{name}.npz')) for name in ('held', 'alt')}
    goals = {name: dict(np.load(out/f'goal_both_{name}.npz')) for name in ('held', 'alt')}
    assert all(a['valid'].all() for a in (*inp.values(), *goals.values()))
    shift = np.roll(np.arange(64), -1)
    pairshift = np.column_stack((2*shift, 2*shift+1)).ravel()
    predictions, eta_repeat_checks = {}, []
    for entry in frozen['models']:
        par, norm = load_entry(entry)
        e = (eta-norm['eta_center'])/norm['eta_scale']
        contexts = {name: normalized_context(inp[name], goals[name], norm) for name in inp}
        for condition in CONDITIONS:
            context = contexts['alt' if condition == 'wrong_controller_alt' else 'held']
            if condition in ('joint_state_context_shuffle', 'all_context_wrong_state'):
                context = context[pairshift]
            indices = shift[si] if condition in ('joint_state_context_shuffle', 'state_shuffle') else si
            x = {k: v[indices] for k, v in entities.items()}
            z = forward(par, x, e, context, entry['variant']).reshape(64, 2)
            assert np.isfinite(z).all()
            key = f"{entry['variant']}__{entry['seed']}__{condition}"
            predictions[key] = z
        if entry['variant'] == 'eta_only':
            z = predictions[f"eta_only__{entry['seed']}__correct"]
            # BLAS GEMM row tiles can differ by one float32 ULP for repeated
            # inputs. Require numerical equality AND exactly constant ranking;
            # keep original logits, without any outcome-dependent adjustment.
            assert np.allclose(z, np.broadcast_to(z[0], z.shape), atol=2e-6, rtol=2e-6)
            assert np.all(z.argmax(1) == z[0].argmax())
            eta_repeat_checks.append(dict(seed=entry['seed'],
                max_abs_logit_difference=float(abs(z-z[0]).max()), ranking_exact=True))
            for condition in CONDITIONS:
                assert np.array_equal(z, predictions[f"eta_only__{entry['seed']}__{condition}"])
    np.savez_compressed(out/'frozen_predictions.npz', **predictions)
    filenames = ['protocol.json', 'pairs.json', 'states.json', 'entities.npz', 'planned_rollouts.json',
                 'inputs_held.npz', 'inputs_alt.npz', 'goal_both_held.npz', 'goal_both_alt.npz']
    write(destination, dict(models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
        protocol_sha256=sha(RULE), evaluator_sha256=sha(__file__), numpy_forward_sha256=sha(cpu.__file__),
        input_sha256={name: sha(out/name) for name in filenames},
        predictions_sha256=sha(out/'frozen_predictions.npz'), GPU_source_parity=checks,
        eta_repeat_checks=eta_repeat_checks,
        conditions=CONDITIONS, target_labels_read=False, new_rollouts=0,
        independent_controller_seed=read(out/'protocol.json')['target_controller_seed']))
    print(dict(predictions_frozen=str(destination), models=len(frozen['models']), new_rollouts=0), flush=True)


def correlation(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum()>1 and min(a[ok].std(), b[ok].std())>1e-8 else None


def finite_mean(values):
    values = np.asarray(values)
    ok = np.isfinite(values)
    return float(values[ok].mean()) if ok.any() else None


def measures(z, success, failure):
    s, f = np.asarray(success), np.asarray(failure)
    assert z.shape == s.shape == f.shape and z.shape[1] == 2
    n = s+f
    q = np.divide(s, n, out=np.full(s.shape, np.nan), where=n>0)
    p, chosen = expit(z), z.argmax(1)
    h = np.arange(len(s))
    good, bad = s>=15, f>=2
    lo, hi = s/16, (16-f)/16
    strong = np.where(lo[:, 0]-hi[:, 1]>=.25, 1, np.where(hi[:, 0]-lo[:, 1]<=-.25, -1, 0))
    dt, dp = q[:, 0]-q[:, 1], p[:, 0]-p[:, 1]
    ok = np.isfinite(dt)
    tc, pc = (dt[ok]-dt[ok].mean(), dp[ok]-dp[ok].mean()) if ok.any() else (np.array([]), np.array([]))
    eligible = good.any(1)
    sq = q[h, chosen]
    sqok = np.isfinite(sq)
    loss = s*np.logaddexp(0, -z)+f*np.logaddexp(0, z)
    pair_loss = np.divide(loss, n, out=np.full(s.shape, np.nan), where=n>0)
    reversal = good[:, 0]&bad[:, 1], good[:, 1]&bad[:, 0]
    correct = good[h, chosen]
    return dict(cases=len(s), oracle_B15=int(eligible.sum()), B15=int(correct.sum()),
        non_B15=int(bad[h, chosen].sum()), unresolved=int((~(good|bad))[h, chosen].sum()),
        successful_selection_if_available=float(correct.sum()/eligible.sum()) if eligible.any() else None,
        selected_Q_observed=float(sq[sqok].mean()) if sqok.any() else None,
        selected_Q16_lower=float(lo[h, chosen].mean()), selected_Q16_upper=float(hi[h, chosen].mean()),
        regret=float((q[ok].max(1)-sq[ok]).mean()) if ok.any() else None,
        selected_probability=float(p[h, chosen].mean()),
        severe=int(((p[h, chosen]>.9)&(hi[h, chosen]<=.5)).sum()),
        NLL_trial_weighted=float(loss.sum()/n.sum()) if n.sum() else None,
        NLL_pair_equal=finite_mean(pair_loss), MAE=finite_mean(abs(p-q)),
        centered_contrast_correlation=correlation(dt, dp),
        centered_contrast_skill=1-float(np.mean((tc-pc)**2)/np.mean(tc**2)) if len(tc) and tc.var()>1e-12 else None,
        strong_contrast_cases=int((strong!=0).sum()),
        strong_contrast_correct=int(((np.sign(dp)==strong)&(strong!=0)).sum()),
        eta0_only_B15=int(reversal[0].sum()), eta1_only_B15=int(reversal[1].sum()),
        eta0_only_B15_selected=int((reversal[0]&correct).sum()), eta1_only_B15_selected=int((reversal[1]&correct).sum()),
        hindsight_best_constant_B15=int(good.sum(0).max()),
        oracle_gap=int(eligible.sum()-correct.sum()))


def evaluate(replica):
    out = folder(replica)
    guard = read(out/'prediction_freeze.json')
    assert sha(__file__) == guard['evaluator_sha256']
    assert sha(cpu.__file__) == guard['numpy_forward_sha256']
    assert sha(RULE) == guard['protocol_sha256']
    assert sha(SOURCE/'models_frozen.json') == guard['models_frozen_sha256']
    for name, digest in guard['input_sha256'].items():
        assert sha(out/name) == digest, name
    assert sha(out/'frozen_predictions.npz') == guard['predictions_sha256']
    assert read(out/'working_state.json')['phase'] == 'postflight_complete'
    data = dict(np.load(out/'dataset.npz'))
    assert data['valid'].all()
    s, f, numeric = [data[k].reshape(64, 2) for k in ('success', 'failure', 'numerical')]
    assert np.all(s+f+numeric == 16)
    n = s+f
    q = np.divide(s, n, out=np.full(s.shape, np.nan), where=n>0)
    good, bad = s>=15, f>=2
    pred = dict(np.load(out/'frozen_predictions.npz'))
    frozen = read(SOURCE/'models_frozen.json')
    rows, decisions, comparisons = [], [], []
    pairs = read(out/'pairs.json'); h = np.arange(64)
    boot = np.random.default_rng(202610044032).integers(64, size=(10000, 64))
    for entry in frozen['models']:
        variant, seed = entry['variant'], entry['seed']
        for condition in CONDITIONS:
            z = pred[f'{variant}__{seed}__{condition}']
            rows.append(dict(variant=variant, seed=seed, condition=condition, steps=entry['steps'],
                diagnostic_only=entry['diagnostic_only'], **measures(z, s, f)))
            chosen = z.argmax(1)
            for i, choice in enumerate(chosen):
                pair = pairs[2*i+choice]
                decisions.append(dict(variant=variant, seed=seed, condition=condition,
                    state_uid=pair['state_uid'], eta_uid=pair['eta_uid'], eta_index=int(pair['eta_index']),
                    success=int(s[i, choice]), failure=int(f[i, choice]), numerical=int(numeric[i, choice]),
                    B15=bool(good[i, choice]), non_B15=bool(bad[i, choice]), probability=float(expit(z[i, choice])),
                    eligible=bool(good[i].any())))
        a = pred[f'{variant}__{seed}__correct']; ca = a.argmax(1)
        references = {condition:pred[f'{variant}__{seed}__{condition}'] for condition in CONDITIONS[1:]}
        for reference in ('eta_only', 'full_update', 'trunk_only'):
            references[reference] = pred[f'{reference}__{seed}__correct']
        for reference, b in references.items():
            cb = b.argmax(1)
            ga, gb, ba, bb = good[h, ca], good[h, cb], bad[h, ca], bad[h, cb]
            rescue, breaks = int((ga&bb).sum()), int((ba&gb).sum())
            known = (ga|ba)&(gb|bb)
            delta_b = ga.astype(float)-gb.astype(float)
            # Unknown outcomes retain conservative lower/upper bounds.
            delta_lo = ga.astype(float)-(~bb).astype(float)
            delta_hi = (~ba).astype(float)-gb.astype(float)
            dq = q[h, ca]-q[h, cb]
            qci = np.quantile(dq[boot].mean(1), [.025, .975]) if np.isfinite(dq).all() else [None, None]
            bci = np.quantile(delta_b[boot].mean(1), [.025, .975]) if known.all() else [None, None]
            comparisons.append(dict(variant=variant, seed=seed, reference=reference,
                rescue=rescue, breaks=breaks, net_rescue=rescue-breaks,
                unknown_comparisons=int((~known).sum()), B15_gain_lower=float(delta_lo.mean()),
                B15_gain_upper=float(delta_hi.mean()), B15_gain_CI_low=bci[0], B15_gain_CI_high=bci[1],
                selected_Q_gain=finite_mean(dq), selected_Q_CI_low=qci[0], selected_Q_CI_high=qci[1],
                paired_exact_p=float(binomtest(rescue,rescue+breaks,.5).pvalue) if rescue+breaks else 1.,
                changed_top1=int((ca!=cb).sum()), mean_probability_change=float(abs(expit(a)-expit(b)).mean())))
    csvwrite(out/'metrics.csv', rows); csvwrite(out/'decisions.csv', decisions)
    csvwrite(out/'paired_comparisons.csv', comparisons)
    write(out/'evaluation_audit.json', dict(controller_seed=read(out/'protocol.json')['target_controller_seed'],
        families=64, oracle_B15=int(good.any(1).sum()), both_B15=int(good.all(1).sum()),
        coverage_failure=int((~good.any(1)).sum()), fixed_B15=good.sum(0).tolist(),
        numerical=int(numeric.sum()), collision=read(out/'alignment_audit.json')['collision'],
        predictions_frozen_before_labels=True, target_labels_used_for_selection=False,
        models_frozen_sha256=guard['models_frozen_sha256'], dataset_sha256=sha(out/'dataset.npz'),
        probability_semantics='Observed Q=s/(s+f); Q16 bounds=[s/16,(16-f)/16]. Numerical outcomes never imputed.',
        uncertainty='Family bootstrap within each controller. Training seeds are not independent families; no across-controller pooling.',
        controller_intervention_semantics='The existing common base Flow supplies the committed t0 query; the new held Flow supplies subsequent queries. This is future-controller transfer, not an entirely unseen base-policy pipeline.',
        scope=read(out/'protocol.json')['scope'], new_evaluation_rollouts=0))
    print([r for r in rows if r['condition']=='correct'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('predict', 'evaluate'))
    p.add_argument('--replicate', type=int, default=0)
    a = p.parse_args()
    globals()[a.action](a.replicate)
