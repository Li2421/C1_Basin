"""Source-only nested real-Flow checkpoint matching on archived P1 Q4 labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / 'controller_q_transfer_round3_v1'
R2 = HERE.parent / 'part3_necessity_representation_round2_20261012'
ARMS = ('R0', 'R3', 'R5', 'R3_near', 'R3_R4', 'R5_R4')
BETAS = (0.5, 1.0, 2.0, 4.0)
MATCH = (0.25, 0.5, 0.75, 1.0)
METRICS = ('coordinate', 'block_rms')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    protocol = json.loads((P1 / 'P1_PROTOCOL.json').read_text())
    names = protocol['controller_order']
    with np.load(P1 / 'P1_LABELS.npz') as z:
        assert z['controller_names'].tolist() == names
        assert z['state_uids'].tolist() == [s['state_uid'] for s in protocol['states']]
        y, u = z['Y'].copy(), z['U'].copy()
    assert y.shape == u.shape == (14, 24, 16, 4)
    assert np.all((y >= 0) & (u >= 0) & (y + u <= 1))
    assert np.all(y + u <= 1)
    blocks = {arm: [] for arm in ARMS if arm != 'R0'}
    # Each semantic block pools the same type across all four agents.
    for name in names:
        with np.load(P1 / 'p1_features' / (name + '.npz')) as old, np.load(R2 / 'features' / (name + '.npz')) as new:
            assert old['state_uids'].tolist() == new['state_uids'].tolist()
            r3 = old['R3'].astype(np.float64).reshape(24, 4, 64)
            r5 = new['R5'].astype(np.float64).reshape(24, 4, 238)
            assert np.max(np.abs(r3 - r5[:, :, :64])) < 1e-5
            r4 = old['R4_action'].astype(np.float64)
            base = [r3[:, :, :14].reshape(24, -1), r3[:, :, 14:56].reshape(24, -1),
                    r3[:, :, 56:60].reshape(24, -1), r3[:, :, 60:64].reshape(24, -1)]
            indexed = r5[:, :, 64:148].reshape(24, 4, 7, 3, 2, 2)
            near = indexed[:, :, 4:7].reshape(24, -1)
            extra = [r5[:, :, 64:148].reshape(24, -1), r5[:, :, 148:238].reshape(24, -1)]
            r4blocks = [r4[:, :32], r4[:, 32:]]
            for arm, bs in [('R3', base), ('R5', base + extra), ('R3_near', base + [near]),
                            ('R3_R4', base + r4blocks), ('R5_R4', base + extra + r4blocks)]:
                blocks[arm].append(bs)
    for arm in blocks:
        blocks[arm] = [np.stack([blocks[arm][c][k] for c in range(14)]) for k in range(len(blocks[arm][0]))]
        assert all(b.shape[:2] == (14, 24) and np.isfinite(b).all() for b in blocks[arm])
    lo = y.mean(-1)
    hi = (y + u).mean(-1)
    mid = (lo + hi) / 2
    return names, blocks, y, u, lo, hi, mid


def distances(blocks, train, held, metric):
    parts, srcparts = [], []
    for block in blocks:
        source = block[train]
        mean = source.mean(axis=(0, 1))
        std = source.std(axis=(0, 1))
        scale = np.maximum(std, 0.05) if metric == 'coordinate' else max(float(np.sqrt(np.mean(std**2))), 0.05)
        parts.append((source - block[held][None]) / scale)
        srcparts.append((source - mean) / scale)
    if metric == 'coordinate':
        d = sum(np.sum(part**2, axis=2) for part in parts) / sum(p.shape[2] for p in parts)
        pairvalues = []
        for i in range(len(train)):
            for j in range(i + 1, len(train)):
                pairvalues.append(sum(np.sum((p[i] - p[j])**2, axis=1) for p in srcparts) / sum(p.shape[2] for p in srcparts))
    else:
        d = sum(np.mean(part**2, axis=2) for part in parts) / len(parts)
        pairvalues = []
        for i in range(len(train)):
            for j in range(i + 1, len(train)):
                pairvalues.append(sum(np.mean((p[i] - p[j])**2, axis=1) for p in srcparts) / len(srcparts))
    pairvalues = np.concatenate(pairvalues)
    positive = pairvalues[pairvalues > 1e-12]
    median = float(np.median(positive)) if len(positive) else 1.0
    assert np.isfinite(median) and median > 0
    return d / median


def predictions(blocks, q, train, held, metric):
    blind = q[train].mean(axis=0)
    d = distances(blocks, train, held, metric)
    out = {}
    for beta in BETAS:
        logits = -beta * d
        w = np.exp(logits - logits.max(axis=0, keepdims=True))
        w /= w.sum(axis=0, keepdims=True)
        matched = np.einsum('cs,csk->sk', w, q[train], optimize=True)
        for match in MATCH:
            out[(metric, beta, match)] = (1 - match) * blind + match * matched
    return out


def chosen_configuration(blocks, q, outer_sources):
    errs = {key: [] for key in ((m, b, a) for m in METRICS for b in BETAS for a in MATCH)}
    for inner_held in outer_sources:
        train = [i for i in outer_sources if i != inner_held]
        for metric in METRICS:
            for key, pred in predictions(blocks, q, train, inner_held, metric).items():
                errs[key].append(float(np.mean((pred - q[inner_held])**2)))
    scores = {key: float(np.mean(v)) for key, v in errs.items()}
    best = min(scores, key=lambda key: (scores[key], METRICS.index(key[0]), key[1], key[2]))
    return best, scores


def values(q, idx):
    return q[np.arange(q.shape[0]), idx]


def paired_gain(sel, ref, lo, hi):
    a_lo, a_hi = values(lo, sel), values(hi, sel)
    b_lo, b_hi = values(lo, ref), values(hi, ref)
    lower = np.where(sel == ref, 0, a_lo - b_hi)
    upper = np.where(sel == ref, 0, a_hi - b_lo)
    return [float(lower.mean()), float(upper.mean())]


def ranking(pred, q, lo, hi):
    ranks = []
    correct = total = 0
    for s in range(24):
        score = spearmanr(pred[s], q[s]).statistic
        if np.isfinite(score):
            ranks.append(float(score))
        for i in range(16):
            for j in range(i + 1, 16):
                sign = 1 if lo[s, i] > hi[s, j] else (-1 if lo[s, j] > hi[s, i] else 0)
                if sign:
                    total += 1
                    correct += int(sign * (pred[s, i] - pred[s, j]) > 0)
    return {'mean_state_spearman': float(np.mean(ranks)) if ranks else None,
            'states_with_rank': len(ranks), 'certain_pair_accuracy': correct / total if total else None,
            'certain_pair_count': total}


def score(pred, lo, hi, mid, fixed, refsel=None):
    sel = np.argmax(pred, axis=1)
    sqmin = np.maximum(0, np.maximum(lo - pred, pred - hi))**2
    sqmax = np.maximum((lo - pred)**2, (hi - pred)**2)
    regret_lo, regret_hi = [], []
    for s, j in enumerate(sel):
        others = [k for k in range(16) if k != j]
        regret_lo.append(max(0., float(lo[s, others].max() - hi[s, j])))
        regret_hi.append(max(0., float(hi[s, others].max() - lo[s, j])))
    ans = {'q_mse_mid': float(np.mean((pred - mid)**2)),
           'q_mse_unknown_safe': [float(sqmin.mean()), float(sqmax.mean())],
           'top1_Q_interval': [float(values(lo, sel).mean()), float(values(hi, sel).mean())],
           'empirical_oracle_regret_interval': [float(np.mean(regret_lo)), float(np.mean(regret_hi))],
           'gain_over_source_fixed_interval': paired_gain(sel, np.full(24, fixed), lo, hi),
           'selected_eta': sel.tolist(), **ranking(pred, mid, lo, hi)}
    if refsel is not None:
        ans['gain_over_R0_interval'] = paired_gain(sel, refsel, lo, hi)
    err = pred - mid
    level = err.mean()
    state = err.mean(axis=1) - level
    eta = err.mean(axis=0) - level
    residual = err - level - state[:, None] - eta[None, :]
    ans['mse_decomposition'] = {'controller_level': float(level**2), 'state_main': float(np.mean(state**2)),
                                'eta_curve': float(np.mean(eta**2)), 'state_eta_residual': float(np.mean(residual**2))}
    return ans


def checkpoint_bootstrap(x, n=20000, seed=20261012):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    ids = rng.integers(0, len(x), (n, len(x)))
    return [float(v) for v in np.quantile(x[ids].mean(axis=1), [.025, .975])]


def crossfit_headroom(y, u, fixed):
    # Target-only diagnostic: half-seed oracle chosen on one half, evaluated on the other.
    gains = []
    for train, test in [([0, 1], [2, 3]), ([2, 3], [0, 1])]:
        choice = np.argmax((y[:, :, train] + .5*u[:, :, train]).mean(axis=-1), axis=1)
        qlo = y[:, :, test].mean(-1)
        qhi = (y[:, :, test] + u[:, :, test]).mean(-1)
        gains.append(paired_gain(choice, np.full(24, fixed), qlo, qhi))
    return [float(np.mean([g[k] for g in gains])) for k in (0, 1)]


def main():
    names, blocks, y, u, lo, hi, q = load()
    hashes = {'preregistration': sha(HERE / 'PREREGISTRATION.md'),
              'p1_labels': sha(P1 / 'P1_LABELS.npz'),
              'p1_protocol': sha(P1 / 'P1_PROTOCOL.json')}
    preds = np.zeros((len(ARMS), 14, 24, 16), np.float64)
    rows = []
    for held, name in enumerate(names):
        train = [i for i in range(14) if i != held]
        fixed = int(np.argmax(q[train].mean(axis=(0, 1))))
        preds[0, held] = q[train].mean(axis=0)
        chosen = {}
        inner = {}
        for a, arm in enumerate(ARMS[1:], 1):
            best, scores = chosen_configuration(blocks[arm], q, train)
            chosen[arm] = {'metric': best[0], 'inverse_temperature': best[1], 'match_weight': best[2]}
            inner[arm] = {'chosen_MSE': scores[best], 'R0_MSE': float(np.mean([
                np.mean((q[[i for i in train if i != inner_held]].mean(axis=0)-q[inner_held])**2)
                for inner_held in train]))}
            preds[a, held] = predictions(blocks[arm], q, train, held, best[0])[best]
        r0sel = np.argmax(preds[0, held], axis=1)
        arm_scores = {arm: score(preds[a, held], lo[held], hi[held], q[held], fixed,
                                 None if arm == 'R0' else r0sel) for a, arm in enumerate(ARMS)}
        complete = u[held].sum(-1) == 0
        halfdiff = (y[held, :, :, :2].mean(-1) - y[held, :, :, 2:].mean(-1))**2 / 4
        row = {'held': name, 'source_fixed_eta': fixed,
               'source_fixed_Q_interval': [float(lo[held, :, fixed].mean()), float(hi[held, :, fixed].mean())],
               'target_Q4_oracle_interval': [float(lo[held].max(axis=1).mean()), float(hi[held].max(axis=1).mean())],
               'target_half_crossfit_oracle_minus_source_fixed': crossfit_headroom(y[held], u[held], fixed),
               'complete_cell_Q4_noise_proxy_MSE': float(halfdiff[complete].mean()),
               'complete_cells': int(complete.sum()), 'unknown_seeds': int(u[held].sum()),
               'chosen': chosen, 'inner': inner, 'arms': arm_scores}
        rows.append(row)
        print(json.dumps({'held': name, 'fixed': fixed, 'top1': {a: round(arm_scores[a]['top1_Q_interval'][0], 4) for a in ARMS},
                          'MSE': {a: round(arm_scores[a]['q_mse_mid'], 4) for a in ARMS}}), flush=True)
    state_uids = [s['state_uid'] for s in json.loads((P1 / 'P1_PROTOCOL.json').read_text())['states']]
    np.savez_compressed(HERE / 'outer_predictions.npz', predictions=preds, arms=np.array(ARMS),
                        controller_names=np.array(names), state_uids=np.array(state_uids))
    summary = {}
    for arm in ARMS:
        vals = [r['arms'][arm] for r in rows]
        summary[arm] = {'mean_Q_MSE_mid': float(np.mean([v['q_mse_mid'] for v in vals])),
                        'mean_Q_MSE_unknown_safe': [float(np.mean([v['q_mse_unknown_safe'][k] for v in vals])) for k in (0, 1)],
                        'mean_rank_spearman': float(np.mean([v['mean_state_spearman'] for v in vals if v['mean_state_spearman'] is not None])),
                        'mean_certain_pair_accuracy': float(np.mean([v['certain_pair_accuracy'] for v in vals if v['certain_pair_accuracy'] is not None])),
                        'mean_top1_Q_interval': [float(np.mean([v['top1_Q_interval'][k] for v in vals])) for k in (0, 1)],
                        'mean_oracle_regret_interval': [float(np.mean([v['empirical_oracle_regret_interval'][k] for v in vals])) for k in (0, 1)],
                        'mean_gain_over_source_fixed_interval': [float(np.mean([v['gain_over_source_fixed_interval'][k] for v in vals])) for k in (0, 1)]}
        if arm != 'R0':
            summary[arm]['mean_gain_over_R0_interval'] = [float(np.mean([v['gain_over_R0_interval'][k] for v in vals])) for k in (0, 1)]
            for ref, key in [('R0', 'gain_over_R0_interval'), ('source_fixed', 'gain_over_source_fixed_interval')]:
                lower = [v[key][0] for v in vals]
                upper = [v[key][1] for v in vals]
                summary[arm][f'{ref}_nonnegative_lower_checkpoint_count'] = int(np.sum(np.array(lower) >= 0))
                summary[arm][f'{ref}_gain_lower_bootstrap_95'] = checkpoint_bootstrap(lower)
                summary[arm][f'{ref}_gain_upper_bootstrap_95'] = checkpoint_bootstrap(upper)
    baseline_mse = summary['R0']['mean_Q_MSE_mid']
    r3_mse = summary['R3']['mean_Q_MSE_mid']
    passed = []
    for arm in ARMS[1:]:
        s = summary[arm]
        if (s['mean_Q_MSE_mid'] < min(baseline_mse, r3_mse) and
                s['mean_gain_over_R0_interval'][0] > 0 and s['mean_gain_over_source_fixed_interval'][0] > 0 and
                s['R0_nonnegative_lower_checkpoint_count'] >= 10 and
                s['source_fixed_nonnegative_lower_checkpoint_count'] >= 10 and
                s['R0_gain_lower_bootstrap_95'][0] > 0 and
                s['source_fixed_gain_lower_bootstrap_95'][0] > 0):
            passed.append(arm)
    out = {'status': 'TRANSFER_GATE_PASSED' if passed else 'TRANSFER_GATE_FAILED', 'passing_arms': passed,
           'scope': 'Opened P1 24 development states and 14 selected real Flow checkpoints; outer LOCO source-only, not a fresh-label confirmation.',
           'hashes': hashes, 'new_rollouts': 0, 'arms': list(ARMS),
           'unknown_seeds': int(u.sum()), 'mean_complete_cell_Q4_noise_proxy_MSE': float(np.mean([r['complete_cell_Q4_noise_proxy_MSE'] for r in rows])),
           'mean_target_half_crossfit_oracle_minus_source_fixed_interval': [float(np.mean([r['target_half_crossfit_oracle_minus_source_fixed'][k] for r in rows])) for k in (0, 1)],
           'summary': summary, 'per_checkpoint': rows}
    (HERE / 'results.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': out['status'], 'summary': summary}, indent=2), flush=True)


if __name__ == '__main__':
    main()
