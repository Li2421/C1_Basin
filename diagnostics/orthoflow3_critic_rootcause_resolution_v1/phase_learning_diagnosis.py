"""Source-only decomposition of learning versus controller transfer.

All checkpoint paths and data are already frozen. This is a diagnostic of the
entire trajectory, not a new selection rule or independent confirmation.
Bootstrap units are physical families, jointly across controller conditions.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'phase_factorial_support'
DEST = DATA / 'learning_diagnosis'
SEEDS = (17, 23, 41)
STEPS = (75, 150, 300, 500, 700, 1000, 1500)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csvwrite(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else None


def centered(a):
    return a - a.mean(axis=1, keepdims=True)


def summarize(z, q, s, f, prior, controllers, boot):
    z, q, s, f, prior = [a[controllers] for a in (z, q, s, f, prior)]
    p = expit(z)
    loss = q * np.logaddexp(0, -z) + (1 - q) * np.logaddexp(0, z)
    pp = np.clip(prior, 1e-5, 1 - 1e-5)
    lp = -q * np.log(pp) - (1 - q) * np.log1p(-pp)
    ch = z.argmax(-1)
    cp = pp.argmax(-1)
    c = np.arange(len(controllers))[:, None]
    h = np.arange(q.shape[1])[None, :]
    good, bad = s >= 15, f >= 2
    qa, qb = q[c, h, ch], q[c, h, cp]
    ga, gb, ba, bb = good[c, h, ch], good[c, h, cp], bad[c, h, ch], bad[c, h, cp]
    td = centered(q[..., 0] - q[..., 1])
    pd = centered(p[..., 0] - p[..., 1])
    qr, pr = centered(q), centered(p)
    noise_var = float(np.mean(qr * qr))
    delta = (loss - lp).mean((0, 2))
    state_mse_delta = ((pr - qr) ** 2 - qr ** 2).mean((0, 2))
    return dict(
        NLL=float(loss.mean()), prior_NLL=float(lp.mean()),
        NLL_minus_prior=float(delta.mean()),
        NLL_minus_prior_CI=np.quantile(delta[boot].mean(1), [.025, .975]).tolist(),
        B15=int(ga.sum()), prior_B15=int(gb.sum()), oracle_B15=int(good.any(-1).sum()),
        rescue=int((ga & bb).sum()), breaks=int((ba & gb).sum()), cases=int(ga.size),
        selected_Q=float(qa.mean()), prior_selected_Q=float(qb.mean()),
        selected_Q_delta_CI=np.quantile((qa - qb).mean(0)[boot].mean(1), [.025, .975]).tolist(),
        state_residual_correlation=corr(pr, qr),
        state_residual_skill=1 - float(np.mean((pr - qr) ** 2)) / max(noise_var, 1e-12),
        state_residual_MSE_delta_CI=np.quantile(state_mse_delta[boot].mean(1), [.025, .975]).tolist(),
        state_eta_contrast_correlation=corr(td, pd),
        predicted_state_contrast_std=float(pd.std()), true_state_contrast_std=float(td.std()),
        state_contrast_skill=1 - float(np.mean((pd - td) ** 2)) / max(float(np.mean(td ** 2)), 1e-12),
        centering_note='VAL means used only for retrospective variance decomposition, never predictions or choice',
    )


def main():
    DEST.mkdir(exist_ok=True)
    d = np.load(DATA / 'dataset.npz')
    tr, va = [np.flatnonzero(d['split'] == name) for name in ('train', 'validation')]
    s, f = d['success'][:, va].reshape(14, 16, 2), d['failure'][:, va].reshape(14, 16, 2)
    q = s / (s + f)
    trainq = d['success'][:, tr] / (d['success'][:, tr] + d['failure'][:, tr])
    pri = np.broadcast_to(trainq.reshape(14, 64, 2).mean(1)[:, None], (14, 16, 2))
    boot = np.random.default_rng(202610042135).integers(16, size=(10000, 16))
    rows, causalrows = [], []
    for arm in ('native_repeated', 'crossed_phase'):
        for fold in (0, 1):
            for seed in SEEDS:
                path = DATA / 'cv' / f'fold{fold}' / arm / 'H20_goal' / f'seed{seed}'
                meta = json.loads((path / 'complete.json').read_text())
                held = meta['held_native_controllers']
                seen = [i for i in range(12) if i not in held]
                with np.load(path / 'validation_predictions.npz') as predictions:
                    for step in STEPS:
                        z = predictions[f'step{step}'].reshape(14, 16, 2)
                        for label, cs in (('phase_seen_program_new_family', [12, 13]), ('native_seen_controller_new_family', seen), ('native_held_controller_new_family', held)):
                            rows.append(dict(arm=arm, fold=fold, seed=seed, step=step, evaluation=label, **summarize(z, q, s, f, pri, cs, boot)))
                        p = expit(z)
                        dq, dp = q[[12, 13]] - q[[0, 1]], p[[12, 13]] - p[[0, 1]]
                        tq, pq = centered(dq[..., 0] - dq[..., 1]), centered(dp[..., 0] - dp[..., 1])
                        btq = tq[:, boot].transpose(1, 0, 2)
                        bpq = pq[:, boot].transpose(1, 0, 2)
                        btq -= btq.mean(2, keepdims=True)
                        bpq -= bpq.mean(2, keepdims=True)
                        bc = (btq*bpq).sum((1, 2)) / np.maximum(np.sqrt((btq*btq).sum((1, 2))*(bpq*bpq).sum((1, 2))), 1e-12)
                        causalrows.append(dict(arm=arm, fold=fold, seed=seed, step=step,
                            centered_causal_state_eta_correlation=corr(tq, pq),
                            centered_causal_state_eta_correlation_CI=np.quantile(bc,[.025,.975]).tolist(),
                            centered_causal_state_eta_skill=1 - float(np.mean((tq-pq)**2)) / max(float(np.mean(tq*tq)), 1e-12),
                            predicted_std=float(pq.std()), true_std=float(tq.std()),
                            strong_contrast_cells=int((abs(tq) >= .5).sum()),
                            strong_sign_correct=int(((np.sign(tq) == np.sign(pq)) & (abs(tq) >= .5)).sum())))
    csvwrite(DEST / 'source_trajectory_decomposition.csv', rows)
    csvwrite(DEST / 'source_causal_centered_trajectory.csv', causalrows)
    summary = []
    for arm in ('native_repeated', 'crossed_phase'):
        for step in STEPS:
            rr = [r for r in rows if r['arm'] == arm and r['step'] == step]
            gg = [r for r in causalrows if r['arm'] == arm and r['step'] == step]
            out = dict(arm=arm, step=step, causal_centered_r_mean=float(np.mean([r['centered_causal_state_eta_correlation'] for r in gg])))
            for ev in sorted({r['evaluation'] for r in rr}):
                xx = [r for r in rr if r['evaluation'] == ev]
                out[ev] = {k: float(np.mean([r[k] for r in xx])) for k in ('NLL', 'prior_NLL', 'B15', 'prior_B15', 'oracle_B15', 'state_residual_skill', 'state_eta_contrast_correlation', 'state_contrast_skill')}
                out[ev]['B15_by_fit'] = [r['B15'] for r in xx]
                out[ev]['NLL_gain_CI_excludes_zero_fits'] = sum(r['NLL_minus_prior_CI'][1] < 0 for r in xx)
                out[ev]['residual_gain_CI_excludes_zero_fits'] = sum(r['state_residual_MSE_delta_CI'][1] < 0 for r in xx)
            summary.append(out)
    write(DEST / 'summary.json', dict(
        source_only=True, new_rollouts=0, target_labels_used=False,
        status='POST_HOC_TRAINING_DIAGNOSTIC_NOT_NEW_CHECKPOINT_SELECTION',
        privileged_prior='TRAIN per-controller/program per-eta mean; oracle controller identity. Not an unseen-controller deployment model.',
        replicates='2 controller CV folds times 3 seeds; same 16 VAL families, not 6 independent test datasets.',
        trajectories=summary))
    for r in summary:
        a = r['phase_seen_program_new_family']; b = r['native_held_controller_new_family']
        print(r['arm'], r['step'], 'phase_NLL', round(a['NLL'],4), 'B15', a['B15_by_fit'], 'state_skill', round(a['state_residual_skill'],4), 'causal_r', round(r['causal_centered_r_mean'],4), 'held_NLL', round(b['NLL'],4), flush=True)


if __name__ == '__main__':
    main()
