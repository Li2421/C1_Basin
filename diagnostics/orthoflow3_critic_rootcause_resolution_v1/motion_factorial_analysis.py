"""Family-clustered diagnosis of the frozen motion-factorial source experiment.

This does not retune checkpoints. All model choices come from source-native
controller CV. The controller/eta prior is privileged and explicitly diagnostic.
"""
import argparse
import numpy as np
from scipy.special import expit
from .motion_factorial_train import OUT, ARMS, KINDS, SEEDS, STEPS, folds, read, write, csvwrite
from .phase_learning_diagnosis import summarize as prior_diagnosis, centered, corr


def ci(v, boot):
    return np.quantile(np.asarray(v)[boot].mean(1), [.025, .975]).tolist()


def causal_details(z, q, boot):
    p = expit(z)
    dq = q[[14, 15]]-q[[0, 1]]
    dp = p[[14, 15]]-p[[0, 1]]
    tq, pq = centered(dq[..., 0]-dq[..., 1]), centered(dp[..., 0]-dp[..., 1])
    aa, bb = tq[:, boot].transpose(1, 0, 2), pq[:, boot].transpose(1, 0, 2)
    aa -= aa.mean(2, keepdims=True)
    bb -= bb.mean(2, keepdims=True)
    bc = (aa*bb).sum((1, 2))/np.maximum(np.sqrt((aa*aa).sum((1, 2))*(bb*bb).sum((1, 2))), 1e-12)
    true_base = q[[0, 1], :, 0]-q[[0, 1], :, 1]
    true_motion = q[[14, 15], :, 0]-q[[14, 15], :, 1]
    pred_base = p[[0, 1], :, 0]-p[[0, 1], :, 1]
    pred_motion = p[[14, 15], :, 0]-p[[14, 15], :, 1]
    reversals = (true_base*true_motion<0)&(abs(true_base)>=.25)&(abs(true_motion)>=.25)
    correct = (np.sign(true_base)==np.sign(pred_base))&(np.sign(true_motion)==np.sign(pred_motion))
    return dict(
        centered_causal_r=corr(tq, pq), centered_causal_r_CI=np.quantile(bc, [.025, .975]).tolist(),
        centered_causal_skill=1-float(np.mean((tq-pq)**2))/max(float(np.mean(tq*tq)), 1e-12),
        state_specific_delta_mse_minus_zero=float(np.mean((tq-pq)**2-tq*tq)),
        state_specific_delta_mse_minus_zero_CI=ci(((tq-pq)**2-tq*tq).mean(0), boot),
        true_causal_std=float(tq.std()), predicted_causal_std=float(pq.std()),
        strong_controller_eta_reversals=int(reversals.sum()),
        both_directions_correct=int((correct&reversals).sum()),
        mean_predicted_abs_controller_effect=float(abs(dp).mean()),
        mean_true_abs_controller_effect=float(abs(dq).mean()),
    )


def compare(z, zz, q, s, f, controllers, boot):
    z, zz, q, s, f = [a[controllers] for a in (z, zz, q, s, f)]
    p, pp = expit(z), expit(zz)
    l = q*np.logaddexp(0, -z)+(1-q)*np.logaddexp(0, z)
    ll = q*np.logaddexp(0, -zz)+(1-q)*np.logaddexp(0, zz)
    ch, other = z.argmax(-1), zz.argmax(-1)
    c, h = np.arange(len(controllers))[:, None], np.arange(q.shape[1])[None, :]
    a, b = q[c, h, ch], q[c, h, other]
    good, bad = s>=15, f>=2
    return dict(
        prediction_abs_change=float(abs(p-pp).mean()),
        ranking_flip_count=int((ch!=other).sum()),
        correct_minus_control_NLL=float((l-ll).mean()),
        correct_minus_control_NLL_CI=ci((l-ll).mean((0, 2)), boot),
        correct_minus_control_selected_Q=float((a-b).mean()),
        correct_minus_control_selected_Q_CI=ci((a-b).mean(0), boot),
        correct_B15=int(good[c, h, ch].sum()), control_B15=int(good[c, h, other].sum()),
        rescue=int((good[c, h, ch]&bad[c, h, other]).sum()),
        breaks=int((bad[c, h, ch]&good[c, h, other]).sum()),
        unresolved_correct=int((~(good|bad))[c, h, ch].sum()),
        unresolved_control=int((~(good|bad))[c, h, other].sum()),
    )


def main():
    dest = OUT/'learning_diagnosis'
    dest.mkdir(exist_ok=True)
    selection = read(OUT/'source_selection.json')['selection']
    d = np.load(OUT/'dataset.npz')
    tr, va = [np.flatnonzero(d['split']==a) for a in ('train', 'validation')]
    s, f = [d[key][:, va].reshape(16, 16, 2) for key in ('success', 'failure')]
    q = s/(s+f)
    tq = (d['success'][:, tr]/(d['success'][:, tr]+d['failure'][:, tr])).reshape(16, 64, 2)
    prior = np.broadcast_to(tq.mean(1)[:, None], (16, 16, 2))
    boot = np.random.default_rng(202610042120).integers(16, size=(10000, 16))
    rows, contrasts, controls, effects = [], [], [], []
    selected_keys = {}
    for arm in ARMS:
        for kind in KINDS:
            chosen = selection[arm][kind]['step']
            for fold in range(3):
                held = folds()[fold]
                for seed in SEEDS:
                    path = OUT/'cv'/f'fold{fold}'/arm/kind/f'seed{seed}'
                    pred = np.load(path/'predictions.npz')
                    for step in sorted(set((75, 150, 300, 500, 700, 1000, 1500, chosen))):
                        z = pred[f'step{step}'].reshape(16, 16, 2)
                        meta = dict(arm=arm, kind=kind, fold=fold, seed=seed, step=step,
                            selected=step==chosen, constituents_seen=fold!=2)
                        for ev, cs in (('held_native', held), ('motion', [14, 15])):
                            rows.append(dict(**meta, evaluation=ev, **prior_diagnosis(z, q, s, f, prior, cs, boot)))
                        contrasts.append(dict(**meta, **causal_details(z, q, boot)))
                    z = pred[f'step{chosen}'].reshape(16, 16, 2)
                    selected_keys[(arm, kind, fold, seed)] = z
                    for cond in ('motion_parent_replacement', 'motion_wrong_controller', 'motion_wrong_state',
                                 'context_wrong_state', 'joint_state_context_shuffle', 'state_shuffle', 'eta_shuffle'):
                        zz = pred[f'step{chosen}__{cond}'].reshape(16, 16, 2)
                        for ev, cs in (('held_native', held), ('motion', [14, 15])):
                            controls.append(dict(arm=arm, kind=kind, fold=fold, seed=seed, step=chosen,
                                constituents_seen=fold!=2, condition=cond, evaluation=ev,
                                **compare(z, zz, q, s, f, cs, boot)))
    for fold in range(3):
        for seed in SEEDS:
            combinations = (
                ('motion_feature_given_intervention', ('motion_intervention','rest_motion'), ('motion_intervention','rest_only')),
                ('motion_supervision_given_feature', ('motion_intervention','rest_motion'), ('native_repeated','rest_motion')),
                ('motion_feature_without_intervention', ('native_repeated','rest_motion'), ('native_repeated','rest_only')),
                ('motion_supervision_without_feature', ('motion_intervention','rest_only'), ('native_repeated','rest_only')),
            )
            for name, a, b in combinations:
                for ev, cs in (('held_native', folds()[fold]), ('motion', [14, 15])):
                    effects.append(dict(contrast=name, fold=fold, seed=seed, constituents_seen=fold!=2,
                        evaluation=ev, **compare(selected_keys[(*a, fold, seed)], selected_keys[(*b, fold, seed)], q, s, f, cs, boot)))
    csvwrite(dest/'prior_comparison_trajectories.csv', rows)
    csvwrite(dest/'causal_interaction_trajectories.csv', contrasts)
    csvwrite(dest/'selected_input_replacement.csv', controls)
    csvwrite(dest/'factorial_effects.csv', effects)
    summary = []
    for arm in ARMS:
        for kind in KINDS:
            rr = [r for r in rows if r['arm']==arm and r['kind']==kind and r['selected'] and r['constituents_seen'] and r['evaluation']=='motion']
            cc = [r for r in contrasts if r['arm']==arm and r['kind']==kind and r['selected'] and r['constituents_seen']]
            qq = [r for r in controls if r['arm']==arm and r['kind']==kind and r['constituents_seen'] and r['evaluation']=='motion' and r['condition']=='motion_parent_replacement']
            summary.append(dict(arm=arm, kind=kind, step=selection[arm][kind]['step'],
                held_native_CV_NLL=selection[arm][kind]['NLL'],
                held_native_B15_by_seed=selection[arm][kind]['B15_by_seed'],
                motion_B15_by_fit=[r['B15'] for r in rr],
                motion_privileged_prior_B15=rr[0]['prior_B15'], motion_oracle_B15=rr[0]['oracle_B15'], motion_cases=rr[0]['cases'],
                motion_NLL=float(np.mean([r['NLL'] for r in rr])),
                motion_prior_NLL=rr[0]['prior_NLL'],
                motion_NLL_better_prior_CI_fits=sum(r['NLL_minus_prior_CI'][1]<0 for r in rr),
                motion_state_residual_skill_mean=float(np.mean([r['state_residual_skill'] for r in rr])),
                centered_causal_r_by_fit=[r['centered_causal_r'] for r in cc],
                centered_causal_skill_by_fit=[r['centered_causal_skill'] for r in cc],
                correct_motion_input_NLL_gain_by_fit=[-r['correct_minus_control_NLL'] for r in qq],
                correct_motion_input_B15_net_rescue_by_fit=[r['rescue']-r['breaks'] for r in qq]))
    write(dest/'summary.json', dict(source_only=True, target_labels_used=False, new_rollouts=0,
        checkpoint_selection='Frozen pooled source native-controller VAL NLL; never selected using these program outcomes',
        privileged_prior='TRAIN mean per known controller/program and eta; not an unseen-controller deployment baseline',
        replication='Two controller folds times three seeds on the same16VALfamilies, not six independent tests',
        results=summary))
    print(summary, flush=True)


if __name__=='__main__':
    main()
