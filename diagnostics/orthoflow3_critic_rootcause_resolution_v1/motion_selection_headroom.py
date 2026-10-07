"""Cached source-only adjudication: controller preference versus state adaptation.

No fitting, checkpoint selection, environment step, or rollout DB mutation.
The per-controller TRAIN prior is a privileged diagnostic, NOT zero-shot.
The best constant eta computed on VAL is an optimistic hindsight bound, NOT
a deployable baseline. Comparing to both prevents global/controller preference
learning from being mistaken for state-specific robust selection.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'motion_factorial_support'
OUT = ROOT / 'motion_selection_headroom'
SEEDS = (17, 23, 41)


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csvwrite(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(z, q, s, f):
    p = expit(z)
    choice = z.argmax(-1)
    ci, hi = np.arange(len(z))[:, None], np.arange(z.shape[1])[None, :]
    good, bad = s >= 15, f >= 2
    selected_q = q[ci, hi, choice]
    selected_good = good[ci, hi, choice]
    selected_bad = bad[ci, hi, choice]
    true_contrast = q[..., 0] - q[..., 1]
    pred_contrast = p[..., 0] - p[..., 1]
    tc = true_contrast - true_contrast.mean(1, keepdims=True)
    pc = pred_contrast - pred_contrast.mean(1, keepdims=True)
    loss = q * np.logaddexp(0, -z) + (1 - q) * np.logaddexp(0, z)
    skill = 1 - ((pc - tc) ** 2).mean() / max((tc ** 2).mean(), 1e-12)
    corr = float(np.corrcoef(tc.ravel(), pc.ravel())[0, 1]) if pc.std() > 1e-10 else None
    return dict(
        cases=int(choice.size), B15=int(selected_good.sum()),
        non_B15=int(selected_bad.sum()),
        unresolved_selected=int((~(selected_good | selected_bad)).sum()),
        oracle_B15=int(good.any(-1).sum()), NLL=float(loss.mean()),
        selected_Q=float(selected_q.mean()),
        regret=float((q.max(-1) - selected_q).mean()),
        state_centered_contrast_correlation=corr,
        state_centered_contrast_skill=float(skill),
        true_state_contrast_std=float(tc.std()),
        predicted_state_contrast_std=float(pc.std()),
        controllers_using_both_eta=int(((choice == 0).any(1) & (choice == 1).any(1)).sum()),
        strong_Q_contrast_cases=int((abs(true_contrast) >= .25).sum()),
        strong_Q_contrast_correct=int(((np.sign(pred_contrast) == np.sign(true_contrast)) & (abs(true_contrast) >= .25)).sum()),
    )


def paired(z, zz, q, s, f, boot):
    ci, hi = np.arange(len(z))[:, None], np.arange(z.shape[1])[None, :]
    ca, cb = z.argmax(-1), zz.argmax(-1)
    good, bad = s >= 15, f >= 2
    ga, gb = good[ci, hi, ca], good[ci, hi, cb]
    ba, bb = bad[ci, hi, ca], bad[ci, hi, cb]
    difference = (q[ci, hi, ca] - q[ci, hi, cb]).mean(0)
    bdiff = (ga.astype(float) - gb.astype(float)).mean(0)
    return dict(rescue=int((ga & bb).sum()), breaks=int((ba & gb).sum()),
                unresolved_discordance=int(((ga != gb) & ~((ga & bb) | (ba & gb))).sum()),
                selected_Q_delta=float(difference.mean()),
                selected_Q_delta_CI=np.quantile(difference[boot].mean(1), [.025, .975]).tolist(),
                B15_rate_delta=float(bdiff.mean()),
                B15_rate_delta_CI=np.quantile(bdiff[boot].mean(1), [.025, .975]).tolist())


def checkpoint(arm, fold, seed, choices):
    if arm == 'old_rest_only':
        folder = DATA/'cv'/f'fold{fold}'/'native_repeated'/'rest_only'/f'seed{seed}'
    elif arm == 'full_update':
        folder = DATA/'cv'/f'fold{fold}'/'motion_intervention'/'rest_motion'/f'seed{seed}'
    elif arm == 'trunk_only':
        folder = ROOT/'motion_freeze_control/cv'/f'fold{fold}'/arm/f'seed{seed}'
    elif arm == 'drop_rest':
        folder = DATA/'cv'/f'fold{fold}'/'motion_intervention'/'motion_only'/f'seed{seed}'
    else:
        assert arm == 'eta_only'
        folder = ROOT/'motion_eta_control/cv'/f'fold{fold}'/f'seed{seed}'
    done = read(folder/'complete.json')
    held = done['held']
    with np.load(folder/'predictions.npz') as values:
        z = values[f'step{choices[arm]}'].reshape(16, 16, 2)
    return z, held, folder


def regression_headroom():
    folder = ROOT/'motion_counterexample_regression'
    with (folder/'per_pair_predictions.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    counts = {}
    for row in rows:
        key = row['state_uid'], row['eta_uid']
        value = int(row['parent_success']), int(row['motion_success'])
        assert key not in counts or counts[key] == value
        counts[key] = value
    pairs = read(folder/'pairs.json')
    assert len(counts) == len(pairs) == 16
    success = np.array([counts[p['state_uid'], p['eta_uid']] for p in pairs]).T.reshape(2, 8, 2)
    good = success >= 15
    prior = np.array(read(folder/'audit.json')['privileged_TRAIN_prior_probabilities'])
    chosen = prior.argmax(-1)
    ci, hi = np.arange(2)[:, None], np.arange(8)[None, :]
    out = dict(cases=16, families=8, cached_exact_rollouts=512, new_rollouts=0,
               labels_previously_opened=True, independent_confirmation=False,
               B15_by_condition_eta=good.sum(1).tolist(),
               oracle_B15=int(good.any(-1).sum()),
               best_global_fixed_B15_hindsight=int(good.sum((0, 1)).max()),
               privileged_TRAIN_prior_B15=int(good[ci, hi, chosen[:, None]].sum()),
               strong_counterexample_counts=[int(success[0, 0, 1]), int(success[1, 0, 1])],
               explanation='Counterexample repairs test causal Q prediction. If fixed eta reaches oracle, aggregate B15 cannot test state-adaptive selection on this panel.')
    write(OUT/'counterexample_headroom.json', out)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    source_files = dict(
        factorial=DATA/'source_selection.json',
        freeze=ROOT/'motion_freeze_control/source_selection.json',
        drop_rest=ROOT/'motion_drop_rest_control/source_selection.json',
        eta_only=ROOT/'motion_eta_control/source_selection.json')
    choices = dict(
        old_rest_only=read(source_files['factorial'])['selection']['native_repeated']['rest_only']['step'],
        full_update=read(source_files['freeze'])['selection']['full_update']['step'],
        trunk_only=read(source_files['freeze'])['selection']['trunk_only']['step'],
        drop_rest=read(source_files['drop_rest'])['selection']['step'],
        eta_only=read(source_files['eta_only'])['neural']['step'])
    d = np.load(DATA/'dataset.npz')
    tr, va = [np.flatnonzero(d['split'] == key) for key in ('train', 'validation')]
    s, f = [d[key][:, va].reshape(16, 16, 2) for key in ('success', 'failure')]
    assert np.all(s + f > 0)
    q = s / (s + f)
    trainq = (d['success'][:, tr] / (d['success'][:, tr] + d['failure'][:, tr])).reshape(16, 64, 2)
    # This prior has TRAIN labels from every evaluated controller and is privileged.
    prior = np.broadcast_to(logit(np.clip(trainq.mean(1), 1e-5, 1-1e-5))[:, None], q.shape)
    good, bad = s >= 15, f >= 2
    hindsight_choice = good.sum(1).argmax(-1)
    hindsight = np.broadcast_to(np.eye(2)[hindsight_choice][:, None], q.shape)
    boot = np.random.default_rng(202610041417).integers(16, size=(10000, 16))
    coverage = []
    for c in range(16):
        sole0, sole1 = good[c, :, 0] & bad[c, :, 1], good[c, :, 1] & bad[c, :, 0]
        delta = q[c, :, 0] - q[c, :, 1]
        coverage.append(dict(controller=c, controller_type='native' if c < 12 else 'source_program',
                             families=16, pairs=32, full_Q16_pairs=int(((s[c]+f[c])==16).sum()),
                             numerical_incomplete_pairs=int(((s[c]+f[c])!=16).sum()),
                             B15_eta0=int(good[c, :, 0].sum()), B15_eta1=int(good[c, :, 1].sum()),
                             oracle_B15=int(good[c].any(-1).sum()),
                             known_TRAIN_prior_eta=int(prior[c, 0].argmax()),
                             known_TRAIN_prior_B15=int(good[c, :, prior[c, 0].argmax()].sum()),
                             best_constant_B15_hindsight=int(good[c].sum(0).max()),
                             state_adaptation_B15_headroom=int(good[c].any(-1).sum()-good[c].sum(0).max()),
                             only_eta0_B15=int(sole0.sum()), only_eta1_B15=int(sole1.sum()),
                             robust_ranking_reversal_pairs=int(sole0.sum()*sole1.sum()),
                             empirical_Q_margin_reversal_pairs=int((delta>=.25).sum()*(delta<=-.25).sum())))
    metrics, comparisons, percontroller, provenance = [], [], [], []
    for seed in SEEDS:
        stitched = {}
        for arm in choices:
            z = np.zeros((12, 16, 2))
            assigned = []
            for fold in range(3):
                values, held, folder = checkpoint(arm, fold, seed, choices)
                z[held] = values[held]
                assigned += held
                provenance.append(dict(arm=arm, seed=seed, fold=fold, checkpoint_step=choices[arm],
                                       complete_sha256=digest(folder/'complete.json'), predictions_sha256=digest(folder/'predictions.npz')))
            assert sorted(assigned) == list(range(12))
            stitched[arm] = z
            metrics.append(dict(arm=arm, seed=seed, **stats(z, q[:12], s[:12], f[:12])))
            for c in range(12):
                percontroller.append(dict(arm=arm, seed=seed, controller=c,
                    **stats(z[c:c+1], q[c:c+1], s[c:c+1], f[c:c+1])))
        for arm, z in stitched.items():
            for name, ref in (('matched_eta_only', stitched['eta_only']),
                              ('privileged_controller_TRAIN_prior', prior[:12]),
                              ('hindsight_best_constant_per_controller', hindsight[:12])):
                comparisons.append(dict(arm=arm, seed=seed, reference=name,
                    **paired(z, ref, q[:12], s[:12], f[:12], boot)))
    csvwrite(OUT/'coverage.csv', coverage)
    csvwrite(OUT/'metrics.csv', metrics)
    csvwrite(OUT/'paired_comparisons.csv', comparisons)
    csvwrite(OUT/'per_controller_metrics.csv', percontroller)
    regression = regression_headroom()
    native = coverage[:12]
    summary = dict(
        new_rollouts=0, new_fits=0, target_labels_used=False, checkpoint_selection_changed=False,
        data_sha256=digest(DATA/'dataset.npz'), source_selection_hashes={k:digest(v) for k,v in source_files.items()},
        checkpoint_provenance=provenance, cases=192, unique_VAL_families=16,
        scope='Development/source-controller cross-validation, not independent target confirmation. Shared physical families across controllers; family-bootstrap jointly resamples them.',
        privileged_prior_caveat='Uses each held controller TRAIN labels; not zero-shot. Included to isolate state-specific gain from controller-specific eta preference.',
        hindsight_caveat='Best constant per controller uses VAL outcomes only as an optimistic diagnostic bound; never a deployed selector or checkpoint selection criterion.',
        numerical_semantics='No numerical result fabricated. B15 if observed successes>=15; non-B15 if observed failures>=2; other incomplete cases remain unresolved. Q is observed fraction, not fabricated full Q16.',
        natural_controllers_with_B15_reversals=sum(r['robust_ranking_reversal_pairs']>0 for r in native),
        oracle_B15=sum(r['oracle_B15'] for r in native),
        known_controller_TRAIN_prior_B15=sum(r['known_TRAIN_prior_B15'] for r in native),
        best_constant_per_controller_B15_hindsight=sum(r['best_constant_B15_hindsight'] for r in native),
        B15_headroom_beyond_best_constant=sum(r['state_adaptation_B15_headroom'] for r in native),
        counterexample_regression=regression,
        source_models=metrics)
    write(OUT/'audit.json', summary)
    print(json.dumps({k:summary[k] for k in ('oracle_B15','known_controller_TRAIN_prior_B15','best_constant_per_controller_B15_hindsight','B15_headroom_beyond_best_constant','natural_controllers_with_B15_reversals','counterexample_regression','source_models')}, indent=2))


if __name__ == '__main__':
    main()
