"""Read-only prediction analysis: was already-supervised invariance learned?

No fitting, rollout, checkpoint selection, or pipeline modification. A Jensen
gap measures extra NLL from giving different probabilities to this *observed*
outcome-equivalent pair of controller programs. It is not a universal controller
equivalence, and averaging the two predictions is not a deployable new method.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .motion_factorial_train import OUT as MATRIX, SEEDS, STEPS, csvwrite, write
from .motion_freeze_train import OUT as FREEZE


def ce(q, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -q*np.log(p)-(1-q)*np.log1p(-p)


def main():
    out = MATRIX/'rest_invariance_audit'
    data = np.load(MATRIX/'dataset.npz')
    evidence = []
    for split in ('train', 'validation'):
        ii = np.flatnonzero(data['split'] == split)
        for key in ('success', 'failure'):
            assert np.array_equal(data[key][[12, 13]][:, ii], data[key][[14, 15]][:, ii])
        evidence.append(dict(split=split, matched_canonical_pairs=int(2*len(ii)),
                             observed_counts_exactly_equal=True))
    va = np.flatnonzero(data['split'] == 'validation')
    q = data['success'][[12, 13]][:, va]/(
        data['success'][[12, 13]][:, va]+data['failure'][[12, 13]][:, va])
    rows = []
    selected = json.loads((FREEZE/'source_selection.json').read_text())['selection']
    for arm in selected:
        for fold in range(3):
            for seed in SEEDS:
                folder = (MATRIX/'cv'/f'fold{fold}'/'motion_intervention'/'rest_motion'/f'seed{seed}'
                          if arm == 'full_update' else FREEZE/'cv'/f'fold{fold}'/arm/f'seed{seed}')
                pp = np.load(folder/'predictions.npz')
                history = {x['step']: x for x in json.loads((folder/'history.json').read_text())}
                for step in STEPS:
                    p = np.clip(expit(pp[f'step{step}'].astype(np.float64)), 1e-10, 1-1e-10)
                    a, b = p[[12, 13]], p[[14, 15]]
                    nll = (ce(q, a)+ce(q, b))/2
                    collapsed = ce(q, (a+b)/2)
                    assert float((nll-collapsed).min()) >= -2e-6
                    rows.append(dict(arm=arm, fold=fold, seed=seed, step=step,
                        program_labels_in_train=fold != 2,
                        selected_source_checkpoint=step == selected[arm]['step'],
                        mean_probability_difference=float(np.abs(a-b).mean()),
                        paired_program_NLL=float(nll.mean()),
                        paired_program_collapsed_NLL=float(collapsed.mean()),
                        nuisance_Jensen_NLL_gap=float((nll-collapsed).mean()),
                        ranking_flips=int((a.reshape(2, 16, 2).argmax(-1) !=
                                           b.reshape(2, 16, 2).argmax(-1)).sum()),
                        source_held_controller_NLL=history[step]['held_native_VAL']['NLL'],
                        TRAIN_NLL=history[step]['TRAIN_NLL']))
    csvwrite(out/'training_trajectory.csv', rows)
    groups = []
    for arm in selected:
        for step in STEPS:
            rr = [r for r in rows if r['arm'] == arm and r['step'] == step and r['program_labels_in_train']]
            keys = ('mean_probability_difference', 'paired_program_NLL', 'nuisance_Jensen_NLL_gap',
                    'ranking_flips', 'source_held_controller_NLL', 'TRAIN_NLL')
            groups.append(dict(arm=arm, step=step, fits=len(rr),
                               **{k: float(np.mean([r[k] for r in rr])) for k in keys}))
    csvwrite(out/'pooled_training_trajectory.csv', groups)
    write(out/'supervision_audit.json', dict(count_equality=evidence, new_rollouts=0,
        conclusion='Both nuisance variants have exactly the same observed TRAIN targets; absence of these matched labels is not the cause of their differing predictions.',
        limitations=['Results concern these matched stationary programs, not all native controllers.',
                     'Jensen collapse is a diagnostic, not an available test-time solution.',
                     'Six source fits share the same families and are not independent datasets.',
                     'Late checkpoints are not selected using this diagnostic.']))
    print(json.dumps([r for r in groups if r['step'] in (25, 400, 1500)], indent=2))


if __name__ == '__main__':
    main()
