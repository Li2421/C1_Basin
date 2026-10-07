"""Forensic trajectory, not checkpoint selection, on already-opened alias data.

Final source fitting choices were frozen before this audit. Never promote a
checkpoint based on these regression labels; report the entire registered grid.
"""
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy.special import expit
from .cached_numpy_critic import ROOT, regression_predictions, read, sha

OUT = ROOT/'motion_causal_fit_trajectory'
STEPS = (25, 75, 150, 400, 700, 1500)


def main():
    OUT.mkdir(exist_ok=True)
    selection = ROOT/'motion_final_source_models/selection_frozen.json'
    assert selection.exists(), 'Lock subsequent model choices before this forensic audit'
    original = ROOT/'motion_counterexample_regression'
    dropped = ROOT/'motion_drop_rest_regression'
    # Requires the read-only NumPy evaluation to have passed GPU parity checks.
    parity = read(dropped/'numpy_check/audit.json')
    assert parity['all_top1_equal'] and parity['max_logit_error'] < 2e-5
    counts = {}
    with (original/'per_pair_predictions.csv').open() as handle:
        for row in csv.DictReader(handle):
            counts[row['state_uid'], row['eta_uid']] = [int(row['parent_success']), int(row['motion_success'])]
    pairs = read(original/'pairs.json')
    s = np.array([counts[p['state_uid'], p['eta_uid']] for p in pairs]).T
    q = s/16; dq = q[1]-q[0]
    entries = [(original, e) for e in read(original/'protocol.json')['models'] if e['arm'] in ('full_update', 'trunk_only')]
    entries += [(dropped, e) for e in read(dropped/'protocol.json')['models']]
    rows = []
    for folder, entry in entries:
        if entry['fold'] == 2:
            continue  # Controller constituents were not trained in fold2.
        history = {row['step']: row for row in read(Path(entry['path'])/'history.json')}
        for step in STEPS:
            checkpoint = Path(entry['path'])/f'step{step}.msgpack'
            spec = {**entry, 'step': step, 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint)}
            z = regression_predictions(folder, spec); p = expit(z); dp = p[1]-p[0]
            rows.append(dict(arm=entry['arm'], fold=entry['fold'], seed=entry['seed'], step=step,
                originally_selected=step==entry['step'],
                source_held_controller_NLL=float(history[step]['held_native_VAL']['NLL']),
                source_TRAIN_NLL=float(history[step]['TRAIN_NLL']),
                regression_NLL=float((q*np.logaddexp(0,-z[:2])+(1-q)*np.logaddexp(0,z[:2])).mean()),
                regression_causal_MAE=float(abs(dp-dq).mean()),
                regression_causal_correlation=float(np.corrcoef(dp,dq)[0,1]),
                example_parent_prediction=float(p[0,1]), example_motion_prediction=float(p[1,1]),
                example_predicted_delta=float(dp[1]), example_true_delta=float(dq[1]),
                example_direction_correct=bool(dp[1]*dq[1]>0)))
    with (OUT/'trajectory.csv').open('w', newline='') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for arm in ('full_update','trunk_only','drop_rest'):
        for step in STEPS:
            rr=[r for r in rows if r['arm']==arm and r['step']==step]
            summary.append(dict(arm=arm,step=step,fits=len(rr),
                example_direction_correct=sum(r['example_direction_correct'] for r in rr),
                example_pred_delta_mean=float(np.mean([r['example_predicted_delta'] for r in rr])),
                source_held_NLL_mean=float(np.mean([r['source_held_controller_NLL'] for r in rr])),
                regression_causal_MAE_mean=float(np.mean([r['regression_causal_MAE'] for r in rr])),
                regression_NLL_mean=float(np.mean([r['regression_NLL'] for r in rr]))))
    audit=dict(new_rollouts=0,new_fits=0,source_selection_frozen_sha256=sha(selection),
        primary_checkpoint_selection_changed=False,posthoc_diagnostic=True,independent_confirmation=False,
        code_sha256=sha(__file__),cached_GPU_parity_sha256=sha(dropped/'numpy_check/audit.json'),
        caveat='Same8previouslyopenedfamilies and2sourceCVfolds;not6independentdatasets. No new model selected from this trajectory.',summary=summary)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
