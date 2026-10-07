"""Pooled, pre-frozen goal-response confirmations; no target/model selection."""
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from .goal_response import OUT as SOURCE
from .function_support import read, write
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import csvwrite, sha


def main():
    targets = [SOURCE.parent / name for name in
               ('goal_response_confirmation', 'goal_response_confirmation_b')]
    audits = [read(t / 'evaluation_audit.json') for t in targets]
    frozen = read(SOURCE / 'models_frozen.json')
    assert all(a['models_frozen_sha256'] == sha(SOURCE / 'models_frozen.json') for a in audits)
    data = [np.load(t / 'dataset.npz') for t in targets]
    pred = [np.load(t / 'predictions.npz') for t in targets]
    s = np.concatenate([d['success'].reshape(64, 2) for d in data])
    f = np.concatenate([d['failure'].reshape(64, 2) for d in data])
    n = s + f; q = s / n; good = s >= 15; bad = f >= 2; ix = np.arange(128)
    rng = np.random.default_rng(202610048131)
    boot = np.concatenate([rng.integers(64, size=(10000, 64)) + 64*j for j in range(2)], axis=1)
    rows, comparisons = [], []
    for entry in frozen['models']:
        variant, seed = entry['variant'], entry['seed']; key = f'{variant}__{seed}'
        z = np.concatenate([p[key + '__correct'] for p in pred]); p = expit(z); ch = z.argmax(1)
        ll = (s*np.logaddexp(0, -z) + f*np.logaddexp(0, z)).sum(1)/n.sum(1)
        dc = (q[:, 0]-q[:, 1]).reshape(2, 64); pc = (p[:, 0]-p[:, 1]).reshape(2, 64)
        dc -= dc.mean(1, keepdims=True); pc -= pc.mean(1, keepdims=True)
        rows.append(dict(variant=variant, seed=seed, cases=128, NLL=float(ll.mean()),
            MAE=float(abs(p-q).mean()), B15=int(good[ix, ch].sum()), oracle_B15=int(good.any(1).sum()),
            unknown=int((~good[ix, ch] & ~bad[ix, ch]).sum()), selected_Q_observed=float(q[ix, ch].mean()),
            per_controller_B15=[int(good[j:j+64][np.arange(64), ch[j:j+64]].sum()) for j in (0, 64)],
            centered_state_eta_contrast_correlation=float(np.corrcoef(dc.ravel(), pc.ravel())[0, 1]) if pc.std()>1e-8 else None,
            severe_false_positive=int(((p[ix,ch]>.9)&((16-f[ix,ch])/16<=.5)).sum())))
        refs = {k.split('__',2)[2]: k for k in pred[0].files if k.startswith(key+'__') and not k.endswith('__correct')}
        refs.update({b:f'{b}__{seed}__correct' for b in ('eta_only','H20_only','H20_only_matched_early')})
        for ref, rk in refs.items():
            zz = np.concatenate([pr[rk] for pr in pred]); cb = zz.argmax(1)
            lb = (s*np.logaddexp(0,-zz)+f*np.logaddexp(0,zz)).sum(1)/n.sum(1)
            dl, dq = ll-lb, q[ix,ch]-q[ix,cb]
            ga,gb,ba,bb=good[ix,ch],good[ix,cb],bad[ix,ch],bad[ix,cb]
            r,b=int((ga&bb).sum()),int((ba&gb).sum())
            comparisons.append(dict(variant=variant,seed=seed,reference=ref,NLL_delta=float(dl.mean()),
                NLL_CI=np.quantile(dl[boot].mean(1),[.025,.975]).tolist(),
                selected_Q_delta=float(dq.mean()),selected_Q_CI=np.quantile(dq[boot].mean(1),[.025,.975]).tolist(),
                rescue=r,breaks=b,net_rescue=r-b,paired_exact_p=float(binomtest(r,r+b,.5).pvalue) if r+b else 1.,
                unknown_comparisons=int(((~ga&~ba)|(~gb&~bb)).sum()),top1_changes=int((ch!=cb).sum()),
                mean_probability_change=float(abs(p-expit(zz)).mean())))
    csvwrite(SOURCE/'independent_confirmation_pooled_metrics.csv',rows)
    csvwrite(SOURCE/'independent_confirmation_pooled_comparisons.csv',comparisons)
    write(SOURCE/'independent_confirmation_summary.json',dict(target_audits=audits,pooled_metrics=rows,
        comparisons=[r for r in comparisons if r['variant']=='H20_goal'],
        uncertainty_scope='Family bootstrap conditional on two fixed unseen controllers, not a population interval over controllers. Training seeds not independent data replicates.',
        label_semantics='Observed Q excludes numerical exceptions; B15 requires >=15 successes, non-B15 >=2 failures. Numerical cases never imputed.',
        scope='Known Ring geometry; unseen controller parameters and source families; two seen exact eta. Not cross-scene LOSO, unseen eta or K16 deployment.',
        target_labels_used_for_selection=False,generator_modified=False,models_frozen_sha256=sha(SOURCE/'models_frozen.json')))
    for r in rows: print(r)


if __name__=='__main__': main()
