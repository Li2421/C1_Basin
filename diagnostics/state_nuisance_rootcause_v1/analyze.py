"""Metrics for cached and newly paired nuisance-input interventions."""
from pathlib import Path
import json
import csv
import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'orthoflow3_tt_ff_matched_learnability_v1'
SEEDS = (17, 23, 41)


def write(path, x):
    path.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def csvout(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def labels(scene, chain, split, data, truth):
    ci = ('TT', 'FF').index(chain)
    ix = np.flatnonzero((data['scene'] == scene) & (data['split'] == split))
    if split == 'test':
        loc = np.flatnonzero(data['scene'][truth['indices']] == scene)
        assert np.array_equal(truth['indices'][loc], ix)
        o = truth['outcomes'][:, loc]
        valid = (o[0, ..., 1] == 0) & (o[1, ..., 1] == 0)
        s = ((o[ci, ..., 0] == 1) & (o[ci, ..., 1] == 0)).sum(-1)
        u = o[ci, ..., 1].sum(-1)
        f = 16 - u - s
        cs = (o[ci, ..., 0] * valid).sum(-1)
        cf = valid.sum(-1) - cs
    else:
        cs, cf = data['success'][ci, ix], data['failure'][ci, ix]
        s, f = data['own_success'][ci, ix], data['own_failure'][ci, ix]
        u = data['numerical'][ci, ix]
    return dict(s=s, f=f, u=u, cs=cs, cf=cf, ix=ix)


def measures(p, y, split):
    p = np.clip(np.asarray(p, float), 1e-7, 1-1e-7)
    cs, cf, s, f, u = [y[k] for k in ('cs', 'cf', 's', 'f', 'u')]
    cell_loss = -cs * np.log(p) - cf * np.log1p(-p)
    per_state = cell_loss.sum(1) / (cs + cf).sum(1)
    chosen = p.argmax(1)
    ii = np.arange(len(p))
    q = s / np.maximum(s+f, 1)
    out = dict(states=len(p), NLL=float(cell_loss.sum()/(cs+cf).sum()),
               trial_Brier=float((cs*(1-p)**2+cf*p**2).sum()/(cs+cf).sum()),
               selected_Q_observed=float(q[ii,chosen].mean()), distinct_selected_eta=len(set(chosen.tolist())),
               between_state_probability_sd_mean=float(p.std(0).mean()))
    details = dict(nll=per_state, chosen=chosen, q=q[ii, chosen])
    if split != 'train':
        hit = s[ii, chosen] >= 15
        bad = f[ii, chosen] >= 2
        unknown = ~(hit | bad)
        out.update(B15=int(hit.sum()), selection_unknown=int(unknown.sum()),
                   oracle_B15=int((s >= 15).any(1).sum()),
                   selected_Q16_lower=float((s[ii, chosen]/16).mean()),
                   selected_Q16_upper=float(((s[ii,chosen]+u[ii,chosen])/16).mean()))
        details.update(hit=hit, bad=bad, unknown=unknown)
    return out, details


def paired(a, b, seed=202610077):
    n = len(a[0]['nll'])
    # Average training-seed effects within family, never count seeds as new N.
    nd = np.mean([aa['nll']-bb['nll'] for aa,bb in zip(a,b)], axis=0)
    bd = np.mean([aa['hit'].astype(float)-bb['hit'] for aa,bb in zip(a,b)], axis=0)
    draws = np.random.default_rng(seed).integers(n, size=(10000,n))
    return dict(independent_families=n, delta_NLL_mean=float(nd.mean()),
                delta_NLL_family_bootstrap95=np.quantile(nd[draws].mean(1),[.025,.975]).tolist(),
                delta_B15_rate=float(bd.mean()),
                delta_B15_family_bootstrap95=np.quantile(bd[draws].mean(1),[.025,.975]).tolist(),
                rescue_by_seed=[int((aa['hit']&bb['bad']).sum()) for aa,bb in zip(a,b)],
                break_by_seed=[int((aa['bad']&bb['hit']).sum()) for aa,bb in zip(a,b)])


def cached():
    d = np.load(SOURCE/'dataset.npz')
    t = np.load(SOURCE/'test_truth.npz')
    z = np.load(HERE/'replayed_logits.npz')
    old = np.load(SOURCE/'frozen_predictions.npz')
    alphas = json.loads((HERE/'protocol.json').read_text())['cached_interventions']['alphas']
    metrics, contrasts, selections, provenance = [], [], {}, {}
    for scene in ('toy_give_way','ring_exchange'):
        for chain in ('TT','FF'):
            key = scene+':'+chain
            yl = {sp:labels(scene,chain,sp,d,t) for sp in ('train','validation','test')}
            scores = {}
            alpha_scores = []
            for alpha in alphas:
                vals = []
                for seed in SEEDS:
                    pe = expit(z[f'{scene}__{chain}__eta_only__{seed}__validation'])
                    pf = expit(z[f'{scene}__{chain}__full_context__{seed}__validation'])
                    vals.append(measures((1-alpha)*pe+alpha*pf,yl['validation'],'validation')[0]['NLL'])
                alpha_scores.append(float(np.mean(vals)))
            alpha = alphas[int(np.argmin(alpha_scores))]
            selections[key] = dict(alpha=alpha,mean_VAL_NLL_by_alpha=dict(zip(map(str,alphas),alpha_scores)))
            for seed in SEEDS:
                fullmean = expit(z[f'{scene}__{chain}__full_context__{seed}__train']).mean(0)
                for split in ('train','validation','test'):
                    pe = expit(z[f'{scene}__{chain}__eta_only__{seed}__{split}'])
                    pf = expit(z[f'{scene}__{chain}__full_context__{seed}__{split}'])
                    methods = dict(eta_only=pe, full=pf, full_state_marginal=np.broadcast_to(fullmean,pf.shape),
                                   source_selected_mixture=(1-alpha)*pe+alpha*pf)
                    if split != 'train':
                        for cond in ('state_shuffle','context_shuffle','state_context_shuffle'):
                            methods['full_'+cond] = expit(old[f'{scene}__{chain}__full_context__{seed}__{split}__{cond}'])
                    for name,p in methods.items():
                        m,details=measures(p,yl[split],split)
                        metrics.append(dict(scene=scene,chain=chain,split=split,seed=seed,method=name,**m))
                        scores[seed,split,name]=details
            for split in ('validation','test'):
                for a,b in [('full','eta_only'),('full_state_marginal','full'),('full_state_marginal','eta_only'),
                            ('source_selected_mixture','full'),('source_selected_mixture','eta_only'),
                            ('full_state_shuffle','full'),('full_context_shuffle','full'),('full_state_context_shuffle','full')]:
                    contrasts.append(dict(scene=scene,chain=chain,split=split,candidate=a,reference=b,
                        **paired([scores[s,split,a] for s in SEEDS],[scores[s,split,b] for s in SEEDS])))
            for split in ('validation','test'):
                y=yl[split]
                row={}
                for name in ('eta_only','full','full_state_marginal','source_selected_mixture'):
                    selected=[r for r in metrics if (r['scene'],r['chain'],r['split'],r['method'])==(scene,chain,split,name)]
                    row[name]=dict(NLL=[r['NLL'] for r in selected],B15=[r['B15'] for r in selected])
                provenance[key+':'+split]=row
    csvout(HERE/'cached_metrics.csv',metrics)
    write(HERE/'cached_contrasts.json',contrasts)
    write(HERE/'mixture_selection.json',selections)
    write(HERE/'cached_summary.json',provenance)
    print(json.dumps(dict(selection=selections,test={k:v for k,v in provenance.items() if k.endswith(':test')}),indent=2))


def interventions():
    d=np.load(SOURCE/'dataset.npz'); t=np.load(SOURCE/'test_truth.npz')
    z=np.load(HERE/'intervention_logits.npz');original=np.load(HERE/'replayed_logits.npz')
    rows=[];contrasts=[];details={};parity=[]
    yl={sp:labels('ring_exchange','FF',sp,d,t) for sp in ('train','validation','test')}
    for arm in ('full_replay','constant_h','constant_all','constant_context'):
        for seed in SEEDS:
            for checkpoint in ('selected','original_step','final'):
                for split in ('train','validation','test'):
                    zz=z[f'{arm}__{seed}__{checkpoint}__{split}']
                    pp=expit(zz)
                    if arm=='constant_all':assert np.max(np.ptp(zz,axis=0))<1e-5
                    m,dd=measures(pp,yl[split],split)
                    rows.append(dict(arm=arm,seed=seed,checkpoint=checkpoint,split=split,**m))
                    details[arm,seed,checkpoint,split]=dd
                    if arm=='full_replay' and checkpoint=='original_step':
                        diff=float(np.max(np.abs(zz-original[f'ring_exchange__FF__full_context__{seed}__{split}'])))
                        parity.append(dict(seed=seed,split=split,max_abs_logit_error=diff))
    for seed in SEEDS:
        for split in ('train','validation','test'):
            m,dd=measures(expit(original[f'ring_exchange__FF__eta_only__{seed}__{split}']),yl[split],split)
            rows.append(dict(arm='eta_only_original',seed=seed,checkpoint='selected',split=split,**m))
            for ck in ('selected','original_step','final'):details['eta_only_original',seed,ck,split]=dd
    for ck in ('selected','original_step','final'):
        for split in ('validation','test'):
            for a,b in [('constant_h','full_replay'),('constant_all','full_replay'),('constant_all','constant_h'),
                        ('full_replay','eta_only_original'),('constant_h','eta_only_original'),('constant_all','eta_only_original'),
                        ('constant_context','full_replay'),('constant_context','eta_only_original'),('constant_context','constant_all')]:
                contrasts.append(dict(checkpoint=ck,split=split,candidate=a,reference=b,
                    **paired([details[a,s,ck,split] for s in SEEDS],[details[b,s,ck,split] for s in SEEDS])))
    csvout(HERE/'intervention_metrics.csv',rows)
    write(HERE/'intervention_contrasts.json',contrasts)
    write(HERE/'training_replay_parity.json',parity)
    output=[]
    for arm in ('eta_only_original','full_replay','constant_h','constant_all','constant_context'):
        rr=[r for r in rows if r['arm']==arm and r['checkpoint']=='selected']
        output.append(dict(arm=arm,**{sp:{key:[r[key] for r in rr if r['split']==sp] for key in ['NLL']+([] if sp=='train' else ['B15'])} for sp in ('train','validation','test')}))
    write(HERE/'intervention_summary.json',output)
    print(json.dumps(dict(summary=output,parity=parity,contrasts=[r for r in contrasts if r['split']=='test' and r['checkpoint']=='selected']),indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=('cached','interventions'))
    globals()[p.parse_args().action]()
