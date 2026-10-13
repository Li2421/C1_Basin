"""Analyze existing predictions only; all cohorts here are DEVELOPMENT."""
import itertools
import numpy as np
from .study import HERE, read, write, sha
from diagnostics.controller_ff_factorial_repair_v1.review import (
    selection_difference, mse_gain, intervals, derangements)


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    # Float32 inference may leave roundoff in mathematically constant eta-only.
    if min(a.std(), b.std()) < 1e-7:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def center(x, interaction=False):
    x = x-x.mean(0, keepdims=True)
    return x-x.mean(1, keepdims=True) if interaction else x


def reversal(q, raw, boot):
    # Fixed discovery 0:8/evaluation 8:16, symmetric gap >= 3/8.
    assert raw.shape[2] == 16
    lo = (raw[..., 0] == 1) & (raw[..., 1] == 0)
    hi = lo | (raw[..., 1] == 1)
    dl, dh = lo[:, :, :8].mean(-1), hi[:, :, :8].mean(-1)
    el, eh = lo[:, :, 8:].mean(-1), hi[:, :, 8:].mean(-1)
    num, den, en, ed = [np.zeros(len(q)) for _ in range(4)]
    pairs = []
    for a, b in itertools.combinations(range(16), 2):
        pos, neg = dl[:, a]-dh[:, b] >= .375, dl[:, b]-dh[:, a] >= .375
        if min(pos.sum(), neg.sum()) < 2:
            continue
        keep = pos | neg
        d = q[:, a]-q[:, b]
        score = np.where(abs(d) < 1e-7, .5, np.where(pos, d > 0, d < 0))
        num[keep] += score[keep]; den[keep] += 1
        ep, em = el[:, a] > eh[:, b], el[:, b] > eh[:, a]
        valid = keep & (ep | em)
        es = np.where(abs(d) < 1e-7, .5, np.where(ep, d > 0, d < 0))
        en[valid] += es[valid]; ed[valid] += 1
        pairs.append(dict(eta_pair=[a, b], positive=int(pos.sum()), negative=int(neg.sum()),
                          evaluation_positive_lower=float((el[pos, a]-eh[pos, b]).mean()),
                          evaluation_negative_lower=float((el[neg, b]-eh[neg, a]).mean())))
    def ratio(a, b):
        if not b.sum(): return dict(value=None, CI95=None, cases=0, states=0)
        bb=b[boot].sum(1); good=bb>0; v=a[boot].sum(1)[good]/bb[good]
        return dict(value=float(a.sum()/b.sum()), CI95=np.quantile(v,[.025,.975]).tolist(),
                    cases=int(b.sum()), states=int((b>0).sum()))
    return dict(pairs=pairs, discovery_accuracy=ratio(num,den), evaluation_accuracy=ratio(en,ed))


def run():
    d=dict(np.load(HERE/'dataset.npz')); pred={}; rows=[]; arrays={}; saved={}
    for n in (32,128,512):
        for arm in ('eta_only','physical23','current_entity'):
            paths=[HERE/'fits'/arm/f'n{n}'/f'seed{s}' for s in (17,23,41)]
            pred[n,arm]=np.mean([np.load(p/'predictions.npz')['probability'] for p in paths],0)
    for n in (32,128,512):
        for panel in ('train','validation','test_original','test_fresh','test_all'):
            mask=d['split']==panel if not panel.startswith('test') else d['split']=='test'
            if panel in ('test_original','test_fresh'): mask &= d['origin']==panel[5:]
            ix=np.flatnonzero(mask)
            if panel=='train': ix=ix[:n]
            count=d['success'][ix]+d['failure'][ix]+d['unknown'][ix]
            lo=d['success'][ix]/count; hi=(d['success'][ix]+d['unknown'][ix])/count; mid=(lo+hi)/2
            boot=np.random.default_rng(2026101001+len(ix)).integers(len(ix),size=(20000,len(ix)))
            permutations=derangements(len(ix)); saved[f'{panel}_n{n}__permutations']=permutations
            saved[f'{panel}_n{n}__state_uid']=d['state_uid'][ix]
            eta=pred[n,'eta_only'][ix].argmax(-1)
            for arm in ('eta_only','physical23','current_entity'):
                key=f'{panel}_n{n}_{arm}'; q=pred[n,arm][ix]; chosen=q.argmax(-1)
                saved[key+'__matched_prediction']=q
                saved[key+'__mismatch_prediction']=q[permutations]
                value=lambda x:x[np.arange(len(ix)),chosen]
                freq=np.bincount(chosen,minlength=16)/len(ix); nonzero=freq>0
                row=dict(n_train=n,panel=panel,arm=arm,N=len(ix),unknown=int(d['unknown'][ix].sum()),
                    Q_MSE_midpoint=float(np.mean((q-mid)**2)),
                    state_effect_corr=corr(center(q),center(mid)),
                    state_eta_corr=corr(center(q,True),center(mid,True)),
                    state_prediction_std=float(center(q).std()),
                    selected_success_bounds=[float(value(lo).mean()),float(value(hi).mean())],
                    selected_B15_bounds=[int((value(d['success'][ix])>=15).sum()),
                                        int((value(d['success'][ix]+d['unknown'][ix])>=15).sum())] if panel!='train' else None,
                    posthoc_best_fixed_bounds=[float(lo.mean(0).max()),float(hi.mean(0).max())],
                    oracle_bounds=[float(lo.max(1).mean()),float(hi.max(1).mean())],
                    selection_frequency=freq.tolist(),entropy_nats=float(-sum(freq[nonzero]*np.log(freq[nonzero]))))
                ml,mu,ql,qu=[np.zeros(len(ix)) for _ in range(4)]
                for offset in range(1,len(ix)):
                    donor=np.roll(np.arange(len(ix)),offset)
                    a,b=selection_difference(lo,hi,chosen,chosen[donor]);ml+=a/(len(ix)-1);mu+=b/(len(ix)-1)
                    a,b=mse_gain(q[donor],q,lo,hi);ql+=a.mean(1)/(len(ix)-1);qu+=b.mean(1)/(len(ix)-1)
                row['matched_minus_mismatch']=intervals(ml,mu,boot,key+'__mismatch',arrays)
                row['wrong_state_MSE_minus_matched']=intervals(ql,qu,boot,key+'__Q_mismatch',arrays)
                a,b=selection_difference(lo,hi,chosen,eta)
                row['minus_eta']=intervals(a,b,boot,key+'__eta',arrays)
                all_a,all_b=[],[]
                for j in range(16):
                    a,b=selection_difference(lo,hi,chosen,np.full(len(ix),j));all_a.append(a);all_b.append(b)
                all_a,all_b=np.stack(all_a,1),np.stack(all_b,1)
                aa,bb=[],[]
                for start in range(0,len(boot),500):
                    ii=boot[start:start+500];aa.append(all_a[ii].mean(1).min(-1));bb.append(all_b[ii].mean(1).min(-1))
                aa,bb=np.concatenate(aa),np.concatenate(bb)
                arrays[key+'__best_fixed_lower'],arrays[key+'__best_fixed_upper']=aa,bb
                row['minus_bootstrap_reselected_fixed']=dict(
                    identification_bounds=[float(all_a.mean(0).min()),float(all_b.mean(0).min())],
                    censor_CI95=[float(np.quantile(aa,.025)),float(np.quantile(bb,.975))])
                saved[key+'__all_fixed_difference_lower']=all_a
                saved[key+'__all_fixed_difference_upper']=all_b
                if panel.startswith('test'):
                    row['reversal']=reversal(q,d['raw_outcomes'][ix],boot)
                rows.append(row)
    write(HERE/'development_review.json',dict(rows=rows,new_rollouts=0,
        dataset_sha256=sha(HERE/'dataset.npz'),
        note='All opened DEVELOPMENT data. Censor intervals are conservative; midpoint correlations are sensitivity diagnostics. TRAIN fits are not generalization. VAL selected checkpoints. Bootstrap physical state; ensemble fixed three seeds, no best seed chosen on test.',
        mismatch='Exact expectation over all other physical states within the same panel; 200 additional saved random derangements.'))
    np.savez_compressed(HERE/'development_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/'development_predictions.npz',**saved)
    for r in rows:
        if r['n_train']==512 and r['panel']=='test_fresh':
            print({k:r[k] for k in ('arm','N','state_effect_corr','selected_success_bounds','matched_minus_mismatch','minus_eta','minus_bootstrap_reselected_fixed')})


if __name__=='__main__':run()
