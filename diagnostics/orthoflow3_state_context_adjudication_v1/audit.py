"""No rollout: source-only information value and fixed-eta predictability tests."""
from pathlib import Path
from itertools import combinations
import hashlib
import json
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

OUT = Path(__file__).resolve().parent
SRC = OUT.parent / 'orthoflow3_controller_training_repair_v1'
ROOT = OUT.parents[1]


def read(p): return json.loads(Path(p).read_text())
def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load():
    d = dict(np.load(SRC / 'dataset.npz'))
    x = dict(np.load(SRC / 'entities.npz'))
    rows = read(SRC / 'pairs.json')
    states = read(SRC / 'states.json')
    si = d['state_index']
    indices = np.stack([np.flatnonzero(si == j) for j in range(30)])
    assert indices.shape == (30, 16)
    assert len({tuple(rows[i]['eta_uid'] for i in ii) for ii in indices}) == 1
    tr = np.array([j for j, ii in enumerate(indices) if d['split'][ii[0]] == 'train'])
    va = np.array([j for j, ii in enumerate(indices) if d['split'][ii[0]] == 'validation'])
    assert len(tr) == 24 and len(va) == 6
    assert len({s['source_group'] for s in states}) == 30
    seed_success = np.full((3, 30, 16, 16), np.nan)
    profiles = read(SRC / 'protocol.json')['profiles']
    with sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro', uri=True) as db:
        for c, profile in enumerate(profiles):
            for j, ii in enumerate(indices):
                for e, i in enumerate(ii):
                    records = db.execute("SELECT seed_key,success,numerical_failure FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE'",
                        (rows[i]['state_uid'], rows[i]['eta_uid'], profile['controller_uid'])).fetchall()
                    for key, success, numerical in records:
                        k = json.loads(key).get('future_index', -1)
                        if 0 <= k < 16 and not numerical:
                            seed_success[c, j, e, k] = success
    s = np.nansum(seed_success, -1)
    n = np.isfinite(seed_success).sum(-1)
    np.testing.assert_array_equal(s, d['standard_success'][:, indices])
    np.testing.assert_array_equal(n-s, d['standard_failure'][:, indices])
    s4 = np.nansum(seed_success[..., :4], -1)
    n4 = np.isfinite(seed_success[..., :4]).sum(-1)
    return d, x, rows, states, indices, tr, va, seed_success, s, n, s4, n4


def canonical_h(x):
    """Lossless entity flattening modulo agent order; diagnostic kernel only."""
    result = []
    for j in range(len(x['agents'])):
        a = x['agents'][j]
        order = np.lexsort(tuple(a[:, k] for k in reversed(range(a.shape[1]))))
        result.append(np.concatenate((a[order].ravel(), x['pairs'][j][order][:, order].ravel(),
                                      x['obstacles'][j][order].ravel(), x['globals'][j])))
    return np.asarray(result, dtype=float)


def evaluate(p, s, n, va):
    p = np.clip(p, .001, .999)
    q = s / np.maximum(n, 1)
    valid = n > 0
    loss = -(s*np.log(p)+(n-s)*np.log1p(-p))
    loss_family = np.mean(loss.sum(-1)/np.maximum(n.sum(-1), 1), axis=0)
    selected = p.argmax(-1)
    selected_s = np.take_along_axis(s, selected[..., None], -1)[..., 0]
    selected_q = np.take_along_axis(q, selected[..., None], -1)[..., 0]
    eligible = (s >= 15).any(-1)
    lo, hi = s/16, (16-(n-s))/16
    total = right = 0
    for a, b in combinations(range(len(va)), 2):
        for c in range(3):
            for e, f in combinations(range(16), 2):
                sa = 1 if lo[c,a,e]-hi[c,a,f] >= .25 else -1 if hi[c,a,e]-lo[c,a,f] <= -.25 else 0
                sb = 1 if lo[c,b,e]-hi[c,b,f] >= .25 else -1 if hi[c,b,e]-lo[c,b,f] <= -.25 else 0
                if sa*sb < 0:
                    total += 1
                    right += int(np.sign(p[c,a,e]-p[c,a,f]) == sa and np.sign(p[c,b,e]-p[c,b,f]) == sb)
    return {'NLL': float(np.mean(loss.sum((1,2))/n.sum((1,2)))),
            'MAE': float(np.abs(q-p)[valid].mean()),
            'selected_B15': int(((selected_s >= 15) & eligible).sum()),
            'eligible': int(eligible.sum()), 'selected_Q': float(selected_q[eligible].mean()),
            'regret': float((q.max(-1)-selected_q)[eligible].mean()),
            'state_reversals_correct': right, 'state_reversals_total': total,
            'per_family_NLL': loss_family.tolist(), 'selected_eta_indices': selected.tolist()}


def fit_predict(blocks, s, n, train, test, alpha):
    """48 independent fixed-controller/eta heads, shared predeclared kernel."""
    result = np.zeros((3, len(test), 16))
    for c in range(3):
        for e in range(16):
            dtrain = np.zeros((len(train), len(train)))
            dtest = np.zeros((len(test), len(train)))
            for block in blocks:
                xx = block[c, :, e]
                mu = xx[train].mean(0)
                scale = xx[train].std(0)
                active = scale > 1e-6
                if not active.any(): continue
                z = (xx[:, active]-mu[active])/np.maximum(scale[active], .05)
                dt = cdist(z[train], z[train], 'sqeuclidean') / active.sum()
                dv = cdist(z[test], z[train], 'sqeuclidean') / active.sum()
                median = np.median(dt[np.triu_indices(len(train), 1)])
                dtrain += dt/max(median, 1e-8)/len(blocks)
                dtest += dv/max(median, 1e-8)/len(blocks)
            kernel, kv = np.exp(-dtrain/2), np.exp(-dtest/2)
            ss, nn = s[c, train, e], n[c, train, e]
            prior = (ss.sum()+.5)/(nn.sum()+1)
            q = ss/np.maximum(nn, 1)
            weight = nn/4
            coef = np.linalg.solve(kernel + np.diag(alpha/np.maximum(weight, 1e-5)), q-prior)
            result[c, :, e] = np.clip(prior+kv@coef, .001, .999)
    return result


def choose_alpha(blocks, s, n, tr, alphas):
    cv = []
    for alpha in alphas:
        losses = []
        for fold in range(6):
            val = tr[np.arange(len(tr)) % 6 == fold]
            fit = tr[np.arange(len(tr)) % 6 != fold]
            p = fit_predict(blocks, s, n, fit, val, alpha)
            ss, nn = s[:, val], n[:, val]
            losses.append(float((-(ss*np.log(p)+(nn-ss)*np.log1p(-p))).sum()/nn.sum()))
        cv.append(float(np.mean(losses)))
    return alphas[int(np.argmin(cv))], cv


def main():
    d, x, rows, states, indices, tr, va, seeds, s, n, s4, n4 = load()
    protocol = read(OUT / 'protocol.json')
    good = s[:, va] >= 15
    eligible = good.any(-1)
    # Descriptive oracle ceilings; VAL labels are not used to train or select a deployable model.
    ceiling = {'families': 6, 'controller_cases': 18,
        'oracle_B15': int(eligible.sum()),
        'best_one_global_eta_B15_posthoc': int(good.sum((0,1)).max()),
        'best_fixed_eta_per_controller_B15_posthoc': int(good.sum(1).max(1).sum()),
        'per_controller_oracle': eligible.sum(1).tolist(),
        'per_controller_best_fixed': good.sum(1).max(1).tolist(),
        'maximum_incremental_B15_from_state_beyond_best_controller_fixed': int(eligible.sum()-good.sum(1).max(1).sum()),
        'warning': 'Descriptive panel ceilings, not trained baselines and not population optimality claims.'}
    write('selection_headroom.json', ceiling)
    h = canonical_h(x)
    shuffled_x = {k: v.copy() for k,v in x.items()}
    order = [2,0,3,1]
    shuffled_x['agents'] = x['agents'][:,order]
    shuffled_x['pairs'] = x['pairs'][:,order][:,:,order]
    shuffled_x['obstacles'] = x['obstacles'][:,order]
    np.testing.assert_array_equal(h, canonical_h(shuffled_x))
    cx = d['context'][:, indices]
    hh = np.broadcast_to(h[None,:,None,:], (3,30,16,h.shape[1]))
    native = np.array([np.concatenate([np.asarray(t['physical'][k]).ravel() for k in ('positions','velocities','goals')]) for t in states])
    raw = np.broadcast_to(native[None,:,None,:], (3,30,16,native.shape[1]))
    features = {'unified_h':[hh], 'H20_context':[cx], 'unified_h_plus_H20':[hh,cx], 'native_state_diagnostic':[raw]}
    prior = (s4[:,tr].sum(1)+.5)/(n4[:,tr].sum(1)+1.)
    predictions = {'controller_eta_prior':np.broadcast_to(prior[:,None,:], (3,len(va),16))}
    summary = {'scope':protocol['scope'], 'new_rollouts':0, 'target_labels_opened':False,
               'priors':evaluate(predictions['controller_eta_prior'],s[:,va],n[:,va],va), 'models':{}}
    alphas = protocol['kernel_probes']['alpha']
    for name, blocks in features.items():
        alpha, cv = choose_alpha(blocks,s4,n4,tr,alphas)
        p = fit_predict(blocks,s4,n4,tr,va,alpha)
        predictions[name] = p
        result = {'alpha_from_TRAIN_CV':alpha, 'TRAIN_CV_NLL_by_alpha':cv,
                  'validation':evaluate(p,s[:,va],n[:,va],va)}
        # Audit generalization on all train families using out-of-fold predictions.
        oof = np.zeros((3,len(tr),16))
        for fold in range(6):
            hold = np.arange(len(tr))%6 == fold
            oof[:,hold] = fit_predict(blocks,s4,n4,tr[~hold],tr[hold],alpha)
        result['TRAIN_OOF_NLL'] = float((-(s4[:,tr]*np.log(oof)+(n4[:,tr]-s4[:,tr])*np.log1p(-oof))).sum()/n4[:,tr].sum())
        result['note'] = 'OOF alpha selected on these folds: descriptive, selection-optimistic. External source VAL is the held-out score.'
        null = []
        rng = np.random.default_rng(protocol['kernel_probes']['seed'])
        for b in range(protocol['kernel_probes']['permutations']):
            perm = rng.permutation(tr)
            ps,pn = s4.copy(),n4.copy()
            ps[:,tr],pn[:,tr] = s4[:,perm],n4[:,perm]
            pa,_ = choose_alpha(blocks,ps,pn,tr,alphas)
            pp = fit_predict(blocks,ps,pn,tr,va,pa)
            null.append(evaluate(pp,s[:,va],n[:,va],va)['NLL'])
            if (b+1)%25 == 0: print(name, 'permutations', b+1, flush=True)
        result['state_block_permutation_null'] = {'repetitions':len(null), 'NLL_median':float(np.median(null)),
            'NLL_quantiles':np.quantile(null,[.025,.25,.75,.975]).tolist(),
            'one_sided_empirical_p':float((1+np.sum(np.asarray(null)<=result['validation']['NLL']))/(len(null)+1)),
            'NLL_values':null}
        summary['models'][name] = result
        write('fixed_eta_predictability.json',summary)
        print(name,result['validation'],flush=True)
    np.savez_compressed(OUT/'predictions.npz', **predictions)
    # Read old checkpoints' saved predictions; no retraining or checkpoint selection.
    trajectories = []
    for seed in (17,23,41):
        path = SRC/f'models/physical_context/seed{seed}'
        history=read(path/'history.json');best=read(path/'summary.json')['best_step']
        zz=np.load(path/'validation_predictions.npz')['correct'][:,indices[va]]
        pp=expit(zz)
        m=evaluate(pp,s[:,va],n[:,va],va)
        trajectories.append({'seed':seed,'best_step':best,'best_VAL_NLL':min(r['NLL'] for r in history),
            'final_VAL_NLL':history[-1]['NLL'], 'last_minibatch_train_NLL':history[-1]['training_objective'],
            'initial_minibatch_train_NLL':history[0]['training_objective'],'selected_checkpoint_metrics':m,
            'minibatch_warning':'TRAIN losses are individual minibatches, not full TRAIN error.'})
    write('trajectory_audit.json',trajectories)
    # Seed split diagnostics use complete Q16 cells only, avoiding numerical imputation.
    split_results=[]
    for a,b in ((slice(0,8),slice(8,16)),(slice(8,16),slice(0,8))):
        qa=np.nanmean(seeds[:,va,:,a],-1);qb=np.nanmean(seeds[:,va,:,b],-1)
        full=np.isfinite(seeds[:,va]).all(-1)
        qa=np.where(full,qa,-np.inf)
        fixed=np.argmax(np.mean(np.where(full,qa,np.nan),axis=1),-1)
        # A column incomplete in any family is excluded from the fixed action comparison.
        fixed=np.array([np.argmax(np.where(full[c].all(0),np.mean(qa[c],0),-np.inf)) for c in range(3)])
        personal=qa.argmax(-1)
        fq=qb[np.arange(3)[:,None],np.arange(6)[None,:],fixed[:,None]]
        pq=np.take_along_axis(qb,personal[...,None],-1)[...,0]
        split_results.append({'first_half':str(a),'complete_cells':int(full.sum()),
            'state_personalized_halfA_oracle_halfB_Q':float(pq.mean()),
            'controller_fixed_halfA_oracle_halfB_Q':float(fq.mean()),
            'difference':float((pq-fq).mean())})
    write('split_seed_oracle_diagnostic.json',{'results':split_results,
        'warning':'Both policies see first-half VAL outcomes: descriptive reproducibility check, not learned h evidence or deployment result.'})
    write('dataset_fingerprints.json',{'source_dataset_sha256':hashlib.sha256((SRC/'dataset.npz').read_bytes()).hexdigest(),
        'source_entities_sha256':hashlib.sha256((SRC/'entities.npz').read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        'database_read_only':True,'new_rollouts':0,'TRAIN_state_count':len(tr),'VAL_state_count':len(va)})


if __name__ == '__main__': main()
