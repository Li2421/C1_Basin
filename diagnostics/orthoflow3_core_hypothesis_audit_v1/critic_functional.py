"""Frozen shared-critic state intervention on the original K16 candidate pools."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.special import expit
from scipy.stats import rankdata

from diagnostics.orthoflow3_loso_partial_count_v1.data import OUT as PRE, OLD, FOLDS, load, sha
from diagnostics.orthoflow3_loso_partial_count_v1.train import data
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import model_for, gather

OUT = Path(__file__).resolve().parent
jax.config.update('jax_default_matmul_precision', 'highest')


def csvout(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def auc(y, p):
    y = np.asarray(y, bool); n1 = int(y.sum()); n0 = len(y)-n1
    if not n1 or not n0: return None
    return float((rankdata(p)[y].sum()-n1*(n1+1)/2)/(n1*n0))


def model(fold, kind):
    ck = load(PRE/fold/'partial_count/models_frozen.json')[kind]['selected']
    assert sha(ck['checkpoint']) == ck['sha256']
    m = model_for(kind)
    init = m.init(jax.random.PRNGKey(0), gather(dict(np.load(PRE/'entities.npz')), [0]), jnp.zeros((1,3)))
    pars = serialization.from_bytes(init, Path(ck['checkpoint']).read_bytes())
    return m, pars


def logits(m, pars, x, e, state_index):
    fn = jax.jit(lambda bx, be: m.apply(pars, bx, be))
    return np.concatenate([np.asarray(fn(gather(x, state_index[j:j+256]), jnp.asarray(e[j:j+256])))
                           for j in range(0,len(e),256)])


def summary(p, truth, baseline_index=None):
    index=p.argmax(1); n=len(index); yes=[truth[i]['robust'][j] for i,j in enumerate(index)]
    qlo=np.array([truth[i]['lower'][j] for i,j in enumerate(index)]);qhi=np.array([truth[i]['upper'][j] for i,j in enumerate(index)])
    selected=p[np.arange(n),index]
    return {'B15_confirmed':sum(x is True for x in yes), 'B15_unknown':sum(x is None for x in yes),
            'Q16_lower_mean':float(qlo.mean()),'Q16_upper_mean':float(qhi.mean()),
            'severe_high_confidence':int(((selected>.95)&(qhi<=.5)).sum()),
            'top1_same_as_original':float(np.mean(index==baseline_index)) if baseline_index is not None else 1.,
            'mean_selected_prediction':float(selected.mean())}


def source_validation():
    out=[]
    for fold in FOLDS:
        rows,x,e,s,f,binary,inv,groups=data(fold,'partial_count')
        si=np.array([r['state_index'] for r in rows]);pred={}
        for kind in ('shared','eta_only'):
            m,pars=model(fold,kind)
            for source,g in groups.items():
                ix=g['validation'];z=logits(m,pars,x,e[ix],si[ix]);p=expit(z)
                y=np.array([rows[i]['b15_confirmed'] for i in ix],bool)
                neg=np.array([rows[i]['non_b15_confirmed'] for i in ix],bool)
                known=y|neg
                loss=np.logaddexp(0,-z)*s[ix]+np.logaddexp(0,z)*f[ix]
                out.append({'held_out_scene':fold,'source_scene':source,'model':kind,
                            'VAL_pairs':len(ix),'VAL_certified_B15':int(y.sum()),'VAL_certified_nonB15':int(neg.sum()),
                            'observed_trial_NLL':float(loss.sum()/(s[ix]+f[ix]).sum()),
                            'B15_AUROC':auc(y[known],p[known]),
                            'mean_p_nonB15':float(p[neg].mean()) if neg.any() else None,
                            'mean_p_B15':float(p[y].mean()) if y.any() else None})
        print('source',fold,flush=True)
    csvout('source_VAL_state_value.csv',out)


def frozen_target():
    scored=load(PRE/'target_predictions.json')['folds'];truth=load(PRE/'cached_truth.json');result=[];deltas=[]
    for fold in FOLDS:
        manifest=load(OLD/'targets'/fold/'manifest.json');x=dict(np.load(OLD/'targets'/fold/'entities.npz'))
        norm=load(PRE/fold/'normalization.json');e=np.array([r['eta'] for r in manifest],np.float32)
        n=len(manifest);flat=(e.reshape(-1,3)-np.array(norm['eta_center'],np.float32))/np.array(norm['eta_radius'],np.float32)
        m,pars=model(fold,'shared');orig=expit(logits(m,pars,x,flat,np.repeat(np.arange(n),16))).reshape(n,16)
        cached=np.asarray(scored[fold]['scores']['partial_count_shared'])
        assert np.max(abs(orig-cached))<2e-5,(fold,'preprocessing mismatch')
        original_choice=orig.argmax(1);base=summary(orig,truth[fold])
        result.append({'fold':fold,'intervention':'original','seed':None,**base})
        etaonly=np.asarray(scored[fold]['scores']['partial_count_eta_only']);eta_choice=etaonly.argmax(1)
        for i in range(n):
            j=int(original_choice[i]);k=int(eta_choice[i]);tt=truth[fold][i]
            deltas.append({'fold':fold,'state_uid':manifest[i]['state_uid'],'shared_choice':j,'eta_only_choice':k,
                           'same_choice':j==k,'shared_B15':tt['robust'][j],'eta_only_B15':tt['robust'][k],
                           'shared_Q_lower':tt['lower'][j],'shared_Q_upper':tt['upper'][j],
                           'eta_only_Q_lower':tt['lower'][k],'eta_only_Q_upper':tt['upper'][k],
                           'shared_selected_p':orig[i,j],'eta_only_selected_p':etaonly[i,k]})
        # All donor physical states are unlabeled; target outcomes enter only summary().
        for seed in load(OUT/'protocol.json')['state_shuffle_seeds']:
            rng=np.random.default_rng(seed);perm=rng.permutation(n)
            while np.any(perm==np.arange(n)):perm=rng.permutation(n)
            q=expit(logits(m,pars,x,flat,np.repeat(perm,16))).reshape(n,16)
            result.append({'fold':fold,'intervention':'state_shuffle','seed':seed,**summary(q,truth[fold],original_choice)})
        # Exact target-h marginal over every state, irrespective of actual outcome.
        total=np.zeros((n,16),np.float64)
        for donor in range(n):
            q=expit(logits(m,pars,x,flat,np.full(n*16,donor,dtype=int))).reshape(n,16)
            total+=q
        avg=total/n
        result.append({'fold':fold,'intervention':'target_h_marginal_score','seed':None,**summary(avg,truth[fold],original_choice)})
        print('target',fold,flush=True)
    csvout('target_state_interventions.csv',result)
    csvout('shared_vs_eta_paired.csv',deltas)
    (OUT/'checkpoint_replay.json').write_text(json.dumps({'pass':True,'max_allowed_probability_difference':2e-5,
        'target_outcomes_used_for_model_selection':False,'target_states_used_for_marginalization_without_outcome_labels':True},indent=2)+'\n')


if __name__=='__main__':
    assert load(OUT/'protocol.json')['new_rollout_at_freeze']==0
    source_validation()
    frozen_target()
