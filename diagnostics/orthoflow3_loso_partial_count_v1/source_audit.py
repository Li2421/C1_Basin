"""Matched source-VAL failure audit; no training/model selection changes."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from scipy.special import expit
from scipy.stats import rankdata
from .data import OUT,FOLDS,load,baseline,csvout,dump
from .train import data
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import model_for,gather
jax.config.update('jax_default_matmul_precision','highest')

def roc_auc_score(labels,scores):
    labels=np.asarray(labels,dtype=bool); npos=int(labels.sum()); nneg=len(labels)-npos
    return (rankdata(scores)[labels].sum()-npos*(npos+1)/2)/(npos*nneg)

def audit():
    result=[]
    for fold in FOLDS:
        rows,x,e,s,f,binary,inv,groups=data(fold,'partial_count');si=np.array([r['state_index'] for r in rows]);ix=np.concatenate([g['validation'] for g in groups.values()]);model=model_for('shared');init=model.init(jax.random.PRNGKey(0),gather(x,[si[0]]),jnp.zeros((1,3)))
        for method,folder in (('old_full',baseline(fold)),('full_continuation',OUT/fold/'full_continuation'),('partial_count',OUT/fold/'partial_count'),('negative_binary',OUT/fold/'negative_binary')):
            ck=load(folder/'models_frozen.json')['shared']['selected'];p=serialization.from_bytes(init,open(ck['checkpoint'],'rb').read());fn=jax.jit(lambda bx,be:model.apply(p,bx,be));z=np.concatenate([np.asarray(fn(gather(x,si[jj]),jnp.asarray(e[jj]))) for jj in (ix[j:j+256] for j in range(0,len(ix),256))]);pr=expit(z)
            for scene in groups:
                at=np.array([k for k,i in enumerate(ix) if rows[i]['scenario']==scene]);r=[rows[ix[k]] for k in at];ll=(s[ix[at]]*np.logaddexp(0,-z[at])+f[ix[at]]*np.logaddexp(0,z[at]));probs=pr[at];neg=np.array([rr['non_b15_confirmed'] for rr in r]);pos=np.array([rr['b15_confirmed'] for rr in r]);label=pos[pos|neg].astype(int)
                result.append({'held_out_scene':fold,'source_scene':scene,'method':method,'VAL_pairs':len(at),'count_likelihood_per_observed_trial':float(ll.sum()/(s[ix[at]]+f[ix[at]]).sum()),'certified_nonB15':int(neg.sum()),'mean_p_on_nonB15':float(probs[neg].mean()) if neg.any() else None,'p95_on_nonB15':int(sum(probs[neg]>.95)),'mean_p_on_B15':float(probs[pos].mean()) if pos.any() else None,'B15_AUROC':float(roc_auc_score(label,probs[pos|neg])) if len(set(label))==2 else None,'partial_rate_not_treated_as_exact_Q16':True})
    csvout('source_failure_learning.csv',result);dump('source_failure_learning_complete.json',{'models_changed':False,'source_VAL_only':True})
    print('source audit complete',len(result))

if __name__=='__main__':audit()
