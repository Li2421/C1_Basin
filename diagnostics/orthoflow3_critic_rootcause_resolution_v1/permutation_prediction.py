"""Frozen source-crossfit predictions under the measured slot intervention.

The parent family's entire fold is withheld. Recompute real h and C; do not
claim a full-input alias merely because the unordered geometry is unchanged.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import csv,json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr
from .role_control import add_roles
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import OUT as SRC,read,write,csvwrite
OUT=Path(__file__).resolve().parent

def main():
    import jax,jax.numpy as jnp
    from flax import serialization
    import flax.linen as nn
    from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    class Response(nn.Module):
        kind:str
        @nn.compact
        def __call__(self,x,e,c):
            r=c[:,24:].reshape(-1,4,16)
            if self.kind=='response_mean':r=jnp.broadcast_to(r.mean(1,keepdims=True),r.shape)
            if self.kind=='response_zero':r=jnp.zeros_like(r)
            xx={**x,'agents':jnp.concatenate([x['agents'],r],-1)}
            return Critic(True,True,False,name='core')(xx,e,c[:,:24],jnp.zeros((len(e),3)))
    base=OUT/'permutation_outcomes';pairs=read(base/'pairs.json');oldstates=read(SRC/'source_states.json')
    parent={s['uid']:i for i,s in enumerate(oldstates)};folds=read(SRC/'protocol.json')['folds']
    x0=dict(np.load(base/'entities.npz'));derived=np.load(base/'derived_inputs.npz')
    olddata=np.load(SRC/'source_data.npz')
    truth={(r['controller'],r['parent_state_uid'],int(r['eta_index'])):r for r in csv.DictReader((OUT/'role_permutation_Q.csv').open())}
    variants=[(SRC,'full_raw'),(OUT/'role_control','wide_db_role_zero'),(OUT/'role_control','wide_db_role_aware')]
    variants += [(OUT/'response_control',k) for k in ('response_zero','response_mean','response_entity')]
    rows=[]
    for root,kind in variants:
      xx=add_roles(x0,kind.endswith('role_aware')) if 'role_' in kind else x0
      m=Response(kind) if kind.startswith('response_') else Critic(True,True,False)
      if kind.startswith('response_'):forward=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
      else:forward=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c,jnp.zeros((len(e),3))))
      for seed in (17,23,41):
       for fold in folds:
        ids=[i for i,p in enumerate(pairs) if parent[p['parent_state_uid']] in fold['test']]
        if not ids:continue
        dest=root/'models'/kind/f"fold{fold['fold']}_seed{seed}";norm=read(dest/'normalization.json')
        params=serialization.msgpack_restore((dest/'best.msgpack').read_bytes());prev=np.load(dest/'predictions.npz')
        prevmap={int(ii):j for j,ii in enumerate(prev['indices'])}
        eta=(np.array([pairs[i]['eta'] for i in ids])-norm['eta_center'])/norm['eta_scale']
        for ci,controller in enumerate(('alt','second')):
            cc=(derived['context'][ci,ids]-norm['context_center'])/norm['context_scale']
            if kind.startswith('response_'):
                response=(derived['agent_response'][ci,ids]-norm['agent_response_center'])/norm['agent_response_scale']
                cc=np.concatenate([cc,response.reshape(len(ids),-1)],-1)
            z=np.asarray(forward(params,gather(xx,np.array([pairs[i]['state_index'] for i in ids])),jnp.asarray(eta,dtype=jnp.float32),jnp.asarray(cc,dtype=jnp.float32)))
            for j,i in enumerate(ids):
                p=pairs[i];oldi=parent[p['parent_state_uid']]*16+p['eta_index']
                oz=float(prev['best_correct'][ci,prevmap[oldi]]);r=truth[(controller,p['parent_state_uid'],p['eta_index'])]
                delta=(derived['context'][ci,i]-olddata['context'][ci,oldi])/np.array(norm['context_scale'])
                rows.append(dict(kind=kind,seed=seed,controller=controller,parent_state_uid=p['parent_state_uid'],eta_index=p['eta_index'],
                    original_Q=float(r['original_Q']),permuted_Q=float(r['permuted_Q']),true_delta=float(r['Q_difference']),
                    original_prediction=float(expit(oz)),permuted_prediction=float(expit(z[j])),pred_delta=float(expit(z[j])-expit(oz)),
                    global_context_normalized_L2=float(np.linalg.norm(delta)),significant_Holm=float(r['Holm_p'])<.05))
    csvwrite(OUT/'role_permutation_prediction.csv',rows)
    summary=[]
    for root,kind in variants:
      for seed in (17,23,41):
        rr=[r for r in rows if r['kind']==kind and r['seed']==seed];a=np.array([r['true_delta'] for r in rr]);b=np.array([r['pred_delta'] for r in rr])
        mask=np.array([r['significant_Holm'] for r in rr]);summary.append(dict(kind=kind,seed=seed,
            delta_correlation=float(pearsonr(a,b).statistic),delta_MAE=float(abs(a-b).mean()),
            significant_cells=int(mask.sum()),correct_sign_significant=int((np.sign(a[mask])==np.sign(b[mask])).sum()),
            mean_predicted_abs_delta=float(abs(b).mean()),mean_true_abs_delta=float(abs(a).mean()),
            context_distance_median=float(np.median([r['global_context_normalized_L2'] for r in rr]))))
    csvwrite(OUT/'role_permutation_prediction_summary.csv',summary)
    write(OUT/'role_permutation_prediction_audit.json',dict(new_rollouts=0,models_retrained=False,parent_family_held_out=True,
        caveat='Posthoc source intervention diagnosis. Current Flow reference and future response are genuinely recomputed, not held equal. No exact full-input alias claimed.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
