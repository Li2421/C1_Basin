"""One frozen confirmation of the controller-diversity intervention."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from pathlib import Path
from scipy.special import expit
from scipy.stats import pearsonr,binomtest
from .diversity_confirmation import OUT,SOURCE,read,write
from .support_train import model
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite

def main():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    frozen=read(SOURCE/'models_frozen.json');d=np.load(OUT/'dataset.npz');x=dict(np.load(OUT/'entities.npz'))
    assert d['valid'].all()
    s=d['success'].reshape(64,2);f=d['failure'].reshape(64,2);n=s+f;q=s/n;good=s>=15;bad=f>=2;eligible=good.any(1)
    wrong={k:np.load(OUT/f'inputs_{k}.npz') for k in ('alt','second','extra_a','extra_b')}
    assert all(v['valid'].all() for v in wrong.values())
    shift=np.roll(np.arange(64),-1);pairshift=np.column_stack([2*shift,2*shift+1]).ravel();ix=np.arange(64)
    conditions=('correct',*(f'wrong_{k}' for k in wrong),'state_shuffle','context_state_shuffle','joint_state_context_shuffle')
    rows=[];decisions=[];preds={}
    def calculate(z):
        p=expit(z);ch=z.argmax(1);sel=q[ix,ch];g=good[ix,ch];b=bad[ix,ch]
        delta=q[:,0]-q[:,1];dp=p[:,0]-p[:,1];strong=abs(delta)>=.25
        return dict(NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/n.sum()),
            MAE=float(abs(p-q).mean()),Brier=float(((p-q)**2).mean()),B15=int(g.sum()),unknown=int((~g&~b).sum()),
            cases=64,oracle_B15=int(eligible.sum()),eligible_selected=int(g[eligible].sum()),
            selected_Q=float(sel.mean()),regret=float((q.max(1)-sel).mean()),predicted_selected=float(p[ix,ch].mean()),
            severe=int(((p[ix,ch]>.9)&((16-f[ix,ch])/16<=.5)).sum()),
            strong_cases=int(strong.sum()),strong_correct=int((np.sign(delta[strong])==np.sign(dp[strong])).sum()),
            state_contrast_correlation=float(pearsonr(delta,dp).statistic) if dp.std()>1e-8 else None),ch
    for kind in ('eta_only','entity_response_mean','entity_response_attached'):
      m=model(kind);apply=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
      for entry in [r for r in frozen['models'] if r['kind']==kind]:
        path=Path(entry['path']);assert sha(path/'best.msgpack')==entry['checkpoint_sha256'];assert sha(path/'normalization.json')==entry['normalization_sha256']
        par=serialization.msgpack_restore((path/'best.msgpack').read_bytes());norm=read(path/'normalization.json')
        eta=(d['eta']-norm['eta_center'])/norm['eta_scale']
        for condition in conditions:
            inp=wrong[condition[6:]] if condition.startswith('wrong_') else d
            cc=(inp['context']-norm['context_center'])/norm['context_scale'];ar=(inp['agent_response']-norm['agent_center'])/norm['agent_scale']
            c=np.concatenate([cc,ar.reshape(128,-1)],-1)
            si=shift[d['state_index']] if condition in ('state_shuffle','joint_state_context_shuffle') else d['state_index']
            if condition in ('context_state_shuffle','joint_state_context_shuffle'):c=c[pairshift]
            z=np.asarray(apply(par,gather(x,si),jnp.asarray(eta,jnp.float32),jnp.asarray(c,jnp.float32))).reshape(64,2)
            key=f"{entry['size']}__{kind}__{entry['seed']}__{condition}";preds[key]=z
            met,ch=calculate(z);rows.append(dict(arm=entry['size'],kind=kind,seed=entry['seed'],condition=condition,**met))
            for state in range(64):
                i=ch[state];decisions.append(dict(arm=entry['size'],kind=kind,seed=entry['seed'],condition=condition,state_index=state,
                    eta_index=(10,15)[i],Q=float(q[state,i]),predicted_Q=float(expit(z[state,i])),B15=bool(good[state,i]),non_B15=bool(bad[state,i]),eligible=bool(eligible[state])))
    np.savez_compressed(OUT/'predictions.npz',**preds);csvwrite(OUT/'metrics.csv',rows);csvwrite(OUT/'decisions.csv',decisions)
    rng=np.random.default_rng(20261004127);bootstrap=rng.integers(64,size=(10000,64));comparisons=[]
    def compare(a,b):
        ca=a.argmax(1);cb=b.argmax(1);ga=good[ix,ca];gb=good[ix,cb];ba=bad[ix,ca];bb=bad[ix,cb]
        la=(s*np.logaddexp(0,-a)+f*np.logaddexp(0,a)).sum(1)/n.sum(1);lb=(s*np.logaddexp(0,-b)+f*np.logaddexp(0,b)).sum(1)/n.sum(1)
        dl=la-lb;dq=q[ix,ca]-q[ix,cb];lci=np.quantile(dl[bootstrap].mean(1),[.025,.975]);qci=np.quantile(dq[bootstrap].mean(1),[.025,.975])
        r=int((ga&bb).sum());br=int((ba&gb).sum())
        return dict(NLL_delta=float(dl.mean()),NLL_CI_low=float(lci[0]),NLL_CI_high=float(lci[1]),
            selected_Q_delta=float(dq.mean()),selected_Q_CI_low=float(qci[0]),selected_Q_CI_high=float(qci[1]),
            rescue=r,breaks=br,net_rescue=r-br,paired_exact_p=float(binomtest(r,r+br,.5).pvalue) if r+br else 1.,
            changed_top1=int((ca!=cb).sum()),mean_probability_change=float(abs(expit(a)-expit(b)).mean()))
    for entry in frozen['models']:
        arm,kind,seed=(entry[k] for k in ('size','kind','seed'));key=f'{arm}__{kind}__{seed}';a=preds[key+'__correct']
        references={c:preds[key+'__'+c] for c in conditions[1:]}
        references['eta_only']=preds[f'{arm}__eta_only__{seed}__correct']
        if arm=='four_controller':references['matched_two_controller']=preds[f'two_controller__{kind}__{seed}__correct']
        for ref,b in references.items():comparisons.append(dict(arm=arm,kind=kind,seed=seed,reference=ref,**compare(a,b)))
    csvwrite(OUT/'paired_comparisons.csv',comparisons)
    write(OUT/'evaluation_audit.json',dict(families=64,held_controller_seed=88127,source_controllers=4,
        both_candidates_B15=int(good.all(1).sum()),oracle_B15=int(eligible.sum()),no_known_B15=int((~eligible).sum()),
        fixed_eta10_B15=int(good[:,0].sum()),fixed_eta15_B15=int(good[:,1].sum()),numerical=int(d['numerical'].sum()),
        unknown_cells=int((~good&~bad).sum()),models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
        target_controller_labels_used_for_selection=False,scope=read(OUT/'protocol.json')['scope'],generator_modified=False))
    print([r for r in rows if r['condition']=='correct'],flush=True)

if __name__=='__main__':main()
