"""Frozen new-controller predictions, selection and matched input controls."""
import os,json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr,binomtest
from .fresh_controller_test import OUT,SOURCE,read,write
from . import support_train as tr
from .support_confirmation import OUT as CONF
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite

def main():
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    frozen=read(SOURCE/'models_frozen.json');d=np.load(OUT/'dataset.npz');x=dict(np.load(OUT/'entities.npz'))
    assert d['valid'].all(),'Do not silently remove target numerical response cases'
    s=d['success'].reshape(64,2);f=d['failure'].reshape(64,2);n=s+f;q=s/n;good=s>=15;bad=f>=2;eligible=good.any(1)
    wrong={c:np.load(CONF/f'inputs_{c}.npz') for c in ('alt','second')}
    shift=np.roll(np.arange(64),-1);pair_shift=np.column_stack([shift*2,shift*2+1]).ravel()
    rows=[];decisions=[];predictions={};conditions=('correct','wrong_alt','wrong_second','state_shuffle','joint_state_context_shuffle')
    def metrics(z):
        z=z.reshape(64,2);p=expit(z);choice=z.argmax(1);idx=np.arange(64);selected=q[idx,choice];g=good[idx,choice];b=bad[idx,choice]
        severe=(p[idx,choice]>.9)&((16-f[idx,choice])/16<=.5)
        delta=q[:,0]-q[:,1];pred_delta=p[:,0]-p[:,1];strong=abs(delta)>=.25
        return dict(NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/n.sum()),MAE=float(abs(p-q).mean()),
            B15=int(g.sum()),unknown=int((~g&~b).sum()),cases=64,oracle_B15=int(eligible.sum()),
            eligible_B15=int(g[eligible].sum()),eligible_cases=int(eligible.sum()),
            eligible_selection_rate=float(g[eligible].mean()) if eligible.any() else None,
            selected_Q=float(selected.mean()),regret=float((q.max(1)-selected).mean()),
            selected_Q_eligible=float(selected[eligible].mean()) if eligible.any() else None,
            predicted_selected=float(p[idx,choice].mean()),severe=int(severe.sum()),
            strong=int(strong.sum()),strong_correct=int((np.sign(delta[strong])==np.sign(pred_delta[strong])).sum()),
            state_contrast_correlation=float(pearsonr(delta,pred_delta).statistic) if np.std(pred_delta)>1e-9 else None),choice
    for kind in tr.KINDS:
      model=tr.model(kind);apply=jax.jit(lambda p,x,e,c:model.apply(p,x,e,c))
      for entry in [e for e in frozen['models'] if e['kind']==kind]:
        path=Path(entry['path']);assert sha(path/'best.msgpack')==entry['checkpoint_sha256'];norm=read(path/'normalization.json')
        params=serialization.msgpack_restore((path/'best.msgpack').read_bytes());eta=(d['eta']-norm['eta_center'])/norm['eta_scale']
        for condition in conditions:
            inputs=wrong[condition[6:]] if condition.startswith('wrong_') else d
            cc=(inputs['context']-norm['context_center'])/norm['context_scale']
            ar=(inputs['agent_response']-norm['agent_center'])/norm['agent_scale'];ctx=np.concatenate([cc,ar.reshape(128,-1)],-1)
            stateidx=shift[d['state_index']] if condition in ('state_shuffle','joint_state_context_shuffle') else d['state_index']
            if condition=='joint_state_context_shuffle':ctx=ctx[pair_shift]
            z=np.asarray(apply(params,gather(x,stateidx),jnp.asarray(eta,jnp.float32),jnp.asarray(ctx,jnp.float32))).reshape(64,2)
            met,choice=metrics(z);label=f"{entry['size']}__{kind}__{entry['seed']}__{condition}";predictions[label]=z
            rows.append(dict(size=entry['size'],kind=kind,seed=entry['seed'],condition=condition,**met))
            for state in range(64):
                i=int(choice[state]);decisions.append(dict(size=entry['size'],kind=kind,seed=entry['seed'],condition=condition,state_index=state,
                    eta_index=(10,15)[i],Q=float(q[state,i]),prediction=float(expit(z[state,i])),B15=bool(good[state,i]),non_B15=bool(bad[state,i]),eligible=bool(eligible[state])))
    np.savez_compressed(OUT/'predictions.npz',**predictions);csvwrite(OUT/'metrics.csv',rows);csvwrite(OUT/'decisions.csv',decisions)
    rng=np.random.default_rng(20261004081);boot=rng.integers(64,size=(10000,64));comparisons=[]
    def compare(a,b):
        ia=np.argmax(a,1);ib=np.argmax(b,1);ix=np.arange(64);ga=good[ix,ia];gb=good[ix,ib];ba=bad[ix,ia];bb=bad[ix,ib]
        la=(s*np.logaddexp(0,-a)+f*np.logaddexp(0,a)).sum(1)/n.sum(1);lb=(s*np.logaddexp(0,-b)+f*np.logaddexp(0,b)).sum(1)/n.sum(1)
        dl=la-lb;dq=q[ix,ia]-q[ix,ib];lci=np.quantile(dl[boot].mean(1),[.025,.975]);qci=np.quantile(dq[boot].mean(1),[.025,.975])
        rescue=int((ga&bb).sum());brk=int((ba&gb).sum())
        return dict(NLL_delta=float(dl.mean()),NLL_CI_low=float(lci[0]),NLL_CI_high=float(lci[1]),
            selected_Q_delta=float(dq.mean()),selected_Q_CI_low=float(qci[0]),selected_Q_CI_high=float(qci[1]),
            rescue=rescue,breaks=brk,net_rescue=rescue-brk,paired_exact_p=float(binomtest(rescue,rescue+brk,.5).pvalue) if rescue+brk else 1.,
            changed_top1=int((ia!=ib).sum()),mean_abs_probability_change=float(abs(expit(a)-expit(b)).mean()))
    for entry in [e for e in frozen['models'] if e['kind'] in tr.KINDS]:
        size,kind,seed=(entry[k] for k in ('size','kind','seed'));prefix=f'{size}__{kind}__{seed}';z=predictions[prefix+'__correct']
        for reference in ('eta_only','wrong_alt','wrong_second','state_shuffle','joint_state_context_shuffle'):
            r=predictions[f'{size}__eta_only__{seed}__correct'] if reference=='eta_only' else predictions[prefix+'__'+reference]
            comparisons.append(dict(size=size,kind=kind,seed=seed,reference=reference,**compare(z,r)))
    csvwrite(OUT/'paired_comparisons.csv',comparisons)
    protocol=read(OUT/'protocol.json')
    write(OUT/'evaluation_audit.json',dict(source_TRAIN_families=206,independent_target_families=64,target_controllers=1,
        fixed_eta_count=2,oracle_B15=int(eligible.sum()),both_candidates_B15=int(good.all(1).sum()),
        no_known_B15=int((~eligible).sum()),fixed_eta10_B15=int(good[:,0].sum()),fixed_eta15_B15=int(good[:,1].sum()),
        numerical_attempts=int(d['numerical'].sum()),target_labels_used_for_model_selection=False,
        target_controller_sha256=protocol['profiles'][0]['sha256'],models_frozen_sha256=sha(SOURCE/'models_frozen.json'),
        privileged_controller_ID_model_excluded=True,scope=protocol['scope'],generator_modified=False))
    print(json.dumps([r for r in rows if r['size']=='expanded206_pair_equal' and r['kind'] in ('eta_only','entity_response_mean')],indent=2))

if __name__=='__main__':main()
