"""Frozen goal-response confirmation; all models/targets/seeds retained."""
import argparse,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import binomtest
from .goal_confirmation import configure,SOURCE,parent,read,write,sha
from .goal_response_cv import model,csvwrite


def main(replica):
    import jax,jax.numpy as jnp
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    out=configure(replica);frozen=read(SOURCE/'models_frozen.json');d=np.load(out/'dataset.npz');x=dict(np.load(out/'entities.npz'))
    assert d['valid'].all();g=np.load(out/'goal_held.npz');assert g['valid'].all()
    names=('alt','second','extra_a','extra_b');initial={k:np.load(out/f'inputs_{k}.npz') for k in names};goals={k:np.load(out/f'goal_{k}.npz') for k in names}
    assert all(v['valid'].all() for v in [*initial.values(),*goals.values()])
    s=d['success'].reshape(64,2);f=d['failure'].reshape(64,2);n=s+f;q=s/n;good=s>=15;bad=f>=2;eligible=good.any(1);ix=np.arange(64)
    shift=np.roll(ix,-1);pairshift=np.column_stack([2*shift,2*shift+1]).ravel()
    conditions=('correct',*(f'goal_wrong_{k}' for k in names),*(f'whole_wrong_{k}' for k in names),
        'goal_wrong_state','initial_wrong_state','all_context_wrong_state','state_shuffle','joint_state_context_shuffle')
    rows=[];decisions=[];preds={}
    for kind in ('eta_only','H20_only','H20_goal'):
      m=model(kind);apply=jax.jit(lambda p,xx,e,c:m.apply(p,xx,e,c))
      for entry in [r for r in frozen['models'] if r['kind']==kind]:
        path=Path(entry['path']);assert sha(path/entry['checkpoint'])==entry['checkpoint_sha256'];assert sha(path/'normalization.json')==entry['normalization_sha256']
        par=serialization.msgpack_restore((path/entry['checkpoint']).read_bytes());norm=read(path/'normalization.json');eta=(d['eta']-norm['eta_center'])/norm['eta_scale']
        for condition in conditions:
            name=next((k for k in names if condition in (f'goal_wrong_{k}',f'whole_wrong_{k}')),None)
            inp=initial[name] if condition.startswith('whole_wrong_') else d
            gg=goals[name]['goal_response'] if name else g['goal_response']
            cc=(inp['context']-norm['context_center'])/norm['context_scale'];ar=(inp['agent_response']-norm['agent_center'])/norm['agent_scale'];gc=(gg-norm['goal_center'])/norm['goal_scale']
            c=np.concatenate([cc,ar.reshape(128,-1),gc],-1)
            if condition=='goal_wrong_state':c[:,88:]=c[pairshift,88:]
            if condition=='initial_wrong_state':c[:,:88]=c[pairshift,:88]
            if condition in ('all_context_wrong_state','joint_state_context_shuffle'):c=c[pairshift]
            si=shift[d['state_index']] if condition in ('state_shuffle','joint_state_context_shuffle') else d['state_index']
            z=np.asarray(apply(par,gather(x,si),jnp.asarray(eta,jnp.float32),jnp.asarray(c,jnp.float32))).reshape(64,2);p=expit(z);ch=z.argmax(1)
            key=f"{entry['variant']}__{entry['seed']}__{condition}";preds[key]=z
            dc=q[:,0]-q[:,1];pc=p[:,0]-p[:,1];lo=s/16;hi=(16-f)/16
            strong=np.where(lo[:,0]-hi[:,1]>=.25,1,np.where(hi[:,0]-lo[:,1]<=-.25,-1,0))
            rows.append(dict(variant=entry['variant'],seed=entry['seed'],condition=condition,steps=entry['steps'],
                NLL=float((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum()/n.sum()),MAE=float(abs(p-q).mean()),
                B15=int(good[ix,ch].sum()),unknown=int((~good[ix,ch]&~bad[ix,ch]).sum()),oracle_B15=int(eligible.sum()),cases=64,
                selected_Q_observed=float(q[ix,ch].mean()),regret=float((q.max(1)-q[ix,ch]).mean()),selected_predicted=float(p[ix,ch].mean()),
                severe=int(((p[ix,ch]>.9)&(hi[ix,ch]<=.5)).sum()),strong_cases=int((strong!=0).sum()),strong_correct=int(((np.sign(pc)==strong)&(strong!=0)).sum()),
                state_contrast_correlation=float(np.corrcoef(dc,pc)[0,1]) if pc.std()>1e-8 else None))
            for state in ix:decisions.append(dict(variant=entry['variant'],seed=entry['seed'],condition=condition,state_index=int(state),eta_index=(10,15)[ch[state]],Q_observed=float(q[state,ch[state]]),predicted=float(p[state,ch[state]]),B15=bool(good[state,ch[state]]),non_B15=bool(bad[state,ch[state]]),eligible=bool(eligible[state])))
    csvwrite(out/'metrics.csv',rows);csvwrite(out/'decisions.csv',decisions);np.savez_compressed(out/'predictions.npz',**preds)
    rng=np.random.default_rng(202610048131);boot=rng.integers(64,size=(10000,64));comparisons=[]
    for entry in frozen['models']:
        variant,seed=entry['variant'],entry['seed'];key=f'{variant}__{seed}';a=preds[key+'__correct'];ca=a.argmax(1)
        refs={c:preds[key+'__'+c] for c in conditions[1:]}
        for baseline in ('eta_only','H20_only','H20_only_matched_early'):refs[baseline]=preds[f'{baseline}__{seed}__correct']
        for ref,b in refs.items():
            cb=b.argmax(1);ga,gb=good[ix,ca],good[ix,cb];ba,bb=bad[ix,ca],bad[ix,cb]
            la=(s*np.logaddexp(0,-a)+f*np.logaddexp(0,a)).sum(1)/n.sum(1);lb=(s*np.logaddexp(0,-b)+f*np.logaddexp(0,b)).sum(1)/n.sum(1)
            dl=la-lb;dq=q[ix,ca]-q[ix,cb];lci=np.quantile(dl[boot].mean(1),[.025,.975]);qci=np.quantile(dq[boot].mean(1),[.025,.975]);r=int((ga&bb).sum());br=int((ba&gb).sum())
            comparisons.append(dict(variant=variant,seed=seed,reference=ref,NLL_delta=float(dl.mean()),NLL_CI_low=float(lci[0]),NLL_CI_high=float(lci[1]),selected_Q_delta=float(dq.mean()),selected_Q_CI_low=float(qci[0]),selected_Q_CI_high=float(qci[1]),
                rescue=r,breaks=br,net_rescue=r-br,paired_exact_p=float(binomtest(r,r+br,.5).pvalue) if r+br else 1.,unknown_comparisons=int(((~ga&~ba)|(~gb&~bb)).sum()),changed_top1=int((ca!=cb).sum()),mean_probability_change=float(abs(expit(a)-expit(b)).mean())))
    csvwrite(out/'paired_comparisons.csv',comparisons)
    write(out/'evaluation_audit.json',dict(families=64,held_controller_seed=read(out/'protocol.json')['target_controller_seed'],source_controllers=12,oracle_B15=int(eligible.sum()),both_candidates_B15=int(good.all(1).sum()),no_known_B15=int((~eligible).sum()),fixed_eta10_B15=int(good[:,0].sum()),fixed_eta15_B15=int(good[:,1].sum()),numerical=int(d['numerical'].sum()),unknown_cells=int((~good&~bad).sum()),models_frozen_sha256=sha(SOURCE/'models_frozen.json'),target_labels_used_for_selection=False,generator_modified=False,scope=read(out/'protocol.json')['scope'],probability_note='Q_observed=s/(s+f); numericalexceptionsnotimputed; B15>=15observedsuccess/nonB15>=2failures.'))
    print([r for r in rows if r['condition']=='correct'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replicate',type=int,default=0);a=p.parse_args();main(a.replicate)
