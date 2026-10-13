"""One controlled intervention: preserve API agent slot in an otherwise identical net.

This is a diagnostic of unjustified invariance, not an approved deployment input.
Old current_entity512 predictions are the control; do not retrain that control.
"""
import argparse
import time
from pathlib import Path
import numpy as np
from diagnostics.controller_entity_coverage_check_v1 import study as base

HERE=Path(__file__).resolve().parent
SEEDS=(17,23,41)


def prepare():
    base.freeze(HERE/'protocol.json',dict(
        question='Does enforced agent-exchangeability cause the current encoder failure on the proven Ring512 actualFF cache?',
        evidence_before_new_fit=[
            'controller_entity_coverage_check_v1/development_review.json',
            'orthoflow3_unified_representation_v1/active_metamorphic.json',
            'transferable_state_conditioning_v1/encoder_invariance_audit.json',
            'orthoflow3_controller_training_repair_v1/state_normalized_gate.json'],
        reuse_audit='Old passive invariance and native active policy tests reused. Current512 pooled control reused. No prior same-network slot-preservation intervention found in controller/state diagnosis scripts. Scaling-only repair not repeated.',
        intervention='Replace last four of83 zero controller channels with one-hot native API agent slot. All other inputs/architecture/parameter shapes/init/draws/optimizer/checkpoint rule unchanged. Diagnostic only; no scene ID, no controller query.',
        control='controller_entity_coverage_check_v1/fits/current_entity/n512',
        dataset=str(base.HERE/'dataset.npz'),dataset_sha256=base.sha(base.HERE/'dataset.npz'),
        base_protocol_sha256=base.sha(base.HERE/'protocol.json'),
        base_code_sha256=base.sha(base.__file__),code_sha256=base.sha(__file__),
        training=base.read(base.HERE/'protocol.json')['training'],
        train_states=512,validation_states=32,opened_development_states=64,
        seeds=list(SEEDS),new_rollouts=0,physical_steps=0,
        evaluation='All existing panels including original16 and fresh-origin48 separately; same-state paired Q and selection comparisons, allfixed16 and state mismatch. No test-based checkpoint selection.',
        boundary='Slot identity is a diagnostic input intervention, not proof that Q is noninvariant or that this explains LOSO. Native policy noninvariance alone does not prove Q noninvariance on the natural distribution.',
        literature=[
            'https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html',
            'https://proceedings.mlr.press/v97/wagstaff19a.html'],
        literature_implication='Invariant architectures approximate invariant targets; theorem applicability does not establish that the controller-conditioned value is invariant. No width lower-bound violation is claimed.'))


def fit(seed):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic
    p=base.read(HERE/'protocol.json');assert base.sha(base.HERE/'dataset.npz')==p['dataset_sha256']
    d=dict(np.load(base.HERE/'dataset.npz'));tr=np.flatnonzero(d['split']=='train');va=np.flatnonzero(d['split']=='validation')
    cfg=p['training'];out=HERE/'fits'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')}
    eta=jnp.asarray(d['eta']);s=jnp.asarray(d['success'],jnp.float32);f=jnp.asarray(d['failure'],jnp.float32)
    net=Critic('state_field_compact')
    def args(ii,jj):
        c=jnp.zeros((len(ii),4,83),jnp.float32).at[:,:,-4:].set(jnp.eye(4))
        return {k:v[ii] for k,v in x.items()},c,eta[jj]
    par=net.init(jax.random.PRNGKey(seed),*args(jnp.array([0]),jnp.array([0])))
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(cfg['lr'],weight_decay=cfg['weight_decay']));opt=optimizer.init(par)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            pp,oo=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(p):
                z=net.apply(p,*args(ii,jj));ss,ff=s[ii,jj],f[ii,jj]
                return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(ss+ff,1))
            value,grad=jax.value_and_grad(loss)(pp);up,oo=optimizer.update(grad,oo,pp)
            return (optax.apply_updates(pp,up),oo),value
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def predict(params,ii):
        return jax.vmap(lambda j:net.apply(params,*args(ii,jnp.full(len(ii),j))),out_axes=1)(jnp.arange(16))
    def infer(params,ii):return np.concatenate([np.asarray(predict(params,jnp.asarray(ii[k:k+32]))) for k in range(0,len(ii),32)])
    good=(d['success'][tr]+d['failure'][tr])>0;rr,jj=np.where(good);pool=np.column_stack((tr[rr],jj));rng=np.random.default_rng(seed)
    best=None;history=[];started=time.monotonic()
    for step in range(cfg['validation_every'],cfg['steps']+1,cfg['validation_every']):
        draws=pool[rng.integers(len(pool),size=(cfg['validation_every'],cfg['batch']))]
        (par,opt),loss=block(par,opt,jnp.asarray(draws));z=infer(par,va).astype(float)
        m=base.metrics(z,*(d[k][va] for k in ('success','failure','unknown')))
        row=dict(step=step,training_nll=float(np.mean(loss)),**m);history.append(row)
        key=(m['regret'],-m['ranking'],m['state_eta_MSE'],m['nll'],step)
        if best is None or key<best:best=key;chosen=row;bb=serialization.to_bytes(par)
    (out/'checkpoint.msgpack').write_bytes(bb);par=serialization.from_bytes(par,bb)
    z=infer(par,np.arange(len(d['split']))).astype(float)
    np.savez_compressed(out/'predictions.npz',logits=z,probability=1/(1+np.exp(-np.clip(z,-100,100))))
    base.write(out/'history.json',history);base.write(out/'complete.json',dict(seed=seed,selected=chosen,seconds=time.monotonic()-started,
        protocol_sha256=base.sha(HERE/'protocol.json'),code_sha256=base.sha(__file__),test_used_for_selection=False))
    print(dict(seed=seed,selected=chosen,seconds=time.monotonic()-started),flush=True)


def review():
    from diagnostics.controller_entity_coverage_check_v1.review import corr,center,reversal
    from diagnostics.controller_ff_factorial_repair_v1.review import selection_difference,mse_gain,intervals,derangements
    d=dict(np.load(base.HERE/'dataset.npz'));arrays={};saved={};rows=[]
    grouped={}
    for arm in ('pooled','slot'):
        ps=[]
        for seed in SEEDS:
            p=HERE/'fits'/f'seed{seed}' if arm=='slot' else base.HERE/'fits/current_entity/n512'/f'seed{seed}'
            ps.append(np.load(p/'predictions.npz')['probability'])
        grouped[arm]=np.stack(ps)
    for panel in ('train','validation','test_original','test_fresh','test_all'):
        mask=d['split']==panel if not panel.startswith('test') else d['split']=='test'
        if panel in ('test_original','test_fresh'):mask &= d['origin']==panel[5:]
        ix=np.flatnonzero(mask);n=len(ix);cnt=d['success'][ix]+d['failure'][ix]+d['unknown'][ix]
        lo=d['success'][ix]/cnt;hi=(d['success'][ix]+d['unknown'][ix])/cnt;mid=(lo+hi)/2
        boot=np.random.default_rng(2026101001+n).integers(n,size=(20000,n));perms=derangements(n)
        saved[panel+'__state_uid']=d['state_uid'][ix];saved[panel+'__permutations']=perms
        for seed in list(SEEDS)+['ensemble']:
            pred={a:v.mean(0)[ix] if seed=='ensemble' else v[SEEDS.index(seed),ix] for a,v in grouped.items()}
            q,pred0=pred['slot'],pred['pooled'];sel=q.argmax(-1);sel0=pred0.argmax(-1);key=f'{panel}_{seed}'
            a,b=selection_difference(lo,hi,sel,sel0);selgain=intervals(a,b,boot,key+'__selection_gain',arrays)
            a,b=mse_gain(pred0,q,lo,hi);qgain=intervals(a.mean(1),b.mean(1),boot,key+'__Q_gain',arrays)
            ml,mu,ql,qu=[np.zeros(n) for _ in range(4)]
            for offset in range(1,n):
                donor=np.roll(np.arange(n),offset);a,b=selection_difference(lo,hi,sel,sel[donor]);ml+=a/(n-1);mu+=b/(n-1)
                a,b=mse_gain(q[donor],q,lo,hi);ql+=a.mean(1)/(n-1);qu+=b.mean(1)/(n-1)
            freq=np.bincount(sel,minlength=16)/n;nonzero=freq>0
            row=dict(panel=panel,N=n,seed=seed,Q_MSE_midpoint=float(np.mean((q-mid)**2)),
                state_effect_corr=corr(center(q),center(mid)),state_eta_corr=corr(center(q,True),center(mid,True)),
                success_bounds=[float(lo[np.arange(n),sel].mean()),float(hi[np.arange(n),sel].mean())],
                selection_gain_over_pooled=selgain,Q_MSE_gain_over_pooled=qgain,
                matched_minus_mismatch=intervals(ml,mu,boot,key+'__mismatch',arrays),
                wrong_state_MSE_minus_matched=intervals(ql,qu,boot,key+'__Q_mismatch',arrays),
                frequency=freq.tolist(),entropy_nats=float(-sum(freq[nonzero]*np.log(freq[nonzero]))))
            if seed=='ensemble':
                saved[panel+'__matched_predictions']=q;saved[panel+'__mismatched_predictions']=q[perms]
                dl,du=[],[]
                for j in range(16):
                    a,b=selection_difference(lo,hi,sel,np.full(n,j));dl.append(a);du.append(b)
                dl,du=np.stack(dl,1),np.stack(du,1);aa,bb=[],[]
                for k in range(0,len(boot),500):
                    ii=boot[k:k+500];aa.append(dl[ii].mean(1).min(-1));bb.append(du[ii].mean(1).min(-1))
                aa,bb=np.concatenate(aa),np.concatenate(bb)
                arrays[key+'__fixed_lower'],arrays[key+'__fixed_upper']=aa,bb
                row['minus_bootstrap_reselected_best_fixed']=dict(identification_bounds=[float(dl.mean(0).min()),float(du.mean(0).min())],
                    censor_CI95=[float(np.quantile(aa,.025)),float(np.quantile(bb,.975))])
                if panel.startswith('test'):row['reversal']=reversal(q,d['raw_outcomes'][ix],boot)
            rows.append(row)
    base.write(HERE/'development_results.json',dict(rows=rows,scope='Opened DEVELOPMENT only. Same-shape controlled input intervention. SourceVAL-only checkpoint. State bootstrap20000; censorbounds retained.',new_rollouts=0))
    np.savez_compressed(HERE/'development_bootstrap.npz',**arrays)
    np.savez_compressed(HERE/'development_predictions.npz',**saved)
    for r in rows:
        if r['panel']=='test_fresh':print({k:v for k,v in r.items() if k not in ['frequency','reversal']},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit','review']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='fit':fit(a.seed)
    else:review()
