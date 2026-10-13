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


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','fit']);ap.add_argument('--seed',type=int);a=ap.parse_args()
    prepare() if a.stage=='prepare' else fit(a.seed)
