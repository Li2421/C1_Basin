"""Zero-rollout check: can the current entity encoder learn the proven Ring task?"""
import argparse,time
from pathlib import Path
import numpy as np
from diagnostics.controller_four_scene_mechanism_v1.study import read,write,freeze,sha
from diagnostics.state_nuisance_rootcause_v1.real_state_expansion import OUT as SOURCE

HERE=Path(__file__).resolve().parent
JOBS=[dict(arm=a,n=n,seed=s) for n in (32,128,512) for a in ('eta_only','physical23','current_entity') for s in (17,23,41)]


def prepare():
    from diagnostics.controller_ff_crossed_cache_full_v1.study import full_physical
    from diagnostics.orthoflow3_unified_representation_v1.representation import batch
    p=read(SOURCE/'protocol.json');d=dict(np.load(SOURCE/'test_dataset.npz'));features=dict(np.load(SOURCE/'physical_inputs.npz'))
    assert len(p['states'])==len(d['split'])==608
    assert (d['present'].any((1,2))).all()
    physical=[full_physical('ring_exchange',s['physical'],p['environment']['config']['config']) for s in p['states']]
    entities=batch(physical)
    data={'entity_'+k:v for k,v in entities.items()}
    data.update(physical23=features['physical23'],eta=features['eta'],success=d['success'],failure=d['failure'],unknown=d['numerical'],
        split=d['split'],origin=d['origin'],state_uid=np.array([s['state_uid'] for s in p['states']]),
        raw_outcomes=d['outcomes'],present=d['present'])
    freeze(HERE/'protocol.json',dict(schema='current_entity_coverage_check_v1',new_rollouts=0,
        purpose='Separate low-state-coverage failure from inability of current physical encoding/encoder to learn an already-proven realFF Ring relation. Not a repair or confirmation of cross-scene/controller transfer.',
        data='Entire original frozen Ring actualFF512TRAIN/32VAL/64openedTEST cache. All16original eta; Q4TRAIN/Q16VAL/TEST; preserve unknown labels. Reencode only, no simulator or controller calls.',
        source_sha256={name:sha(SOURCE/name) for name in ('protocol.json','test_dataset.npz','physical_inputs.npz','final_cache_identity_audit.json','runtime_integrity.json')},
        inherited_controller=p['controller'],inherited_environment=p['environment'],
        arms=['matched eta-only','original physical23+64/64MLP','current masked entity encoder and critic head,83controller channels zero'],
        comparison_boundary='This checks representation plus encoder family, not an equal-parameter pure-feature ablation. Positive entity results demonstrate capacity on this cached task, not sufficient controller-conditioned transfer.',
        training=dict(steps=12000,batch=128,validation_every=200,lr=.0008,weight_decay=.0001,seeds=[17,23,41],prefixes=[32,128,512]),
        selection='Independent sourceVAL only: pessimistic selection regret, strong pair ranking(gap>=.25), state-eta centered error, NLL, earliest step. Same rule for all arms.',
        normalization='physical23 retains validated original32TRAIN normalization at every prefix; eta retains original fixed candidate normalization; entity deterministic physical units with known execution constants. No TEST fitting.',
        evaluation='Previously opened states, DEVELOPMENT only. Primary old fresh-origin48 TEST stratum; original16 TEST separately. Whole-state mismatch, all16fixed policies and bootstrap within physical states.',
        sampling_audit='Original Ring screen states were frozen before FF outcomes; all12were retained by hash split, not selected by successes. The remainder uses unchanged native development sample_instance. No new state IDs.',
        literature=['https://arxiv.org/abs/2010.09536','https://proceedings.mlr.press/v119/raileanu20a.html']))
    np.savez_compressed(HERE/'dataset.npz',**data)
    freeze(HERE/'dataset_audit.json',dict(sha256=sha(HERE/'dataset.npz'),new_rollouts=0,reused_rollouts=int(d['present'].sum()),
        train_states=512,validation_states=32,opened_test_states=64,unknown=int(d['numerical'].sum()),
        physical_only=True,controller_queries=0))
    print(read(HERE/'dataset_audit.json'),flush=True)


def metrics(z,s,f,u):
    q=1/(1+np.exp(-np.clip(z,-100,100)));lo=s/(s+f+u);hi=(s+u)/(s+f+u)
    selected=q.argmax(-1);delta=hi-lo[np.arange(len(q)),selected,None];delta[np.arange(len(q)),selected]=0
    regret=float(np.maximum(delta.max(-1),0).mean())
    a,b=np.triu_indices(16,1);pos=lo[:,a]-hi[:,b]>=.25;neg=lo[:,b]-hi[:,a]>=.25;strong=pos|neg
    diff=q[:,a]-q[:,b];score=np.where(diff==0,.5,np.where(pos,diff>0,diff<0))
    center=lambda x:x-x.mean(0,keepdims=True)-x.mean(1,keepdims=True)+x.mean()
    return dict(regret=regret,ranking=float(score[strong].mean()) if strong.any() else .5,
        state_eta_MSE=float(np.mean((center(q)-center(lo))**2)),
        nll=float(np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/np.sum(s+f)))


def fit(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.controller_q_diversity_v1.train import Critic
    from diagnostics.state_nuisance_rootcause_v1.state_only_experiment import model
    job=JOBS[index];p=read(HERE/'protocol.json')['training'];d=dict(np.load(HERE/'dataset.npz'));audit=read(HERE/'dataset_audit.json')
    assert sha(HERE/'dataset.npz')==audit['sha256']
    out=HERE/'fits'/job['arm']/f"n{job['n']}"/f"seed{job['seed']}";out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    tr=np.flatnonzero(d['split']=='train')[:job['n']];va=np.flatnonzero(d['split']=='validation')
    freeze(out/'job.json',dict(**job,data_sha256=audit['sha256'],protocol_sha256=sha(HERE/'protocol.json'),code_sha256=sha(__file__),
        train_state_ids=d['state_uid'][tr].tolist(),validation_state_ids=d['state_uid'][va].tolist(),test_used_for_selection=False))
    x={k[7:]:jnp.asarray(v,jnp.float32) for k,v in d.items() if k.startswith('entity_')};flat=jnp.asarray(d['physical23']);eta=jnp.asarray(d['eta'])
    ss,ff=jnp.asarray(d['success'],jnp.float32),jnp.asarray(d['failure'],jnp.float32)
    net=model('local_state_eta') if job['arm']=='physical23' else Critic('eta_only' if job['arm']=='eta_only' else 'state_field_compact')
    def args(ii,jj):
        if job['arm']=='physical23':return flat[ii],eta[jj],jnp.zeros((len(ii),76),jnp.float32)
        return {k:v[ii] for k,v in x.items()},jnp.zeros((len(ii),4,83),jnp.float32),eta[jj]
    params=net.init(jax.random.PRNGKey(job['seed']),*args(jnp.array([0]),jnp.array([0])))
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(p['lr'],weight_decay=p['weight_decay']));opt=optimizer.init(params)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            par,optim=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(pp):
                z=net.apply(pp,*args(ii,jj));s,f=ss[ii,jj],ff[ii,jj]
                return jnp.mean((s*jax.nn.softplus(-z)+f*jax.nn.softplus(z))/jnp.maximum(s+f,1))
            loss0,grad=jax.value_and_grad(loss)(par);up,optim=optimizer.update(grad,optim,par)
            return (optax.apply_updates(par,up),optim),loss0
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def predict(par,ii):
        return jax.vmap(lambda j:net.apply(par,*args(ii,jnp.full(len(ii),j))),out_axes=1)(jnp.arange(16))
    def infer(par,ii):return np.concatenate([np.asarray(predict(par,jnp.asarray(ii[k:k+32]))) for k in range(0,len(ii),32)])
    good=(d['success'][tr]+d['failure'][tr])>0;rr,jj=np.where(good);pool=np.column_stack((tr[rr],jj));rng=np.random.default_rng(job['seed']);best=None;history=[];started=time.monotonic()
    for step in range(p['validation_every'],p['steps']+1,p['validation_every']):
        draws=pool[rng.integers(len(pool),size=(p['validation_every'],p['batch']))]
        (params,opt),loss=block(params,opt,jnp.asarray(draws));z=infer(params,va).astype(float)
        m=metrics(z,*(d[k][va] for k in ('success','failure','unknown')));row=dict(step=step,training_nll=float(np.mean(loss)),**m);history.append(row)
        key=(m['regret'],-m['ranking'],m['state_eta_MSE'],m['nll'],step)
        if best is None or key<best:best=key;chosen=row;best_bytes=serialization.to_bytes(params)
        if step%2000==0:write(out/'history.json',history);print(dict(**job,step=step,best_step=chosen['step'],seconds=round(time.monotonic()-started,1)),flush=True)
    (out/'checkpoint.msgpack').write_bytes(best_bytes);par=serialization.from_bytes(params,best_bytes);z=infer(par,np.arange(len(d['split']))).astype(float)
    np.savez_compressed(out/'predictions.npz',logits=z,probability=1/(1+np.exp(-np.clip(z,-100,100))))
    write(out/'history.json',history);write(out/'complete.json',dict(**job,selected=chosen,seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit']);p.add_argument('--index',type=int);a=p.parse_args()
    prepare() if a.stage=='prepare' else fit(a.index)
