"""Representation-only matched learning; established Gaussian/NLL and Q-BCE."""
import os
_backend=os.environ.get('JAX_PLATFORMS','cpu');_devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
import argparse,json
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp,optax
from flax import serialization
from . import representation as rep,build
from diagnostics.orthoflow3_closed_loop_contract_v1 import repair
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a,phase_b as b,train_phase_a as ta
os.environ['JAX_PLATFORMS']=_backend;os.environ['CUDA_VISIBLE_DEVICES']=_devices
base=a.learn;OUT=build.OUT;SEEDS=(17,23)

def prepare(kind):
    repair.install()
    data=b.prepare_generator() if kind=='generator' else b.prepare_critic()[0]
    source=build.rows()+a.load(OUT/'initial_expansion_entities.json');by={r['state_uid']:r for r in source}
    ordered=[r for sc in a.SCENARIOS for r in data[sc]['states']]
    physical=rep.batch([rep.entities(build.scene(by[r['state_uid']])) for r in ordered])
    offset=0
    for sc,d in data.items():
        # Indices are host-side gather keys ONLY, never fed to a neural network.
        d['h']=np.arange(offset,offset+len(d['states']),dtype=np.float32)[:,None]
        d['c']=np.zeros((len(d['states']),0),np.float32);offset+=len(d['states'])
    return data,physical

def gather(physical,h):
    ids=np.asarray(h).ravel().astype(int)
    return {k:jnp.asarray(v[ids]) for k,v in physical.items()}

class ValidationProxy:
    def __init__(self,model,physical,critic=False):
        self.physical=physical;self.critic=critic
        self.predict=jax.jit(lambda params,x,e:model.apply(params,x,e)) if critic else jax.jit(lambda params,x:model.apply(params,x))
    def db(self):pass
    def four(self):pass
    def ring(self):pass
    def apply(self,params,h,c,e=None,method=None):
        x=gather(self.physical,h)
        return self.predict(params,x,e) if self.critic else self.predict(params,x)

def initialize(model,physical,critic=False):
    x={k:jnp.asarray(v[:1]) for k,v in physical.items()}
    return model.init(jax.random.PRNGKey(0),x,jnp.zeros((1,3))) if critic else model.init(jax.random.PRNGKey(0),x)

def train(kind,seed):
    assert a.load(OUT/'structural_tests.json')['passed']
    if kind=='critic':assert (OUT/'generator_frozen.json').exists()
    data,physical=prepare(kind);critic=kind=='critic';model=rep.Critic() if critic else rep.Generator()
    params=initialize(model,physical,critic);proxy=ValidationProxy(model,physical,critic)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3 if critic else 5e-4,weight_decay=1e-4));st=opt.init(params)
    @jax.jit
    def step(params,st,batches):
        def objective(p):
            losses=[]
            for x,e1,e2,w in batches:
                if critic:
                    z=model.apply(p,x,e1);loss=jnp.sum(w*optax.sigmoid_binary_cross_entropy(z,e2))/jnp.sum(w)
                else:
                    raw=model.apply(p,x);pos=-jnp.mean(base.log_prob(raw,e1))
                    rank=jnp.sum(w*jax.nn.softplus(.5+base.log_prob(raw,e2)-base.log_prob(raw,e1)))/jnp.maximum(jnp.sum(w),1.)
                    loss=pos+.25*rank
                losses.append(loss)
            return jnp.mean(jnp.stack(losses))
        value,grad=jax.value_and_grad(objective)(params);up,st=opt.update(grad,st,params)
        return optax.apply_updates(params,up),st,value
    rng={sc:np.random.default_rng(base.stable_int(kind,seed,sc)) for sc in a.SCENARIOS}
    history=[];best=(float('inf'),None,0);stale=0
    for it in range(1,(4000 if critic else 3000)+1):
        batches=[]
        for sc in a.SCENARIOS:
            h,c,e1,e2,w=ta.batch(data,sc,'train',rng[sc]) if critic else base.generator_batch(data,sc,'train',rng[sc])
            batches.append((gather(physical,h),jnp.asarray(e1),jnp.asarray(e2),jnp.asarray(w)))
        params,st,loss=step(params,st,batches)
        if it%100:continue
        val=base.critic_val_loss(proxy,params,data) if critic else base.generator_validation_loss(proxy,params,data)
        history.append({'step':it,'train':float(loss),'validation':val});print(json.dumps({'kind':kind,'seed':seed,**history[-1]}),flush=True)
        if val<best[0]-1e-5:best=(val,serialization.to_bytes(params),it);stale=0
        else:stale+=1
        if stale>=(10 if critic else 8) and it>=(1500 if critic else 1200):break
    path=OUT/f'{kind}/seed{seed}/checkpoint.msgpack';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(best[1])
    build.dump(f'{kind}/seed{seed}/history.json',history)
    result={'seed':seed,'best_step':best[2],'steps':it,'validation_loss':best[0],'checkpoint':str(path),'sha256':build.sha(path),
            'exposure_per_scenario':it*(96 if critic else 64),'encoder_sha256':build.sha(rep.__file__),
            'labels_sha256':build.sha(build.DATA/'eta_labels.parquet'),'train_only_preprocessing':True,
            'only_primary_change':'shared physical entity representation',
            'proposal_aligned_evidence':'same frozen Phase-A full-Q16 train/dev proposal sets as matched R_old; no new labels'}
    if critic:
        result['source_exposure']={sc:dict(d['source_exposure']) for sc,d in data.items()}
        build.dump(f'{kind}/seed{seed}/metrics.json',ta.metrics(proxy,serialization.from_bytes(params,best[1]),data))
    build.dump(f'{kind}/seed{seed}/training.json',result);return result

def freeze(kind):
    rr=[a.load(OUT/f'{kind}/seed{s}/training.json') for s in SEEDS];selected=min(rr,key=lambda r:(r['validation_loss'],r['seed']))
    m={'selected':selected,'results':rr,'selected_on':'validation_loss_only','K_stochastic':16,
       'proposal_convention':'same accepted location candidate + 16 stochastic, finite ranking',
       'architecture':'shared physical entities, no scenario adapters','no_confirmation_labels':True}
    build.dump(f'{kind}_frozen.json',m);return m

def load_models():
    sample=rep.batch([rep.entities(build.scene(build.rows()[0]))]);gm,cm=rep.Generator(),rep.Critic()
    gp=serialization.from_bytes(initialize(gm,sample),Path(a.load(OUT/'generator_frozen.json')['selected']['checkpoint']).read_bytes())
    cp=serialization.from_bytes(initialize(cm,sample,True),Path(a.load(OUT/'critic_frozen.json')['selected']['checkpoint']).read_bytes())
    return gm,gp,cm,cp

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['generator','critic','freeze_generator','freeze_critic']);p.add_argument('--seed',type=int,default=17);x=p.parse_args()
    print(json.dumps(freeze(x.stage.split('_')[1]) if x.stage.startswith('freeze') else train(x.stage,x.seed),indent=2))
