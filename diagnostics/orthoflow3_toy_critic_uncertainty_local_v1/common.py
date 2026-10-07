"""Frozen W1 ensemble inference shared by VAL, hard TEST, and local audit."""
import importlib.util,json,os,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
import jax,jax.numpy as jnp,numpy as np
from flax import serialization
ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=Path(__file__).resolve().parent
RANK=ROOT/'diagnostics/orthoflow3_ranking_aware_critic_v1'
spec=importlib.util.spec_from_file_location('uncertainty_ranklib',RANK/'run_experiment.py')
lib=importlib.util.module_from_spec(spec);sys.modules[spec.name]=lib;spec.loader.exec_module(lib)
BASE=ROOT/'diagnostics/orthoflow3_nll_weighting_ablation_v1'
man=json.loads((ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json').read_text())
state_norm=man['state_normalization']['Toy']
ETA_CENTER=np.asarray(man['eta_normalization']['center'],np.float32)
ETA_SCALE=np.asarray(man['eta_normalization']['scale'],np.float32)
DOMAIN_CENTER=np.asarray([.625,0.,.375],np.float32)
DOMAIN_RADIUS=np.asarray([.625,.5,.375],np.float32)
SEEDS=(17,23,41,47,59)

def checkpoints():
    out=[]
    for seed in SEEDS:
        cp=(BASE/'models/secondary_combined/W1'/f'seed{seed}'/'checkpoint.msgpack') if seed in (17,23,41) else (HERE/f'seed{seed}'/'checkpoint.msgpack')
        out.append(cp)
    return out

def normalize_h(h):
    h=np.asarray(h,np.float32)
    return (h-np.asarray(state_norm['mean'],np.float32))/np.asarray(state_norm['std'],np.float32)

def normalize_eta(eta):
    return (np.asarray(eta,np.float32)-ETA_CENTER)/ETA_SCALE

def predict_members(h,eta,batch_size=4096):
    h=normalize_h(h);eta=normalize_eta(eta)
    assert len(h)==len(eta)
    model=lib.SingleCritic()
    template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214),jnp.float32),jnp.zeros((1,3),jnp.float32))
    apply=jax.jit(lambda p,x,e:model.apply(p,x,e))
    result=[]
    for cp in checkpoints():
        params=serialization.from_bytes(template,cp.read_bytes())
        z=[]
        for start in range(0,len(h),batch_size):
            x=jnp.asarray(h[start:start+batch_size]);e=jnp.asarray(eta[start:start+batch_size])
            z.append(np.asarray(apply(params,x,e)))
        logits=np.concatenate(z)
        result.append(1/(1+np.exp(-np.clip(logits,-30,30))))
    return np.stack(result,axis=0)

def virtual_neighborhood(h,eta,radius):
    """Return N×6 virtual axis perturbations, clipped to frozen E_bridge."""
    h=np.asarray(h,np.float32);eta=np.asarray(eta,np.float32)
    offsets=np.concatenate([np.eye(3),-np.eye(3)],axis=0)*float(radius)*DOMAIN_RADIUS
    pert=np.clip(eta[:,None,:]+offsets[None,:,:],DOMAIN_CENTER-DOMAIN_RADIUS,DOMAIN_CENTER+DOMAIN_RADIUS)
    return np.repeat(h,6,axis=0),pert.reshape(-1,3)
