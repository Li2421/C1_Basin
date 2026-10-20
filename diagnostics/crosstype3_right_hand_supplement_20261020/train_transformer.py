"""Fixed six offline Gaussian Transformer fits; no rollout outcomes are loaded."""
from common import *
from unseen_models import TransformerPolicy,gaussian_nll
from new_benchmark_common.dataset import JointTransitionDataset
import jax,jax.numpy as jnp,optax,pickle,time
from flax import serialization
from functools import lru_cache
jax.config.update('jax_enable_x64',False)

@lru_cache(None)
def compiled(layers,rate):
    net=TransformerPolicy(layers);opt=optax.adam(rate)
    def loss(p,x,y):return gaussian_nll(*net.apply(p,x),y).mean()
    @jax.jit
    def block(p,st,X,Y,ids):
        def update(carry,idx):
            p,st=carry;g=jax.grad(loss)(p,X[idx],Y[idx]);delta,st=opt.update(g,st,p)
            return (optax.apply_updates(p,delta),st),None
        return jax.lax.scan(update,(p,st),ids)[0]
    return net,opt,block,jax.jit(loss)

def train(index):
    prereg()
    for name in ('EXECUTION_REPRODUCTION','GENERATOR_REPRODUCTION','TRANSFORMER_UNIT_TESTS','OPTIMIZER_UNIT_TESTS'):
        assert read(HERE/f'{name}.json')['passed']
    assert all(d.platform=='gpu' for d in jax.devices())
    cfg=read(HERE/'TRANSFORMER_TRAINING_CONFIGS.json')[index]
    folder=HERE/'policies'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.receipt.json').exists():return
    claim=folder/'TRAINING_CLAIM.json'
    with claim.open('x') as f:json.dump(dict(utc=utc(),pid=os.getpid(),job=os.getenv('SLURM_JOB_ID')),f)
    assert sha(cfg['manifest'])==cfg['manifest_sha256']
    assert sha(cfg['flow_checkpoint'])==cfg['flow_checkpoint_sha256']
    original=pickle.loads(Path(cfg['flow_checkpoint']).read_bytes())['config']
    normal={k:np.asarray(original[k],np.float32).tolist() for k in ('obs_mean','obs_scale','act_mean','act_scale')}
    normal['max_speed']=float(original['max_speed'])
    write(folder/'NORMALIZATION.json',normal);write(folder/'CONFIG.json',cfg)
    seal(str(folder.relative_to(HERE)/'CONFIG_FROZEN.receipt.json'),[folder/'CONFIG.json',folder/'NORMALIZATION.json',HERE/'train_transformer.py',HERE/'unseen_models.py'],before_training=True)
    trainset=JointTransitionDataset(Path(cfg['manifest']).parent,'train',seed=cfg['seed'])
    dev=JointTransitionDataset(Path(cfg['manifest']).parent,'dev',seed=cfg['validation_seed'])
    def norm(a,m,s):return (np.asarray(a,np.float32).reshape(len(a),-1)-np.asarray(normal[m],np.float32))/np.asarray(normal[s],np.float32)
    vi=np.random.default_rng(cfg['validation_seed']).integers(len(dev.actions),size=2048);np.save(folder/'DEV_INDICES.npy',vi)
    X,Y,XV,YV=map(jnp.asarray,(norm(trainset.observations,'obs_mean','obs_scale'),norm(trainset.actions,'act_mean','act_scale'),norm(dev.observations[vi],'obs_mean','obs_scale'),norm(dev.actions[vi],'act_mean','act_scale')))
    net,opt,block,loss=compiled(cfg['layers'],cfg['learning_rate']);pa=net.init(jax.random.PRNGKey(cfg['seed']),jnp.zeros((1,92),jnp.float32));st=opt.init(pa)
    best=float(loss(pa,XV,YV));assert np.isfinite(best);beststep=0;history=[dict(step=0,dev_loss=best)];(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
    rng=np.random.default_rng(cfg['seed']);before=time.monotonic()
    for target in range(250,cfg['steps']+1,250):
        ids=jnp.asarray(rng.integers(len(X),size=(250,256)));pa,st=block(pa,st,X,Y,ids)
        value=float(loss(pa,XV,YV));assert np.isfinite(value);history.append(dict(step=target,dev_loss=value))
        if value<best:best=value;beststep=target;(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
        write(folder/'TRAINING_HISTORY.partial.json',history)
        print(json.dumps(dict(name=cfg['name'],step=target,dev_loss=value,elapsed_seconds=time.monotonic()-before)),flush=True)
    write(folder/'TRAINING_HISTORY.json',history);(folder/'final.msgpack').write_bytes(serialization.to_bytes(pa))
    write(folder/'SUMMARY.json',dict(best_step=beststep,best_loss=best,steps=cfg['steps'],train_transitions=len(trainset.actions),dev_transitions=len(dev.actions),x64=False,elapsed_seconds=time.monotonic()-before))
    seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/f for f in ('CONFIG.json','NORMALIZATION.json','best.msgpack','final.msgpack','TRAINING_HISTORY.json','DEV_INDICES.npy','SUMMARY.json')]+[HERE/'train_transformer.py',HERE/'unseen_models.py'],closed_loop_labels_read=False,test_split_opened=False)

if __name__=='__main__':
    with jax.default_matmul_precision('highest'):
        worker=int(sys.argv[1]);[train(i) for i in range(worker,6,2)]
