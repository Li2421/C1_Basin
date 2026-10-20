"""Source-only six-dimensional compressed diagonal Gaussian schedule generator."""
from common import *
from critic_model import source_transform,transform_arrays
from frozen_networks import stable_int,CENTER,RADIUS,LOW,HIGH
from flax import serialization
import flax.linen as nn
import jax,jax.numpy as jnp,optax,itertools,math,time
from functools import lru_cache
from resource_guard import drain

class ScheduleGenerator(nn.Module):
    @nn.compact
    def __call__(self,h,c,C):
        hh=nn.silu(nn.Dense(96,name='ring_adapter')(h))
        cc=nn.silu(nn.Dense(32,name='context')(c))
        xx=nn.silu(nn.Dense(32,name='controller_encoder')(C))
        x=jnp.concatenate([hh,cc,xx],-1)
        x=nn.silu(nn.Dense(128,name='shared1')(x));x=nn.silu(nn.Dense(64,name='shared2')(x))
        return nn.Dense(12,name='out')(x)

def initialize():return ScheduleGenerator().init(jax.random.PRNGKey(41),jnp.zeros((1,100)),jnp.zeros((1,33)),jnp.zeros((1,24)))
def dist_params(raw):return raw[...,:6],.025+.275*jax.nn.sigmoid(raw[...,6:])
def log_prob(raw,eta):
    radius=jnp.tile(jnp.asarray(RADIUS),2);center=jnp.tile(jnp.asarray(CENTER),2)
    u=jnp.clip((eta-center)/radius,-.999999,.999999);z=jnp.arctanh(u);mu,sigma=dist_params(raw)
    return jnp.sum(-.5*(((z-mu)/sigma)**2+2*jnp.log(sigma)+math.log(2*math.pi))-jnp.log(radius*(1-u*u)+1e-9),-1)

def density_data(data):
    eta=data['action'][:,:,:40,:6];label=data['labels'][:,:,:40,0]
    low=np.tile(LOW,2);high=np.tile(HIGH,2)
    supported=np.all((eta>=low)&(eta<=high),-1)
    positive=supported&(label==1);negative=supported&(label==0)
    weights=np.zeros_like(label,dtype=float)
    for ni in range(len(label)):
        valid=positive[ni].any(-1)
        if valid.any():weights[ni]=positive[ni]/(valid.sum()*np.maximum(positive[ni].sum(-1),1)[:,None])
    return eta,positive,negative,weights

def source_weights(data,source,positive,base_weights):
    weights=np.zeros_like(base_weights);gaps={}
    supported_types=[]
    for t in source:
        ids=np.flatnonzero((data['types']==t)&positive.any((1,2)))
        gaps[t]=dict(controllers_without_positive=data['names'][(data['types']==t)&~positive.any((1,2))].tolist(),supported_controllers=len(ids))
        if len(ids):supported_types.append(t);weights[ids]=base_weights[ids]/len(ids)
    assert supported_types,'No source positive density targets'
    weights/=len(supported_types);assert np.isclose(weights.sum(),1)
    assert np.all(weights[~np.isin(data['types'],source)]==0)
    return weights,gaps

@lru_cache(None)
def compiled():
    net=ScheduleGenerator();opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(5e-4,weight_decay=1e-4))
    @jax.jit
    def block(parameters,state,ids,negative_ids,has_negative,arrays,targets):
        ns,na=targets.shape[1:3]
        def step(carry,batch):
            p,st=carry;ix,neg,hn=batch;ci=ix//(ns*na);si=ix//na%ns;ai=ix%na
            def loss(pp):
                raw=net.apply(pp,arrays['h'][si],arrays['c'][si],arrays['controller'][ci]);lp=log_prob(raw,targets[ci,si,ai]);ln=log_prob(raw,targets[ci,si,neg])
                rank=jnp.sum(hn*jax.nn.softplus(.5+ln-lp))/jnp.maximum(hn.sum(),1)
                return -lp.mean()+.25*rank
            ll,g=jax.value_and_grad(loss)(p);updates,st=opt.update(g,st,p);return (optax.apply_updates(p,updates),st),ll
        return jax.lax.scan(step,(parameters,state),(ids,negative_ids,has_negative))
    @jax.jit
    def log_density(p,arrays,targets):
        nc,ns,na=targets.shape[:3]
        h=jnp.broadcast_to(arrays['h'][None],(nc,ns,100));c=jnp.broadcast_to(arrays['c'][None],(nc,ns,33));C=jnp.broadcast_to(arrays['controller'][:,None],(nc,ns,24))
        raw=net.apply(p,h,c,C);return log_prob(raw[:,:,None,:],targets)
    return opt,block,log_density

def folder_for(task):return HERE/'schedule_generator_fits'/task['source_key']
def plan():
    from train_critic import key_for,load_data
    data=load_data();types=sorted(set(data['types'].tolist()));tasks=[]
    for count in (17,18,19):
        for source in itertools.combinations(types,count):tasks.append(dict(source_types=source,source_key=key_for(source)))
    assert len(tasks)==191
    configuration=dict(utc=utc(),tasks=tasks,steps=list(range(100,3001,100)),batch=64,seed=41,learning_rate=5e-4,weight_decay=1e-4,gradient_clip=5,positive_nll_weight=1.,negative_rank_weight=.25,negative_rank_margin=.5,initialization='scratch',bundle_sha256=sha(HERE/'TRAINING_BUNDLE.npz'))
    write(HERE/'GENERATOR_TRAINING_PLAN.json',configuration)
    seal('GENERATOR_TRAINING_PLAN.receipt.json',[HERE/n for n in ('GENERATOR_TRAINING_PLAN.json','TRAINING_BUNDLE.receipt.json','schedule_generator.py','MODEL_TESTS.receipt.json')],source_only=True)

def fit(data,task,configuration):
    folder=folder_for(task);folder.mkdir(parents=True,exist_ok=True);write(folder/'TASK.json',dict(**task,utc=utc()))
    transform=source_transform(data,task['source_types']);write(folder/'TRANSFORM.json',transform)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items() if k!='action'}
    eta,pos,neg,base_weights=density_data(data);weights,gaps=source_weights(data,task['source_types'],pos,base_weights)
    write(folder/'POSITIVE_SUPPORT.json',gaps);targets=jnp.asarray(eta)
    # Empty negative pools use a valid positive target, with the ranking mask zero.
    negative_order=np.argsort(~neg,axis=-1,kind='stable');negative_count=neg.sum(-1)
    opt,block,evaluate=compiled();parameters=initialize();state=opt.init(parameters)
    rng=np.random.default_rng(stable_int('crosstype5-schedule-training',41,task['source_key']));history=[];paths=[];last=0;start=time.monotonic()
    validation=sorted(set(data['types'])-set(task['source_types'])) or sorted(set(data['types']))
    for step in configuration['steps']:
        ix=rng.choice(weights.size,size=(step-last,configuration['batch']),p=weights.reshape(-1)).astype(np.int32);ci=ix//(48*40);si=ix//40%48
        count=negative_count[ci,si];has=count>0;order=(rng.random(ix.shape)*np.maximum(count,1)).astype(np.int32);nix=negative_order[ci,si,order];nix=np.where(has,nix,ix%40).astype(np.int32)
        (parameters,state),loss=block(parameters,state,jnp.asarray(ix),jnp.asarray(nix),jnp.asarray(has,dtype=jnp.float32),arrays,targets)
        lp=np.asarray(evaluate(parameters,arrays,targets));summary={}
        for t in validation:
            ids=np.flatnonzero((data['types']==t)&pos.any((1,2)));values=np.sum(-lp[ids]*base_weights[ids],(1,2))
            summary[t]=dict(positive_nll=float(values.mean()) if len(values) else None,supported_controllers=len(ids),positive_targets=int(pos[data['types']==t].sum()))
        path=folder/f'step{step}.msgpack';path.write_bytes(serialization.to_bytes(parameters));paths.append(path)
        history.append(dict(step=step,training_loss=float(np.asarray(loss).mean()),validation=summary,elapsed_seconds=time.monotonic()-start,validation_used_for_early_stop=False));write(folder/'HISTORY.json',history)
        print(json.dumps(dict(source_key=task['source_key'],sources=len(task['source_types']),step=step,elapsed=time.monotonic()-start)),flush=True);last=step
    seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/'TASK.json',folder/'TRANSFORM.json',folder/'HISTORY.json',folder/'POSITIVE_SUPPORT.json',HERE/'GENERATOR_TRAINING_PLAN.receipt.json',HERE/'schedule_generator.py',*paths],source_only_gradient=True,source_only_preprocessing=True,backend=jax.default_backend())

def run(worker):
    from train_critic import load_data
    for p,h in read(HERE/'GENERATOR_TRAINING_PLAN.receipt.json')['files'].items():assert sha(p)==h,p
    assert jax.default_backend()=='gpu';config=read(HERE/'GENERATOR_TRAINING_PLAN.json');data=load_data();claims=HERE/'generator_training_claims';claims.mkdir(exist_ok=True)
    with jax.default_matmul_precision('highest'):
        for task in config['tasks']:
            if drain():break
            if (folder_for(task)/'COMPLETE.receipt.json').exists():continue
            try:fd=os.open(claims/f"{task['source_key']}.json",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as f:json.dump(dict(utc=utc(),worker=worker,pid=os.getpid(),task=task),f)
            fit(data,task,config)
    write(HERE/f'GENERATOR_WORKER_{worker}_COMPLETE.json',dict(utc=utc()))

def model_record(source,step):
    from train_critic import key_for
    folder=HERE/'schedule_generator_fits'/key_for(source);receipt=read(folder/'COMPLETE.receipt.json')
    result=dict(source_types=sorted(source),step=step)
    for key,path in [('checkpoint',folder/f'step{step}.msgpack'),('transform',folder/'TRANSFORM.json'),('fit_receipt',folder/'COMPLETE.receipt.json')]:
        if key!='fit_receipt':assert sha(path)==receipt['files'][str(path)]
        result[key+'_path']=str(path);result[key+'_sha256']=sha(path)
    return result

def select(source,configuration):
    from train_critic import key_for
    histories={}
    for held in source:
        folder=HERE/'schedule_generator_fits'/key_for([t for t in source if t!=held]);assert (folder/'COMPLETE.receipt.json').exists()
        assert set(read(folder/'TASK.json')['source_types'])==set(source)-{held}
        histories[held]={r['step']:r['validation'][held] for r in read(folder/'HISTORY.json')}
    curve=[]
    for step in configuration['steps']:
        rows={t:histories[t][step] for t in source};values=[r['positive_nll'] for r in rows.values() if r['positive_nll'] is not None]
        curve.append(dict(step=step,source_positive_nll=float(np.mean(values)) if values else None,folds=rows,unsupported_types=[t for t,r in rows.items() if r['positive_nll'] is None]))
    supported=[r for r in curve if r['source_positive_nll'] is not None];assert supported
    picked=min(supported,key=lambda r:(round(r['source_positive_nll'],12),r['step']))
    return model_record(source,picked['step']),dict(source_types=source,selected=picked,curve=curve,outer_labels_used=False)

def predict(model,data,state_uids):
    for key in ('checkpoint','transform','fit_receipt'):assert sha(model[key+'_path'])==model[key+'_sha256']
    transform=read(model['transform_path'])
    for k in ('h_mean','h_scale','c_mean','c_scale'):transform[k]=np.asarray(transform[k],np.float32)
    for k in ('feature_fill','feature_mean','feature_scale'):transform[k]=np.asarray(transform[k],np.float64)
    arrays=transform_arrays(dict(data,action=np.zeros((1,1,1,14))),transform);p=serialization.from_bytes(initialize(),Path(model['checkpoint_path']).read_bytes())
    nc=len(data['names']);ns=len(data['h']);h=np.broadcast_to(arrays['h'][None],(nc,ns,100));c=np.broadcast_to(arrays['c'][None],(nc,ns,33));C=np.broadcast_to(arrays['controller'][:,None],(nc,ns,24))
    raw=np.asarray(ScheduleGenerator().apply(p,jnp.asarray(h),jnp.asarray(c),jnp.asarray(C)));mu,sigma=dist_params(jnp.asarray(raw))
    noise=np.stack([np.vstack([np.zeros(6),np.random.default_rng(stable_int('crosstype5-schedule-proposals',41,str(uid))).normal(size=(8,6))]) for uid in state_uids]).astype(np.float32)
    proposals=jnp.tile(jnp.asarray(CENTER),2)+jnp.tile(jnp.asarray(RADIUS),2)*jnp.tanh(mu[:,:,None,:]+sigma[:,:,None,:]*jnp.asarray(noise)[None])
    result=np.asarray(proposals).reshape(nc,ns,9,2,3);assert np.isfinite(result).all();return result

if __name__=='__main__':plan() if sys.argv[1]=='plan' else run(int(sys.argv[2]))
