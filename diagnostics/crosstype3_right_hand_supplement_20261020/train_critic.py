"""Three-seed source-set cached GPU training over sparse pure schedule labels."""
from common import *
from critic_model import *
from frozen_networks import stable_int
from flax import serialization
from resource_guard import drain
import jax,jax.numpy as jnp,optax,itertools,time
from functools import lru_cache
jax.config.update('jax_enable_x64',False)
def key_for(source):return hashlib.sha256(json.dumps(sorted(source)).encode()).hexdigest()[:20]
def folder_for(task):return HERE/'critic_fits'/task['mode']/task['source_key']
def load_data(path=None):
    path=Path(path or HERE/'TRAINING_BUNDLE.npz');assert path.resolve()==(HERE/'TRAINING_BUNDLE.npz').resolve()
    assert sha(path)==read(HERE/'TRAINING_BUNDLE.receipt.json')['files'][str(path.resolve())]
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert data['labels'].shape==(104,48,26,1) and not np.any(data['labels']==-2)
    assert data['action'].shape==(104,48,26,10) and data['h'].shape==(48,100) and data['c'].shape==(48,33)
    assert len(set(data['types']))==16 and data['features'].shape==(104,24)
    assert np.all(data['candidate_indices'][:,:,:8]//40==data['candidate_indices'][:,:,:8]//5%8)
    assert np.all(data['candidate_indices'][:,:,8:]//40!=data['candidate_indices'][:,:,8:]//5%8)
    return data
@lru_cache(None)
def compiled(mode):
    net=network(mode);opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(1e-3,weight_decay=1e-4))
    @jax.jit
    def block(parameters,states,indices,arrays,Y,N):
        ns,na=arrays['action'].shape[1:3]
        def one(p,st,ids):
            ci=ids//(ns*na);si=ids//na%ns;ai=ids%na
            def objective(pp):
                logit=net.apply(pp,arrays['h'][si],arrays['c'][si],arrays['controller'][ci],arrays['action'][ci,si,ai]);n=N[ids];w=jnp.minimum(n,16.)
                return jnp.sum(w*optax.sigmoid_binary_cross_entropy(logit,Y[ids]/jnp.maximum(n,1)))/jnp.maximum(w.sum(),1)
            loss,grad=jax.value_and_grad(objective)(p);updates,st=opt.update(grad,st,p);return optax.apply_updates(p,updates),st,loss
        def step(carry,ids):
            p,st=carry;p,st,loss=jax.vmap(one)(p,st,ids);return (p,st),loss
        return jax.lax.scan(step,(parameters,states),indices)
    @jax.jit
    def predict(parameters,arrays):
        nc,ns,na=arrays['action'].shape[:3]
        h=jnp.broadcast_to(arrays['h'][None,:,None,:],(nc,ns,na,100)).reshape(-1,100);c=jnp.broadcast_to(arrays['c'][None,:,None,:],(nc,ns,na,33)).reshape(-1,33);C=jnp.broadcast_to(arrays['controller'][:,None,None,:],(nc,ns,na,24)).reshape(-1,24);a=arrays['action'].reshape(-1,10)
        return jax.vmap(lambda p:jax.nn.sigmoid(net.apply(p,h,c,C,a)).reshape(nc,ns,na))(parameters)
    return opt,block,predict
def plan():
    assert read(HERE/'CRITIC_TESTS.json')['passed'] and (HERE/'SELECTABLE_SPACE.receipt.json').exists()
    data=load_data();types=sorted(set(data['types'].tolist()));tasks=[]
    from families import family_groups
    sources=set()
    for count in (14,15,16):sources.update(itertools.combinations(types,count))
    for held,omitted in family_groups(types).items():
        source=tuple(t for t in types if t not in omitted);sources.add(source)
        for inside in source:sources.add(tuple(t for t in source if t!=inside))
    assert len(sources)==156
    for source in sorted(sources,key=lambda x:(len(x),x)):
        for mode in MODES:tasks.append(dict(source_types=source,source_key=key_for(source),mode=mode,task_uid=hashlib.sha256(json.dumps([source,mode]).encode()).hexdigest()))
    assert len(tasks)==312
    configuration=dict(utc=utc(),tasks=tasks,steps=list(range(100,2001,100)),batch=256,seeds=SEEDS,initialization='scratch_only_no_human_eta_pretraining',learning_rate=1e-3,weight_decay=1e-4,gradient_clip=5,x64=False,matmul_precision='highest',bundle_sha256=sha(HERE/'TRAINING_BUNDLE.npz'),source_sets=156,checkpoint_selection='Source-only nested ensemble NLL, then earlier step and larger ranking accuracy',lambda_grid=[.1,.25,.5,.75,1.])
    write(HERE/'CRITIC_TRAINING_PLAN.json',configuration);seal('CRITIC_TRAINING_PLAN.receipt.json',[HERE/'CRITIC_TRAINING_PLAN.json',HERE/'TRAINING_BUNDLE.receipt.json',HERE/'SELECTABLE_SPACE.receipt.json',HERE/'train_critic.py',HERE/'critic_model.py',HERE/'legacy_critic_model.py',HERE/'frozen_networks.py',HERE/'core.py',HERE/'common.py',HERE/'TRAINING_ACTIONS.npz',HERE/'CRITIC_TESTS.receipt.json'],before_real_training=True)
def fit(data,task,configuration):
    folder=folder_for(task);folder.mkdir(parents=True,exist_ok=True);assert not (folder/'COMPLETE.receipt.json').exists();write(folder/'TASK.json',dict(**task,utc=utc(),bundle_sha256=configuration['bundle_sha256']));transform=source_transform(data,task['source_types']);write(folder/'TRANSFORM.json',transform);arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    parameters=jax.tree_util.tree_map(lambda *x:jnp.stack(x),*[initialize(task['mode'],s) for s in SEEDS]);opt,block,predict=compiled(task['mode']);states=jax.vmap(opt.init)(parameters);Y,N,_=observations(data['labels']);Y=jnp.asarray(Y.reshape(-1));N=jnp.asarray(N.reshape(-1));weights=sampling_weights(data,task['source_types']);rngs=[np.random.default_rng(stable_int('crosstype1c-critic',task['source_key'],task['mode'],s)) for s in SEEDS];validation=sorted(set(data['types'])-set(task['source_types']));reference=not validation
    if reference:validation=sorted(set(data['types']))
    history=[];paths=[];last=0;start=time.monotonic()
    for step in configuration['steps']:
        indices=np.stack([r.choice(len(weights),size=(step-last,configuration['batch']),p=weights).astype(np.int32) for r in rngs],axis=1);(parameters,states),losses=block(parameters,states,jnp.asarray(indices),arrays,Y,N);probability=np.asarray(predict(parameters,arrays));summary=metrics(probability,data['labels'],data['types'],validation,data['candidate_indices']);checkpoint=folder/f'step{step}.msgpack';checkpoint.write_bytes(serialization.to_bytes(parameters));packet=folder/f'step{step}_predictions.npz';np.savez_compressed(packet,probability=probability,names=data['names'],types=data['types']);paths.extend([checkpoint,packet]);history.append(dict(step=step,training_nll_by_seed=np.asarray(losses).mean(0),validation=summary,evaluation_cohort='source_training_reference' if reference else 'omitted_types',validation_used_for_early_stop=False,elapsed_seconds=time.monotonic()-start));write(folder/'HISTORY.json',history);print(json.dumps(dict(task=task['task_uid'][:12],mode=task['mode'],sources=len(task['source_types']),step=step,elapsed=time.monotonic()-start)),flush=True);last=step
    seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/'TASK.json',folder/'TRANSFORM.json',folder/'HISTORY.json',HERE/'CRITIC_TRAINING_PLAN.receipt.json',HERE/'train_critic.py',HERE/'critic_model.py',*paths],source_only_gradient=True,source_only_preprocessing=True,initialization='scratch',backend=jax.default_backend(),matmul_precision='highest')
def run(worker):
    for path,h in read(HERE/'CRITIC_TRAINING_PLAN.receipt.json')['files'].items():assert sha(path)==h,path
    assert jax.default_backend()=='gpu';configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data();claims=HERE/'critic_training_claims';claims.mkdir(exist_ok=True)
    with jax.default_matmul_precision('highest'):
        for task in configuration['tasks']:
            if drain():break
            if (folder_for(task)/'COMPLETE.receipt.json').exists():continue
            try:fd=os.open(claims/f"{task['task_uid']}.json",os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as f:f.write(json.dumps(dict(utc=utc(),worker=worker,pid=os.getpid(),task=task)))
            fit(data,task,configuration)
    write(HERE/f'CRITIC_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),temporary_drain=drain()))
if __name__=='__main__':plan() if sys.argv[1]=='plan' else run(int(sys.argv[2]))
