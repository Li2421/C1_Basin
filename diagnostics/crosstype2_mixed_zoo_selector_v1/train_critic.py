"""GPU three-seed fits cached by the exact set of source types."""
from common import *
from critic_model import network,initialize,source_transform,transform_arrays,observations,sampling_weights,metrics,SEEDS,MODES
from frozen_networks import stable_int
from flax import serialization
import jax,jax.numpy as jnp
import optax
from functools import lru_cache
import itertools,os,time,fcntl
jax.config.update('jax_enable_x64',False)


def key_for(source_types):return hashlib.sha256(json.dumps(sorted(source_types)).encode()).hexdigest()[:20]


def load_data(path):
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    assert len(set(data['types']))>=2
    assert data['labels'].shape==(len(data['names']),48,16,2) and not np.any(data['labels'][...,0]==-2)
    assert data['features'].shape==(len(data['names']),34)
    assert data['h'].shape==(48,100) and data['c'].shape==(48,33) and data['action'].shape==(48,16,16)
    return data


@lru_cache(None)
def compiled(mode):
    net=network(mode)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4))
    @jax.jit
    def block(parameters,states,indices,arrays,Y,N):
        def one(p,st,ids):
            ci=ids//(48*16);si=(ids//16)%48;ai=ids%16
            def loss(pp):
                logits=net.apply(pp,arrays['h'][si],arrays['c'][si],arrays['controller'][ci],arrays['action'][si,ai])
                y=Y[ids];n=N[ids];w=jnp.minimum(n,16.)
                return jnp.sum(w*optax.sigmoid_binary_cross_entropy(logits,y/jnp.maximum(n,1.)))/jnp.maximum(w.sum(),1.)
            value,grad=jax.value_and_grad(loss)(p)
            updates,st=opt.update(grad,st,p)
            return optax.apply_updates(p,updates),st,value
        def step(carry,ids):
            p,st=carry;p,st,value=jax.vmap(one)(p,st,ids)
            return (p,st),value
        return jax.lax.scan(step,(parameters,states),indices)
    @jax.jit
    def predict(parameters,arrays):
        nc=arrays['controller'].shape[0];ns=arrays['h'].shape[0];na=arrays['action'].shape[1]
        hh=jnp.broadcast_to(arrays['h'][None,:,None,:],(nc,ns,na,100)).reshape(-1,100)
        cc=jnp.broadcast_to(arrays['c'][None,:,None,:],(nc,ns,na,33)).reshape(-1,33)
        C=jnp.broadcast_to(arrays['controller'][:,None,None,:],(nc,ns,na,34)).reshape(-1,34)
        a=jnp.broadcast_to(arrays['action'][None],(nc,ns,na,16)).reshape(-1,16)
        def one(p):return jax.nn.sigmoid(net.apply(p,hh,cc,C,a)).reshape(nc,ns,na)
        return jax.vmap(one)(parameters)
    return opt,block,predict


def plan():
    assert read(HERE/'CRITIC_MODEL_TESTS.json')['passed']
    assert read(HERE/'CRITIC_TRAINING_TESTS.json')['passed']
    assert read(HERE/'CONTROLLER_INNER_TESTS.json')['passed']
    assert (HERE/'SELECTABLE_SPACE.receipt.json').exists(),'Report selectable space before fitting any real critic'
    target=HERE/'CRITIC_TRAINING_PLAN.json';assert not target.exists()
    data=load_data(HERE/'TRAINING_BUNDLE.npz');types=sorted(set(data['types'].tolist()))
    source_sets=set()
    for count in (len(types)-2,len(types)-1,len(types)):
        source_sets.update(itertools.combinations(types,count))
    families={'learned':{'Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo','ring_ippo_conservative'},'optimization':{'distributed_shooting','centralized_sampling','cross_entropy'},'search':{'prioritized_astar'}}
    for family in families.values():
        source=tuple(t for t in types if t not in family)
        if not source:continue
        source_sets.add(source)
        if len(source)>1:
            source_sets.update(tuple(t for t in source if t!=held) for held in source)
    source_sets=sorted(source_sets,key=lambda x:(len(x),x))
    tasks=[]
    for source in source_sets:
        for mode in MODES:
            for init in ('scratch','warm'):
                tasks.append(dict(source_types=source,source_key=key_for(source),mode=mode,initialization=init,
                    task_uid=hashlib.sha256(json.dumps([source,mode,init]).encode()).hexdigest()))
    record=dict(utc=utc(),bundle_path=str(HERE/'TRAINING_BUNDLE.npz'),bundle_sha256=sha(HERE/'TRAINING_BUNDLE.npz'),
        seeds=SEEDS,steps=list(range(100,2001,100)),batch=256,learning_rate=1e-3,weight_decay=1e-4,gradient_clip=5.,
        x64=False,matmul_precision='highest',tasks=tasks,unique_source_type_sets=len(source_sets),
        nested_reuse_rule='A cached exact source-set fit may validate each omitted type; outer A selects using only omitted B and never uses A labels. No fit is early-stopped using omitted-type labels.',
        selector='Choose initialization/checkpoint using type-equal inner ensemble NLL; tie earlier step, higher within-state ranking, scratch',
        original_weights_immutable=True)
    write(target,record)
    seal('CRITIC_TRAINING_PLAN.receipt.json',[target,HERE/'train_critic.py',HERE/'critic_model.py',HERE/'CRITIC_MODEL_TESTS.receipt.json',HERE/'CONTROLLER_INNER_TESTS.receipt.json',HERE/'CRITIC_PRECISION_DESIGN.receipt.json',HERE/'TRAINING_BUNDLE.receipt.json'],before_real_critic_training=True)
    print('Sealed',len(tasks),'three-seed fits; exact source-type sets:',len(source_sets),flush=True)


def folder_for(task):return HERE/'critic_fits'/f"{task['mode']}_{task['initialization']}"/task['source_key']


def fit(data,task,configuration):
    if task.get('held_controller') is not None:
        data={**data,'types':data['types'].astype('U160').copy()}
        mask=data['names']==task['held_controller'];assert int(mask.sum())==1
        assert str(data['types'][mask][0]) in task['source_types']
        data['types'][mask]='__held_controller__'+task['held_controller']
    folder=folder_for(task);folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'COMPLETE.receipt.json').exists()
    write(folder/'TASK.json',dict(**task,utc=utc(),bundle_sha256=configuration['bundle_sha256']))
    transform=source_transform(data,task['source_types']);write(folder/'TRANSFORM.json',transform)
    arrays={k:jnp.asarray(v) for k,v in transform_arrays(data,transform).items()}
    params=[initialize(task['mode'],seed,task['initialization'],transform) for seed in SEEDS]
    parameters=jax.tree_util.tree_map(lambda *a:jnp.stack(a),*params)
    opt,block,predict=compiled(task['mode']);states=jax.vmap(opt.init)(parameters)
    Y,N,_=observations(data['labels']);Y=jnp.asarray(Y.reshape(-1));N=jnp.asarray(N.reshape(-1))
    weights=sampling_weights(data,task['source_types'])
    rngs=[np.random.default_rng(stable_int('crosstype2-critic',task['source_key'],task['mode'],seed)) for seed in SEEDS]
    validation=sorted(set(data['types'].tolist())-set(task['source_types']))
    evaluation_cohort='omitted_types' if validation else 'source_training_reference'
    if not validation:validation=sorted(set(data['types'].tolist()))
    history=[];paths=[];last=0;started=time.monotonic()
    for step in configuration['steps']:
        indices=np.stack([rng.choice(len(weights),size=(step-last,configuration['batch']),p=weights).astype(np.int32) for rng in rngs],axis=1)
        (parameters,states),losses=block(parameters,states,jnp.asarray(indices),arrays,Y,N)
        probabilities=np.asarray(predict(parameters,arrays));summary=metrics(probabilities,data['labels'],data['types'],validation)
        checkpoint=folder/f'step{step}.msgpack';checkpoint.write_bytes(serialization.to_bytes(parameters));paths.append(checkpoint)
        # Keep all curves/choices, not only a per-fit "best" checkpoint.
        pred_path=folder/f'step{step}_predictions.npz';np.savez_compressed(pred_path,probability=probabilities,names=data['names'],types=data['types']);paths.append(pred_path)
        history.append(dict(step=step,training_nll_by_seed=np.asarray(losses).mean(0),validation=summary,evaluation_cohort=evaluation_cohort,
            checkpoint_sha256=sha(checkpoint),probability_sha256=sha(pred_path),elapsed_seconds=time.monotonic()-started,
            validation_used_for_early_stop=False))
        write(folder/'HISTORY.json',history)
        print(json.dumps(dict(task=task['task_uid'][:12],mode=task['mode'],initialization=task['initialization'],source_types=len(task['source_types']),step=step,validation=summary)),flush=True)
        last=step
    seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/'TASK.json',folder/'TRANSFORM.json',folder/'HISTORY.json',
        HERE/configuration.get('plan_receipt','CRITIC_TRAINING_PLAN.receipt.json'),HERE/'train_critic.py',HERE/'critic_model.py',*paths],seeds=SEEDS,
        omitted_types_do_not_affect_fit=True,backend=jax.default_backend(),x64=False,matmul_precision='highest')


def run(worker):
    for path,digest in read(HERE/'CRITIC_TRAINING_PLAN.receipt.json')['files'].items():assert sha(path)==digest,path
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json')
    assert sha(configuration['bundle_path'])==configuration['bundle_sha256']
    assert jax.default_backend()=='gpu','Real critic training requires a Slurm GPU shard'
    data=load_data(configuration['bundle_path']);queue=HERE/'critic_training_claims';queue.mkdir(exist_ok=True)
    with jax.default_matmul_precision('highest'):
        for task in configuration['tasks']:
            folder=folder_for(task)
            if (folder/'COMPLETE.receipt.json').exists():continue
            claim=queue/f"{task['task_uid']}.json"
            try:fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
            except FileExistsError:continue
            with os.fdopen(fd,'w') as stream:stream.write(json.dumps(dict(utc=utc(),worker=worker,pid=os.getpid(),task=task)))
            fit(data,task,configuration)
    write(HERE/f'CRITIC_TRAINING_WORKER_{worker}_COMPLETE.json',dict(utc=utc(),worker=worker))


if __name__=='__main__':
    if sys.argv[1]=='plan':plan()
    else:run(int(sys.argv[2]))
