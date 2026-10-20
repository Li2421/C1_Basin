"""Fixed 20M robot-step Ring curriculum; checkpoint selection is a later phase."""
from common import *
from training_env import observation,to_world
from conservative_reward import ConservativeEnv as TrainingEnv
from ippo import Actor,Critic,logprob,base_normalize,online_normalize,update_stats,decode_action,generalized_advantage
import jax
import jax.numpy as jnp
import optax
import pickle
from flax import serialization
import time
jax.config.update('jax_enable_x64',False)


def train(index):
    prereg()
    assert read(HERE/'TRAINING_ENV_GATE.json')['passed']
    assert read(HERE/'EXECUTION_REPRODUCTION.json')['group_passed']['crosstype1b']
    assert read(HERE/'CONSERVATIVE_REWARD_TESTS.json')['passed']
    assert all(d.platform=='gpu' for d in jax.devices())
    cfg=read(HERE/'IPPO_TRAINING_CONFIGS.json')[index]
    folder=HERE/'policies'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'TRAINING_COMPLETE.receipt.json').exists()
    for path,digest in cfg['input_sha256'].items():assert sha(path)==digest
    normal=read(cfg['normalization']);pool=read(cfg['training_states'])
    pp=jnp.asarray([s['positions'] for s in pool],jnp.float32)
    vv=jnp.asarray([s['velocities'] for s in pool],jnp.float32)
    gg=jnp.asarray([s['goals'] for s in pool],jnp.float32)
    env=TrainingEnv(cfg['environment'])
    actor,critic=Actor(),Critic();rootkey=jax.random.PRNGKey(cfg['seed'])
    ka,kc=jax.random.split(rootkey)
    parameters=dict(actor=actor.init(ka,jnp.zeros((1,23))),critic=critic.init(kc,jnp.zeros((1,23))))
    # 2000 rollout updates * 4 epochs * 4 minibatches.
    optimizer=optax.chain(optax.clip_by_global_norm(.5),optax.adam(optax.linear_schedule(3e-4,0.,32000)))
    optimizer_state=optimizer.init(parameters)
    stats=dict(count=jnp.asarray(1e-4),mean=jnp.zeros(23),variance=jnp.ones(23))
    write(folder/'CONFIG.json',cfg);write(folder/'NORMALIZATION.json',normal)
    seal(str(folder.relative_to(HERE)/'CONFIG_FROZEN.receipt.json'),
         [folder/'CONFIG.json',folder/'NORMALIZATION.json',HERE/'train_ippo.py',HERE/'ippo.py',HERE/'training_env.py'],before_training=True)
    def reset(key):
        ids=jax.random.randint(key,(32,),0,len(pool))
        return jax.vmap(env.reset)(pp[ids],vv[ids],gg[ids])
    batched_obs=jax.vmap(lambda st:observation(st,cfg['environment'],cfg['frame']))
    batched_world=jax.vmap(lambda act,pos:to_world(act,pos,cfg['frame']))
    batched_step=jax.vmap(env.step)
    @jax.jit
    def rollout(parameters,stats,states,key,active):
        def one(carry,t):
            states,key=carry;key,actionkey,resetkey=jax.random.split(key,3)
            obs=batched_obs(states)
            # Single-robot phase has no other robots in the observation.
            for i in range(4):
                for block,j in enumerate(k for k in range(4) if k!=i):
                    obs=obs.at[:,i,6+4*block:10+4*block].set(
                        jnp.where(active[:,j,None],obs[:,i,6+4*block:10+4*block],0.))
            base=base_normalize(obs,normal)
            x=online_normalize(base,stats)
            mean,logs=actor.apply(parameters['actor'],x)
            latent=mean+jnp.exp(logs)*jax.random.normal(actionkey,mean.shape)
            local=decode_action(latent,normal);world=batched_world(local,states.positions)
            next_state,reward,done,info=batched_step(states,world,active)
            value=critic.apply(parameters['critic'],x)
            initial=reset(resetkey)
            next_state=jax.tree.map(lambda a,b:jnp.where(done.reshape((32,)+(1,)*(a.ndim-1)),b,a),next_state,initial)
            data=dict(x=x,base=base,latent=latent,logp=logprob(latent,mean,logs),value=value,reward=reward,
                      done=done,termination=info['termination'],mask=active)
            return (next_state,key),data
        (states,key),data=jax.lax.scan(one,(states,key),jnp.arange(125))
        obs=batched_obs(states)
        # Identical observation mask for terminal value bootstrapping.
        for i in range(4):
            for block,j in enumerate(k for k in range(4) if k!=i):
                obs=obs.at[:,i,6+4*block:10+4*block].set(jnp.where(active[:,j,None],obs[:,i,6+4*block:10+4*block],0.))
        bootstrap=critic.apply(parameters['critic'],online_normalize(base_normalize(obs,normal),stats))
        advantages,returns=generalized_advantage(data['reward'],data['value'],data['done'],bootstrap)
        data['advantage']=advantages;data['return']=returns
        return states,key,data
    @jax.jit
    def update(parameters,optimizer_state,data,key):
        flat={k:v.reshape((-1,)+v.shape[3:]) for k,v in data.items()
              if k in ('x','latent','logp','mask','advantage','return')}
        # x/latent have leading time,env,robot; scalar arrays use the same flatten.
        weight=flat['mask'].astype(jnp.float32);total=jnp.maximum(weight.sum(),1)
        avg=jnp.sum(flat['advantage']*weight)/total
        variance=jnp.sum(jnp.square(flat['advantage']-avg)*weight)/total
        flat['advantage']=(flat['advantage']-avg)/jnp.sqrt(variance+1e-8)
        size=flat['mask'].shape[0]
        permutations=jax.vmap(lambda k:jax.random.permutation(k,size))(jax.random.split(key,4)).reshape(16,-1)
        def minibatch(carry,indices):
            parameters,st=carry;batch={k:v[indices] for k,v in flat.items()}
            def loss(p):
                mean,logs=actor.apply(p['actor'],batch['x']);newlog=logprob(batch['latent'],mean,logs)
                ratio=jnp.exp(newlog-batch['logp']);adv=batch['advantage'];mask=batch['mask']
                policy=-jnp.minimum(ratio*adv,jnp.clip(ratio,.8,1.2)*adv)
                value=critic.apply(p['critic'],batch['x'])
                entropy=jnp.sum(logs+.5*jnp.log(2*jnp.pi*jnp.e),axis=-1)
                losses=policy+.5*jnp.square(value-batch['return'])-.005*entropy
                return jnp.sum(losses*mask)/jnp.maximum(mask.sum(),1)
            value,grad=jax.value_and_grad(loss)(parameters);delta,st=optimizer.update(grad,st,parameters)
            return (optax.apply_updates(parameters,delta),st),value
        return jax.lax.scan(minibatch,(parameters,optimizer_state),permutations)
    count=0;history=[];before=time.monotonic();key=rootkey
    for phase in ('single','joint'):
        key,resetkey,activekey=jax.random.split(key,3);states=reset(resetkey)
        active=jax.nn.one_hot(jax.random.randint(activekey,(32,),0,4),4,dtype=bool) if phase=='single' else jnp.ones((32,4),bool)
        stride=4000 if phase=='single' else 16000
        for iteration in range(1000):
            states,key,data=rollout(parameters,stats,states,key,active)
            key,updatekey=jax.random.split(key)
            (parameters,optimizer_state),losses=update(parameters,optimizer_state,data,updatekey)
            stats=update_stats(stats,data['base'],data['mask']);count+=stride
            if (iteration+1)%25==0:
                term=np.asarray(data['termination']);done=term!=0
                history.append(dict(robot_steps=count,phase=phase,mean_loss=float(losses.mean()),
                    mean_reward=float(jnp.sum(data['reward'])/jnp.maximum(data['mask'].sum(),1)),
                    completed_episodes=int(done.sum()),successes=int((term==2).sum()),
                    collisions=int((term==1).sum()),timeouts=int((term==3).sum())))
                write(folder/'TRAINING_HISTORY.partial.json',history)
            if count%2000000==0:
                checkpoint=folder/f'step{count}.msgpack'
                checkpoint.write_bytes(serialization.to_bytes(dict(parameters=parameters,statistics=stats)))
                seal(str(checkpoint.relative_to(HERE))+'.receipt.json',[checkpoint,folder/'CONFIG.json',folder/'NORMALIZATION.json'],
                     robot_steps=count,no_safety_projection_in_training=True)
                print(cfg['name'],count,'robot steps',flush=True)
    assert count==20000000
    write(folder/'TRAINING_HISTORY.json',history)
    write(folder/'TRAINING_SUMMARY.json',dict(robot_steps=count,elapsed_seconds=time.monotonic()-before,
         checkpoint_selection='PENDING original-environment nominal-only evaluation on frozen 32 training-held states'))
    seal(str(folder.relative_to(HERE)/'TRAINING_COMPLETE.receipt.json'),
         [folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'TRAINING_HISTORY.json',*sorted(folder.glob('step*.msgpack'))],
         locked_correction_outcomes_read=False)

if __name__=='__main__':train(int(sys.argv[1]))
