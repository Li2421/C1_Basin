"""Second-priority pilots; reuse successful CBS training-state trajectories."""
from experiment import *
from pilot_policies import Policy,observation,PilotEnv
from states import config
from learning_design import selected_states
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization

def data(scene):
    from coordinator import done,SCENES
    assert all(done('fixed_'+s) and done('whitebox_'+s) and done('nominal_'+s) for s in SCENES),'Finish original controller and white-box work in all scenes before BC pilots'
    states=selected_states(scene,'train');rows=read(HERE/'CONTROLLER_MANIFEST.json');teachers=[r for r in rows if r['type_code']=='conflict_based_search'];raw={}
    for p in (HERE/f'safety_{scene}/raw').glob('*.json'):
        r=read(p);c=r['case'];name=c['controller_payload']['nominal']['controller']
        if name in {t['controller'] for t in teachers} and c['seed']==0:raw[(name,c['state']['index'],c['state']['N'])]=(p,r)
    X=[[],[]];Y=[[],[]];sources=[]
    for s in states:
        found=next((raw[(t['controller'],s['index'],s['N'])] for t in teachers if (t['controller'],s['index'],s['N']) in raw and raw[(t['controller'],s['index'],s['N'])][1]['success']),None)
        if found is None:continue
        path,r=found;assert sha(r['trace_path'])==r['trace_sha256'];z=np.load(r['trace_path']);env=PilotEnv(s);split=int(s['index']%4==0)
        for t in range(len(z['actions'])):
            env.positions=z['positions'][t];env.velocities=z['velocities'][t];obs,_=observation(env,s['N']);X[split].append(obs);Y[split].append(z['actions'][t].reshape(-1))
        sources.append(dict(state_index=s['index'],N=s['N'],split='dev' if split else 'train',source=str(path),sha256=sha(path),trace_sha256=r['trace_sha256']))
    if not X[0] or not X[1]:raise ValueError('No successful CBS demonstrations in both fixed internal train/dev state partitions')
    file=HERE/f'PILOT_DEMONSTRATIONS_{scene}.npz';np.savez_compressed(file,train_obs=np.asarray(X[0],np.float32),train_actions=np.asarray(Y[0],np.float32),dev_obs=np.asarray(X[1],np.float32),dev_actions=np.asarray(Y[1],np.float32))
    write(HERE/f'PILOT_DEMONSTRATION_SOURCES_{scene}.json',sources);seal(f'PILOT_DEMONSTRATIONS_{scene}.receipt.json',[file,HERE/f'PILOT_DEMONSTRATION_SOURCES_{scene}.json'],new_expert_rollouts=0,evaluation_states_used=False)

def train(scene):
    assert jax.default_backend()=='gpu';prereg();dataset=HERE/f'PILOT_DEMONSTRATIONS_{scene}.npz'
    with np.load(dataset) as z:raw={k:z[k] for k in z.files}
    n=raw['train_actions'].shape[1]//2;normal=dict(obs_mean=raw['train_obs'].mean(0),obs_scale=np.maximum(raw['train_obs'].std(0),1e-3),act_mean=raw['train_actions'].mean(0),act_scale=np.maximum(raw['train_actions'].std(0),1e-3),max_speed=config(scene,n).max_speed)
    norm=lambda a,m,s:(a-np.asarray(normal[m]))/np.asarray(normal[s]);X=jnp.asarray(norm(raw['train_obs'],'obs_mean','obs_scale'));Y=jnp.asarray(norm(raw['train_actions'],'act_mean','act_scale'))
    for family in ('deterministic','mixture'):
        for seed,validation_seed in ((2026103301,2026104301),(2026103302,2026104302)):
            folder=HERE/'pilot_policies'/scene/f'{family}_{seed}';folder.mkdir(parents=True,exist_ok=True)
            if (folder/'COMPLETE.receipt.json').exists():continue
            cfg=dict(scene=scene,N=n,family=family,seed=seed,validation_seed=validation_seed,steps=5000,batch=256,learning_rate=.0003,architecture='Original three ReLU hidden layers of width256; five diagonal Gaussian components for mixture',observation_frame='global',new_scene_observation_dimension=n*(4*n+7),original_recipe='first matching family in blackbox_families_v1/TRAINING_CONFIGS.json; second run changes only seed',demo_sha256=sha(dataset),source_hashes={str(ROOT/p):sha(ROOT/p) for p in ('diagnostics/blackbox_families_v1/TRAINING_CONFIGS.json','diagnostics/blackbox_families_v1/train_policies.py','diagnostics/blackbox_families_v1/learned.py','ring_exchange/environment.py')})
            write(folder/'CONFIG.json',cfg);write(folder/'NORMALIZATION.json',normal);seal(str(folder.relative_to(HERE)/'CONFIG_FROZEN.receipt.json'),[folder/'CONFIG.json',folder/'NORMALIZATION.json',dataset,HERE/'pilot_training.py',HERE/'pilot_policies.py'],before_training=True)
            rng=np.random.default_rng(validation_seed);vi=rng.integers(len(raw['dev_obs']),size=2048);np.save(folder/'DEV_INDICES.npy',vi);XV=jnp.asarray(norm(raw['dev_obs'][vi],'obs_mean','obs_scale'));YV=jnp.asarray(norm(raw['dev_actions'][vi],'act_mean','act_scale'));net=Policy(family,n);pa=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,X.shape[1])));opt=optax.adam(.0003);st=opt.init(pa)
            def loss(p,x,y):
                if family=='deterministic':return jnp.mean(jnp.square(net.apply(p,x)-y))
                logits,means,logs=net.apply(p,x);component=-jnp.sum(.5*jnp.square((y[:,None]-means)*jnp.exp(-logs))+logs+.5*jnp.log(2*jnp.pi),-1)
                return -jnp.mean(jax.scipy.special.logsumexp(jax.nn.log_softmax(logits,-1)+component,-1))
            @jax.jit
            def block(pa,st,ids):
                def step(carry,idx):
                    p,s=carry;grad=jax.grad(loss)(p,X[idx],Y[idx]);delta,s=opt.update(grad,s,p);return (optax.apply_updates(p,delta),s),None
                return jax.lax.scan(step,(pa,st),ids)[0]
            validate=jax.jit(lambda p:loss(p,XV,YV));best=float(validate(pa));history=[dict(step=0,dev_loss=best)];(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa));rng=np.random.default_rng(seed)
            for target in range(250,5001,250):
                ids=rng.integers(len(X),size=(250,256));pa,st=block(pa,st,jnp.asarray(ids));value=float(validate(pa));assert np.isfinite(value);history.append(dict(step=target,dev_loss=value))
                if value<best:best=value;(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
            (folder/'final.msgpack').write_bytes(serialization.to_bytes(pa));write(folder/'HISTORY.json',history);seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'HISTORY.json',folder/'best.msgpack',folder/'final.msgpack',folder/'DEV_INDICES.npy'],training_states_only=True,source_recipe_steps=5000);print(scene,family,seed,best,flush=True)

if __name__=='__main__':data(sys.argv[2]) if sys.argv[1]=='data' else train(sys.argv[2])
