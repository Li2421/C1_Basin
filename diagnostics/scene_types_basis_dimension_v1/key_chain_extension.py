"""Extra full-horizon RNG-chain coverage, zero environment steps."""
from core import *
import importlib.util,time
import jax.numpy as jnp

def main():
    spec=importlib.util.spec_from_file_location('extra_threefry',HERE/'accel_inputs/np_threefry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    batch=RN.jax.jit(RN.jax.vmap(lambda k,d:RN.jax.random.fold_in(k,d),in_axes=(None,0)))
    by_n={};records=[];comparisons=0;t0=time.monotonic()
    for e in read(HERE/'SCENES_FROZEN.json')['entries']:
        if 'scene'not in e:continue
        s=e['scene'];n=s['config']['num_agents'];steps=s['config']['max_steps']
        if n not in by_n:
            by_n[n]=RN.jax.jit(RN.jax.vmap(lambda k:RN.jax.vmap(lambda i:RN.jax.random.fold_in(k,i))(jnp.arange(n,dtype=jnp.uint32))))
        for state in s['states'][:8]:
            physical={k:state[k]for k in('positions','velocities','goals')};uid=db.uid('rng_physical',physical);h=int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)
            base=RN.jax.random.PRNGKey(2026102004);refroot=RN.jax.random.fold_in(RN.jax.random.fold_in(base,h),0);root=m.fold_in(m.fold_in(np.asarray(base),h),0)
            assert np.array_equal(root,np.asarray(refroot));comparisons+=2
            keys=np.asarray(batch(refroot,jnp.arange(steps,dtype=jnp.uint32)));children=np.asarray(by_n[n](jnp.asarray(keys)))
            for t in range(steps):
                key=m.fold_in(root,t);assert np.array_equal(key,keys[t]);comparisons+=1
                for robot in range(n):assert np.array_equal(m.fold_in(key,robot),children[t,robot]);comparisons+=1
            records.append(dict(type_id=e['type_id'],state_index=state['index'],steps=steps,robots=n,root_uid=uid,jax_chain_sha256=hashlib.sha256(keys.tobytes()+children.tobytes()).hexdigest()))
        print('extended key chain',e['type_id'],'comparisons',comparisons,flush=True)
    dump(HERE/'ACCELERATION_KEY_CHAIN_EXTENSION.json',dict(passed=True,comparisons=comparisons,physical_states=len(records),all_steps_and_all_robot_folds=True,seed=0,records=records,seconds=time.monotonic()-t0,note='No rollout, no control or test outcome observed; every frozen search/holdout physical state, full declared horizon. Reserved states8..11 remain unused.'))

if __name__=='__main__':main()
