"""Implementation integrity only: compare frozen kernels, no new scientific fit."""
import inspect, hashlib, json
import jax
import numpy as np
from flax import serialization
from diagnostics.controller_cached_union_v1 import learn as old
from diagnostics.controller_native_exposure_v1 import study as new


def setup(module,index):
    src=inspect.getsource(module.fit)
    src=src[:src.index('    rng=np.random.default_rng')]
    src=src.replace("    if (out/'complete.json').exists():return\n",'')
    src+='    return locals()\n'
    scope=dict(module.__dict__);scope['write']=lambda *a,**k:None;scope['freeze']=lambda *a,**k:None
    exec(src,scope)
    return scope['fit'](index)


def difference(a,b):
    aa=jax.tree_util.tree_leaves(a);bb=jax.tree_util.tree_leaves(b)
    return dict(max_abs=max(float(np.max(np.abs(np.asarray(x)-np.asarray(y)))) for x,y in zip(aa,bb)),
        differing_arrays=sum(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(aa,bb)))

result=[]
for seed in new.SEEDS:
    a=setup(old,next(i for i,j in enumerate(old.JOBS) if j==dict(fold='joint',arm='union_native_slots',seed=seed)))
    b=setup(new,new.JOBS.index(dict(arm='joint_replay',seed=seed)))
    row=dict(seed=seed,input={k:difference(a[k],b[k]) for k in ['x','c','e','s','f']},initial=difference(a['params'],b['par']))
    draws=np.load(new.HERE/f'draws_seed{seed}.npz')['joint'][:200]
    for steps in [1,200]:
        d=jax.numpy.asarray(draws[:steps]);m=jax.numpy.ones(d.shape[:2],jax.numpy.float32)
        (pa,oa),la=a['block'](a['params'],a['opt'],d)
        (pb,ob),lb=b['block'](b['par'],b['opt'],d,m)
        row[str(steps)]=dict(params=difference(pa,pb),loss=difference(la,lb),old_mean_loss=float(np.mean(la)),new_mean_loss=float(np.mean(lb)))
        if steps==200:
            ref=json.load(open(old.HERE/'fits/joint/union_native_slots'/f'seed{seed}/history.json'))[0]['training_nll']
            row[str(steps)]['old_recorded_mean_loss']=ref
    result.append(row);print(json.dumps(row),flush=True)
new.write(new.HERE/'replay_kernel_probe.json',dict(results=result,new_rollouts=0,no_test_used=True))
