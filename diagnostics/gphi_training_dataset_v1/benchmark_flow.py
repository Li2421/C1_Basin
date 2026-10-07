"""One-off FlowBC inference throughput benchmark; no physical rollout."""
import argparse, json, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from single_integrator.evaluate import load_policy

parser=argparse.ArgumentParser();parser.add_argument('--device',choices=('cpu','gpu'),required=True);parser.add_argument('--batch',type=int,default=64);parser.add_argument('--iterations',type=int,default=300);args=parser.parse_args()
jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name',args.device)
checkpoint=Path('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
policy,_=load_policy(checkpoint)
single=lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]
fn=jax.jit(jax.vmap(single));obs=jnp.zeros((args.batch,2,10),jnp.float32);keys=jax.random.split(jax.random.PRNGKey(901),args.batch)
np.asarray(fn(obs,keys));started=time.monotonic()
for _ in range(args.iterations):np.asarray(fn(obs,keys))
elapsed=time.monotonic()-started
print(json.dumps({'requested':args.device,'devices':[str(x) for x in jax.devices()],'batch':args.batch,'iterations':args.iterations,'elapsed_s':elapsed,'samples_per_s':args.batch*args.iterations/elapsed}))
