"""Nominal only (no safety projection, no correction): same env, controller, RNG as runtime_new.run; action = bounded controller output."""
import os,sys,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R
_W={}
def _init():
 R._init('scenes_all.json');_W.update(R._W)
def run_nominal(c):
 import numpy as np,jax
 import runtime_new as RN
 env=RN.SwapEnv(c['state'],RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config;actor=RN.Nominal(c['controller_payload']['nominal'])
 rng_uid=c.get('rng_state_uid',c['state_uid']);root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8],16)),c['seed'])
 termination='timeout'
 for t in range(cfg.max_steps):
  ref=RN.bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64);ref[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
  _,done,info=env.step(ref);termination=info['termination']
  if done:break
 return dict(success=int(termination=='success' and not env.collided),outcome=termination)
def _run(t):
 v,ci,si=t;sc=_W['scenes'][v]
 c=_W['make_new'](sc,_W['controllers'][ci],sc['states'][si],_W['actions'][R.KEYS.index('safety')],0);r=run_nominal(c)
 return dict(scene=v,key='nominal',controller=ci,state=si,**r)
if __name__=='__main__':
 TE=json.load(open('FINAL_FROZEN.json'))['test'];out='test_nominal.jsonl'
 done=set()
 if os.path.exists(out):
  for l in open(out):d=json.loads(l);done.add((d['scene'],d['controller'],d['state']))
 todo=[(v,ci,si) for v in TE for ci in range(3) for si in range(8,24) if (v,ci,si) not in done]
 print(len(todo),'todo',flush=True);t0=time.time()
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=_init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):f.write(json.dumps(fu.result())+'\n');f.flush()
 print('done',round(time.time()-t0),flush=True)
