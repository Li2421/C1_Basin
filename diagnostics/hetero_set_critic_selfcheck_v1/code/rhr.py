"""Safety + historical fixed right-hand rule (crosstype3 supplement: progress window 40 steps, threshold 0.05, tangential speed 0.35),
trigger generalized from 4 to N robots. Same env, controller, RNG and safety filters as runtime_new.run with eta = 0;
active robots get speed*(q_y,-q_x)/|q| added to the safety action, then the same per-robot safety projection."""
import os,sys,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import run as R,run8
RULE=dict(steps=40,threshold=.05,speed=.35)
def run_rhr(c,speed=RULE['speed'],variant='centre'):
 import numpy as np,jax
 import runtime_new as RN
 env=RN.SwapEnv(c['state'],RN.LiteratureConfig(**c['scenario_payload']['config']));cfg=env.config;actor=RN.Nominal(c['controller_payload']['nominal'])
 rng_uid=c.get('rng_state_uid',c['state_uid']);root=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(2026102004),int(hashlib.sha256(rng_uid.encode()).hexdigest()[:8],16)),c['seed'])
 termination='timeout';hist={};fired=0
 for t in range(cfg.max_steps):
  ref=RN.bound(actor.action(env,jax.random.fold_in(root,t)),cfg.max_speed).astype(np.float64);ref[np.linalg.norm(env.goals-env.positions,axis=-1)<=cfg.goal_tolerance]=0
  snap=env.snapshot();filters=[RN.SwapSafety(snap,i) for i in range(env.num_agents)]
  v=np.concatenate([f.project(ref[i:i+1])[0] for i,f in enumerate(filters)])
  a=np.concatenate([f.project(v[i:i+1])[0] for i,f in enumerate(filters)])   # eta = 0 path of runtime_new
  dist=np.linalg.norm(env.goals-env.positions,axis=1);hist[t]=dist.copy();hist.pop(t-RULE['steps']-1,None)
  active=(hist[t-RULE['steps']]-dist<RULE['threshold'])&(dist>cfg.goal_tolerance) if t>=RULE['steps'] else np.zeros(env.num_agents,bool)
  if speed and active.any():
   target=a.copy();q=env.positions
   for i in np.flatnonzero(active):
    u=q[i] if variant=='centre' else env.goals[i]-q[i]   # centre: clockwise around scene centre; goal: right of own goal direction
    target[i]+=speed*np.array([u[1],-u[0]])/max(np.linalg.norm(u),1e-30)
   a=np.concatenate([filters[i].project(target[i:i+1])[0] if active[i] else a[i:i+1] for i in range(env.num_agents)]);fired+=1
  _,done,info=env.step(a);termination=info['termination']
  if done:break
 return dict(success=int(termination=='success' and not env.collided),outcome=termination,steps_with_rule=fired)
def _run(t):
 v,ci,si,speed=t;W=run8._W;sc=W['scenes'][v]
 c=W['make_new'](sc,W['controllers'][ci],sc['states'][si],W['actions'][R.KEYS.index('safety')],0);r=run_rhr(c,speed)
 return dict(scene=v,key='rhr' if speed else 'rhr_speed0_check',controller=ci,state=si,**r)
if __name__=='__main__':
 TE=json.load(open('FINAL_FROZEN.json'))['test'];mode=sys.argv[1]
 if mode=='check':tasks=[(v,ci,si,0.) for v in TE for ci in range(8) for si in (8,9)];out='test_rhr_check.jsonl'
 else:tasks=[(v,ci,si,RULE['speed']) for v in TE for ci in range(8) for si in range(8,24)];out='test_rhr.jsonl'
 done=set()
 if os.path.exists(out):done={(d['scene'],d['controller'],d['state']) for d in map(json.loads,open(out))}
 todo=[t for t in tasks if t[:3] not in done];todo.sort(key=lambda t:0 if t[0].startswith('hs_obst') else 1);print(len(todo),'todo',flush=True);t0=time.time()
 with ProcessPoolExecutor(18,mp_context=mp.get_context('spawn'),initializer=run8._init) as pool,open(out,'a') as f:
  for fu in as_completed([pool.submit(_run,t) for t in todo]):f.write(json.dumps(fu.result())+'\n');f.flush()
 print('done',round(time.time()-t0),flush=True)
