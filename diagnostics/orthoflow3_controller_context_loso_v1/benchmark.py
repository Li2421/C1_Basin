"""Measured CPU end-to-end one-shot selection; no task success rollout."""
import os,time,platform,csv
import numpy as np
import jax,jax.numpy as jnp
from flax import serialization
from .train import OUT,ROOT,FOLDS,Critic,load,dump,gather
from .evaluate import normalized,TARGET
from .context import Runtime
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
def stats(t):
 a=1000*np.asarray(t);return {'mean_ms':float(a.mean()),'p50_ms':float(np.median(a)),'p95_ms':float(np.quantile(a,.95)),'p99_ms':float(np.quantile(a,.99))}
def main():
 frozen=load(OUT/'models_frozen.json');pred=load(OUT/'target_predictions.json');allrows=[];checks=[]
 for fold,scene in FOLDS.items():
  rt=Runtime(scene);physical=load(OUT/f'target_context_{fold}.json')['physical'][0];eta=np.array(load(TARGET/fold/'manifest.json')[0]['eta'],float)
  entities=rep.batch([rep.entities(physical)]);x=gather(entities,np.zeros(16,int));raw=np.load(OUT/f'target_context_{fold}.npz')
  for kind in ('C1_full','C3_full'):
   norm=load(OUT/fold/kind/'normalization.json');run=frozen['folds'][fold][kind]['selected'];model=Critic()
   init=model.init(jax.random.PRNGKey(0),gather(entities,[0]),jnp.zeros((1,3),jnp.float32),jnp.zeros((1,11),jnp.float32))
   par=serialization.from_bytes(init,open(run['checkpoint'],'rb').read());e=jnp.asarray((eta-np.array(norm['eta_center']))/norm['eta_radius'],jnp.float32)
   fn=jax.jit(lambda xx,cc:jax.nn.sigmoid(model.apply(par,xx,e,cc)))
   cr=np.repeat(raw['C1'][:1],16,axis=0) if kind.startswith('C1') else raw['C3'][:16]
   c=jnp.asarray(normalized(cr,norm));p=np.asarray(fn(x,c).block_until_ready());expected=np.asarray(pred['folds'][fold]['scores'][kind][0]);err=float(abs(p-expected).max());assert err<2e-5,(fold,kind,err)
   checks.append({'fold':fold,'kind':kind,'CPU_vs_GPU_max_abs':err})
   timings={k:[] for k in ['encoding','batched_K16_score','argmax','context_K16','end_to_end']}
   for t in range(1020):
    start=time.perf_counter();xx=rep.batch([rep.entities(physical)]);xx=gather(xx,np.zeros(16,int));jax.block_until_ready(xx);mid=time.perf_counter()
    pr=fn(xx,c).block_until_ready();end=time.perf_counter();np.argmax(np.asarray(pr));done=time.perf_counter()
    if t>=20:
     timings['encoding'].append(mid-start);timings['batched_K16_score'].append(end-mid);timings['argmax'].append(done-end)
   for t in range(105):
    start=time.perf_counter()
    vals=np.repeat([rt.probe(physical,np.zeros(3))],16,axis=0) if kind.startswith('C1') else np.array([rt.probe(physical,a) for a in eta])
    cnew=jnp.asarray(normalized(np.c_[vals,np.ones(16)],norm));mid=time.perf_counter()
    xx=gather(rep.batch([rep.entities(physical)]),np.zeros(16,int));pr=fn(xx,cnew).block_until_ready();np.argmax(np.asarray(pr));end=time.perf_counter()
    if t>=5:timings['context_K16'].append(mid-start);timings['end_to_end'].append(end-start)
   for part,times in timings.items():
    st=stats(times);allrows.append({'fold':fold,'kind':kind,'part':part,'repeats':len(times),**st,'p50_theoretical_Hz':1000/st['p50_ms']})
   dump(OUT/'latency_progress.json',allrows)
 with (OUT/'deployment_latency.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 dump(OUT/'deployment_benchmark.json',{'hardware':platform.uname()._asdict(),'jax_devices':[str(d) for d in jax.devices()],
   'CPU_affinity':sorted(os.sched_getaffinity(0)),'precision':'critic float32; physical projection float64',
   'context':'C1 once per state; C3 16 serial short probes, batched neural scoring; no claim that probes themselves are vectorized',
   'execution':'current one-shot eta selection; frequencies describe hypothetical updates, not low-level control frequency',
   'sync':'block_until_ready on each measured inference; CPU/GPU inference parity checked against frozen GPU scores',
   'checks':checks,'new_full_continuations':0,'context_measurement_steps':4*105*3*(1+16),
   'scope':'first frozen state per scenario, repeated fixed inputs; p95/p99 are runtime jitter, not a cohort-wide latency bound; excludes frozen generator proposal generation'})
if __name__=='__main__':main()
