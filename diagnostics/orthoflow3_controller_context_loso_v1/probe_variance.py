"""Source controller-swap diagnostic: response signal versus probe RNG noise.

No model/feature changes. Four additional short-response noise streams, not Q
labels. This cannot be used to select a new target-tested architecture.
"""
import numpy as np
import jax
from .context import OUT,OLD,SWAP,Runtime,load,dump,cached,PROTOCOL
def main():
 pairs=load(SWAP/'pair_manifest.json');rt=[Runtime('toy_giveway',bool(i)) for i in (0,1)]
 template=next(s['physical'] for s in load(OLD/'states.json') if s['scenario']=='toy_giveway');records=[]
 seeds=[2026100322,2026100323,2026100324,2026100325]
 for r in pairs:
  env=rt[0].make();env.reset(np.asarray(r['initial_positions']));flow=rt[0].flow(env,jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42),r['rollout_id']),0))
  p=dict(template,positions=env.positions.tolist(),velocities=env.velocities.tolist(),goals=env.goals.tolist(),flow=flow.tolist());state={'state_uid':r['state_uid'],'physical':p}
  for kind,eta in [('C1',np.zeros(3)),('C3',r['eta'])]:
   values=[]
   for seed in seeds:
    PROTOCOL['probe_rng']=seed
    values.append([cached(a,state,eta,'noise_diagnostic_'+kind)['context'] for a in rt])
   a=np.array(values);between=float(np.linalg.norm(a[:,1].mean(0)-a[:,0].mean(0)))
   within=float(np.sqrt(np.mean(np.sum((a-a.mean(0))**2,axis=-1))))
   records.append({'state_uid':r['state_uid'],'eta_uid':r['eta_uid'],'kind':kind,'between_controller_distance':between,
     'within_controller_RNG_RMS':within,'signal_to_noise':between/max(within,1e-12),'contexts':a.tolist()})
 summary={kind:{'median_between':float(np.median([r['between_controller_distance'] for r in records if r['kind']==kind])),
  'median_within_noise':float(np.median([r['within_controller_RNG_RMS'] for r in records if r['kind']==kind])),
  'median_signal_to_noise':float(np.median([r['signal_to_noise'] for r in records if r['kind']==kind]))} for kind in ('C1','C3')}
 dump(OUT/'probe_noise_audit.json',{'seeds':seeds,'summary':summary,'records':records,'models_changed':False,'new_full_continuations':0})
 print(summary)
if __name__=='__main__':main()
