import os
os.environ.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
import sys,json,csv,time,hashlib,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]; F=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
for p in (ROOT,F,F/'source',F/'source/scripts'):sys.path.insert(0,str(p))
import numpy as np,jax
from scipy.optimize import brentq
from validation_stage2.run_rollouts import make_runtime
start=time.monotonic();cpu=time.process_time()
A=json.loads((HERE.parent/'agent_a/geometry_contract.json').read_text()); R=A['constants_m']['R_w'];D=A['constants_m']['D_s'];h=.2-R
rt=make_runtime('toy_give_way')
manifest=Path('/home/zhihan/research/Basin_C1_hard_field_20261006/confirmation_manifest.json')
st=next(s for s in json.loads(manifest.read_text())['states'] if s['scenario']=='toy_give_way')
rows=[];traces=[];requests=0;fails=[]
for idx,p in enumerate(([[-.4,0],[0,0]],[[-D,0],[0,0]],[[-D/2,0],[D/2,0]])):
 for seed in (0,1):
  env=rt.make_env(st);env.positions=np.asarray(p,float);env.velocities=np.zeros((2,2))
  key=jax.random.fold_in(jax.random.PRNGKey(2026100601),int(hashlib.sha256(f'K_zero_offset_{idx}'.encode()).hexdigest()[:8],16));key=jax.random.fold_in(jax.random.fold_in(key,seed),0)
  obs,safety,flow,safe,delta,_=rt.prepare(env,key,np.asarray([.5,0,.5]))
  for label,d in [('zero',np.zeros_like(delta)),('witness',delta)]:
   for chain in ('TT','FF'):
    before=safety.calls
    try:
     if chain=='TT':u,diag=safety.project(safe+d)
     else:
      u,trace=rt.integrate(env,obs,key,d,safety,traced=True);diag={'statuses':[x['solver_status'] for x in trace['steps']]}
      traces.append(dict(state_id=f'K_zero_offset_{idx}',seed=seed,parameter=label,path=trace['path'].tolist(),targets=trace['targets'].tolist(),steps=trace['steps']))
     sl=safety.A@u.ravel()-safety.lower
     rows.append(dict(state_id=f'K_zero_offset_{idx}',seed=seed,key=np.asarray(key).tolist(),positions=p,eta=[0,0,0] if label=='zero' else [.5,0,.5],parameter=label,chain=chain,raw_delta=d.tolist(),native_flow=flow.tolist(),safe_anchor=safe.tolist(),executed=u.tolist(),u2y=float(u[1,1]),pass_gap=float(u[0,0]-u[1,0]),active_rows=np.where(sl<=1e-7)[0].tolist(),min_cbf_slack=float(min(sl)),min_speed_slack=float(min(.5-np.linalg.norm(u,axis=1))),projector_requests=safety.calls-before,solver=diag,numerical_failure=False))
    except Exception as e:fails.append(dict(state=idx,seed=seed,chain=chain,error=repr(e)))
  requests+=safety.calls
threshold=[]
# Fixed family p1=(-a,0), p2=(0,z): actual bounded_rows formula.
def raw(z,a=.4):return .5*z*(min(1.,.5/math.hypot(a,z))-.5/math.hypot(1.09,z))
for eps in (.001,.005,.01):
 z=brentq(lambda z:raw(z)-eps,0,.2)
 threshold.append(dict(raw_margin=eps,z_min_m=z,relative_to_main_full_width=z/(2*h),relative_to_y2_upward_range=z/h,fits_y1_0_main_channel=z<=h))
result=dict(geometry_contract_sha256=hashlib.sha256((HERE.parent/'agent_a/geometry_contract.json').read_bytes()).hexdigest(),source_hashes=A['source_hashes'],checkpoint_sha256=A['checkpoint_sha256'],probe_state_count=3,probe_seeds=[0,1],noise_root=2026100601,noise_uid_rule='K_zero_offset_i; SHA256 first eight hex uint32; fold root, UID, seed, step0',action_rows=len(rows),native_samples=6,native_network_steps=60,ff_integrations=12,ff_fields=120,projector_requests=requests,new_rollouts=0,new_steps=0,new_training=0,numerical_failures=fails,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,reference_entry_box=dict(x1=[-.65,-.35],x2=[-.025,.025],y1=[-.02,0],y2=[-.02,.02]),reference_target_y2=.4,reference_vY=.03,reference_pair_cbf_slack_lower=.325**2-D**2-2*.02*.03,reference_corner_speed_lower=math.sqrt(.175**2-R**2),reference_top_cbf_slack_lower=.68-.4-R-.03,reference_min_wall_clearance_lower=.175-R,reference_min_pair_distance=.325,reference_time_upper=14.,reference_steps_upper=280,raw_seed_fixed_family=threshold,raw_at_main_y2_top=raw(h),z_seed_zero_raw=0.)
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');(HERE/'probe_traces.json').write_text(json.dumps(traces,indent=2)+'\n')
with (HERE/'zero_offset_actions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
print(json.dumps(result,indent=2));print('MARGINS',[(r['state_id'],r['seed'],r['parameter'],r['chain'],r['u2y'],r['pass_gap']) for r in rows])
