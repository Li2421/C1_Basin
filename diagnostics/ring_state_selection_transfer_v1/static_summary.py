"""Reproduce static summary artifacts and historical simultaneous CI without new queries."""
import json,hashlib,platform,sys,importlib.metadata as md
from pathlib import Path
import numpy as np
from study import OUT,CONF,read,write
from diagnostics.state_nuisance_rootcause_v1.adaptive_selection_analyze_v1 import all_fixed_falsification
p=np.load(CONF/'frozen_predictions.npz');y=np.load(CONF/'fresh_rollouts.npz')['outcomes'];r=all_fixed_falsification(y[:,:,8:,0].sum(-1),y[:,:,8:,1].sum(-1),np.arange(256),p['physical23_eta_n512_logits'].argmax(-1));write(OUT/'HISTORICAL_SIMULTANEOUS_REPRODUCTION.json',r)
a=np.load(OUT/'EFFECTIVE_BASIS_QUERIES.npz');names=['rotate_negative','-0.5','isotropic','0.5','rotate_positive'];r=[]
for i in range(8):
 acts=np.array([a[f's{i}_{n}_zero_action'] for n in names]);r.append(dict(state=i,zero_eta_projected_action_max_pair_difference=float(np.max(np.linalg.norm(acts[:,None]-acts[None,:],axis=-1)))))
write(OUT/'ZERO_ETA_PROJECTED_ACTION_AUDIT.json',dict(rows=r,important_boundary='Endpoint equality is unprojected nominal equality. Intermediate safety/speed projection can create controller effects even eta=0, so R(M) B alone is not exact FF execution.'))
r=[]
for v in read(OUT/'c1/static_operator_audit.json')['rows'][:5]:
 R=np.array(v['response'])[:2,:2];r.append(dict(controller=v['controller'],effective_unprojected_gain=float(np.linalg.norm(R[:,0])),effective_rotation_degrees=float(np.degrees(np.arctan2(R[1,0],R[0,0])))))
write(OUT/'EFFECTIVE_OPERATOR_COORDINATES.json',dict(rows=r,limitation='kappa changes angle AND gain; formula describes affine unprojected response only.'))
cp=[]
for f in (OUT.parents[1]/'diagnostics/ring_exchange_stage1').glob('*/best.pkl'):
 cp.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size,config=str(f.with_name('config.json'))))
write(OUT/'FLOW_CHECKPOINT_INDEX.json',cp)
write(OUT/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={k:md.version(k) for k in ['numpy','scipy','jax','jaxlib','flax','optax']},resources='Slurm node24CPU/128GB/1GPU8shards;22CPUrolloutworkers;1GPUshardperfit/queryjob'))
