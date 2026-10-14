import os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';sys.dont_write_bytecode=True
from pathlib import Path
import pickle,json,hashlib,sqlite3,datetime
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def leaves(x,p=()):
 if isinstance(x,dict):
  for k in sorted(x):yield from leaves(x[k],p+(str(k),))
 else:yield '/'.join(p),np.asarray(x)
def ph(x):
 h=hashlib.sha256()
 for p,a in leaves(x):h.update(p.encode());h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
refpath=ROOT/'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_source_balanced_macflow/best.pkl';ref=pickle.loads(refpath.read_bytes());rc=ref['config'];rs=[(p,list(a.shape),str(a.dtype)) for p,a in leaves(ref['agent']['network']['params'])]
fields=['actor_hidden_dims','actor_layer_norm','num_agents','obs_dim','act_dim','joint_obs_dim','joint_act_dim','flow_steps','normalize','agent_order','environment_fingerprint','max_speed'];norms=['obs_mean','obs_scale','act_mean','act_scale'];rows=[]
db=sqlite3.connect('file:'+str(ROOT/'shared_rollout_db/rollout.sqlite')+'?mode=ro',uri=True)
for path in sorted((ROOT/'diagnostics/four_way_intersection_stage1').glob('*/best.pkl')):
 best=pickle.loads(path.read_bytes());bh=ph(best['agent']['network']['params'])
 for kind in ['best','final']:
  p=path.with_name(kind+'.pkl');x=pickle.loads(p.read_bytes());c=x['config'];params=x['agent']['network']['params'];struct=[(q,list(v.shape),str(v.dtype)) for q,v in leaves(params)];diff=[k for k in fields if c.get(k)!=rc.get(k)];ndiff=[k for k in norms if not np.array_equal(np.asarray(c.get(k)),np.asarray(rc.get(k)))];h=hashlib.sha256(p.read_bytes()).hexdigest();count=db.execute('SELECT COUNT(*) FROM rollout r JOIN controller_config c ON r.controller_uid=c.controller_uid WHERE c.flow_checkpoint_sha256=?',(h,)).fetchone()[0]
  rows.append(dict(run=path.parent.name,kind=kind,path=str(p),file_sha256=h,param_sha256=ph(params),param_different_from_best=ph(params)!=bh,structure_matches=struct==rs,config_differences=diff,normalizer_differences=ndiff,compatible_metadata=not diff and not ndiff and struct==rs,DB_rollout_count=count))
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference=str(refpath),reference_file_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),best_runs=sum(x['kind']=='best' for x in rows),rows=rows,compatible_best=[x['run'] for x in rows if x['kind']=='best' and x['compatible_metadata']],eligible_final=[x['run'] for x in rows if x['kind']=='final' and x['compatible_metadata'] and x['param_different_from_best']])
(HERE/'COMPATIBILITY_METADATA_AUDIT.json').write_text(json.dumps(obj,indent=2));print(json.dumps({k:v for k,v in obj.items() if k!='rows'},indent=2))
