from core import *
import pickle,copy
from shared_rollout_db.src.rollout_db import uid
from diagnostics.controller_new_checkpoint_confirmation_v1.fp32_n8.compatibility import param_leaves
def paramhash(p):
 x=pickle.loads(Path(p).read_bytes());h=hashlib.sha256();leaves=param_leaves(x['agent']['network']['params'])
 for path,a in leaves.items():h.update(path.encode());h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest(),leaves,x
def main():
 prereg();plan=read(HERE/'TRAINING_PLAN.json');old=read(R10/'protocol.json');existing={};unreadable=[]
 expected={x['path']:x['sha256'] for x in read(R14/'DATASET_FILE_HASHES.json')}
 assert all(sha(path)==digest for path,digest in expected.items()),'Historical training dataset file drift'
 write(HERE/'DATASET_FILE_HASHES_AUDIT.json',dict(utc=utc(),passed=True,unique_files=len(expected),source_sha256=sha(R14/'DATASET_FILE_HASHES.json')))
 for p in sorted((ROOT/'diagnostics').glob('**/*.pkl')):
  if p.is_relative_to(HERE):continue
  try:h,_,_=paramhash(p);existing.setdefault(h,[]).append(str(p))
  except Exception as e:unreadable.append(dict(path=str(p),type=type(e).__name__))
 import jax
 from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
 jax.config.update('jax_enable_x64',False);assert not jax.config.x64_enabled
 probe_state=read(R5/'stage_2/probes/protocol.json')['states'][0];profiles={};checks=[];seen={};training=[]
 for item in plan:
  run=item['run'];seed=item['new_seed'];ck=HERE/'checkpoints'/run/str(seed)/'best.pkl';assert ck.exists(),ck;h,leaves,x=paramhash(ck);oh,oleaves,ox=paramhash(ROOT/item['original_best'])
  assert seed==item['original_seed']+30000+item['wave'] and seed not in {item['original_seed']+10000,*(item['original_seed']+20000+j for j in (1,2,3))}
  start=HERE/'training_receipts'/f'{run}_{seed}.json';receipt=read(start);assert receipt['kwargs']['seed']==seed and any(str(d).startswith('cuda') for d in receipt['devices']) and receipt['prereg_sha256']==sha(HERE/'PREREGISTRATION.md')
  summary=ck.parent/'training_summary.json';metrics=ck.parent/'metrics.jsonl';assert summary.exists() and metrics.exists();training.append(dict(recipe=run,seed=seed,wave=item['wave'],start_receipt=str(start),start_receipt_sha256=sha(start),training_summary=read(summary),training_summary_sha256=sha(summary),metrics_sha256=sha(metrics),best_checkpoint_sha256=sha(ck),final_checkpoint_sha256=sha(ck.parent/'final.pkl')))
  assert h not in existing and h not in seen,(run,seed,existing.get(h),seen.get(h));seen[h]=str(ck)
  cfg=x['config'];ocfg=ox['config'];assert set(leaves)==set(oleaves)
  assert all(leaves[k].shape==oleaves[k].shape and leaves[k].dtype==np.float32 for k in leaves)
  assert all(np.array_equal(np.asarray(cfg[k]),np.asarray(ocfg[k])) for k in ['obs_mean','obs_scale','act_mean','act_scale'])
  for k in ocfg:assert clean(cfg.get(k))==clean(ocfg[k]),(run,seed,k)
  base=old['base_profiles'][run];v=copy.deepcopy(base['payload']);manifest=v.get('adapter_manifest_path',str(FIELD/'assets/diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json'));frame=v.get('observation_frame','local');m=read(manifest)
  rt=RingFlowAdapter(ck,m,frame);env=rt.make_env(probe_state);key=jax.random.PRNGKey(2026101402);a=np.asarray(rt.native_fn(rt.observation(env),key));b=np.asarray(rt.native_fn(rt.observation(env),key));assert a.tobytes()==b.tobytes() and a.dtype==np.float32
  name=f'{run}__seed{item["wave"]}';v.update(flow_checkpoint_path=str(ck),flow_checkpoint_sha256=sha(ck),adapter_manifest_path=manifest,adapter_manifest_sha256=sha(manifest),observation_frame=frame,checkpoint_environment_fingerprint=m['environment_fingerprint'],execution_callable=str(P1/'flow_adapter.py'),execution_callable_sha256=sha(P1/'flow_adapter.py'),controller_source_sha256=sha(P1/'flow_adapter.py'),parent_best_controller_uid=base['controller_uid'],checkpoint_role=f'independent seed original+{30000+item["wave"]}')
  profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_best_name=run,wave=item['wave'],training_seed=seed)
  checks.append(dict(name=name,recipe=run,seed=seed,wave=item['wave'],checkpoint_path=str(ck),checkpoint_sha256=sha(ck),parameter_sha256=h,original_parameter_sha256=oh,normalizers_identical=True,architecture_identical=True,forward_reproducible=True,jax_enable_x64=False))
 assert len(profiles)==56
 freeze(HERE/'FINAL_BASE_PROFILES.json',dict(profiles=profiles,eligible_controller_order=list(profiles),recipe_order=old['controller_order']))
 write(HERE/'COMPATIBILITY_AUDIT.json',dict(status='PASS',usable=56,checks=checks,success_labels_generated=False))
 write(HERE/'CHECKPOINT_NOVELTY_AUDIT.json',dict(utc=utc(),passed=True,historical_unique_param_hashes=len(existing),historical_paths=sum(map(len,existing.values())),unreadable_nonmatching_schemas=unreadable,checkpoints=checks))
 write(HERE/'TRAINING_AUDIT.json',dict(utc=utc(),passed=True,controllers=len(training),GPU_and_historical_package_verified=True,rows=training))
 print('56 checkpoints compatible and parameter-novel')
if __name__=='__main__':main()
