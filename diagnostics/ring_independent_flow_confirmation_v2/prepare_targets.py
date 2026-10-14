from core import *
import pickle,copy
from shared_rollout_db.src.rollout_db import uid,connect
from diagnostics.controller_new_checkpoint_confirmation_v1.fp32_n8.compatibility import param_leaves
# Do not import old core via this path; functions above resolved core to this directory.
def paramhash(p):
 x=pickle.loads(Path(p).read_bytes());leaves=param_leaves(x['agent']['network']['params']);h=hashlib.sha256()
 for path,a in leaves.items():h.update(path.encode());h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest(),leaves,x
prereg();old=read(R10/'protocol.json');plan={x['run']:x for x in read(HERE/'TRAINING_PLAN.json')};source_names=old['controller_order'];existing={};errors=[]
for p in sorted((ROOT/'diagnostics').glob('**/*.pkl')):
 if p.is_relative_to(HERE):continue
 try:h,_,_=paramhash(p);existing.setdefault(h,[]).append(str(p))
 except Exception as ex:errors.append(dict(path=str(p),type=type(ex).__name__))
profiles={};rows=[];data=[]
import jax
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
jax.config.update('jax_enable_x64',False)
probe_state=read(R5/'stage_2/probes/protocol.json')['states'][0]
for n in source_names:
 item=plan[n];new=HERE/('repaired_new_flows' if n=='base_u_plus_w_v4_wide_macflow' else 'new_flows')/n/'best.pkl';h,leaves,x=paramhash(new);oh,oleaves,ox=paramhash(ROOT/item['original_best']);assert h not in existing,(n,existing.get(h));assert set(leaves)==set(oleaves);cfg=x['config'];ocfg=ox['config'];norms=['obs_mean','obs_scale','act_mean','act_scale'];assert all(np.array_equal(np.asarray(cfg[k]),np.asarray(ocfg[k])) for k in norms)
 for k in ocfg:assert clean(cfg.get(k))==clean(ocfg[k]),(n,k)
 d=np.concatenate([(leaves[k].astype(float)-oleaves[k].astype(float)).ravel() for k in leaves]);w=np.concatenate([oleaves[k].astype(float).ravel() for k in leaves]);assert all(leaves[k].shape==oleaves[k].shape and leaves[k].dtype==np.float32 for k in leaves)
 base=old['base_profiles'][n];v=copy.deepcopy(base['payload']);manifest=v.get('adapter_manifest_path',str(FIELD/'assets/diagnostics/ring_exchange_stage1/base_u_v10_local_dataset/manifest.json'));frame=v.get('observation_frame','local');m=read(manifest);rt=RingFlowAdapter(new,m,frame);env=rt.make_env(probe_state);key=jax.random.PRNGKey(2026101402);a=np.asarray(rt.native_fn(rt.observation(env),key));b=np.asarray(rt.native_fn(rt.observation(env),key));assert a.tobytes()==b.tobytes() and a.dtype==np.float32
 name=n+'__independent';v.update(flow_checkpoint_path=str(new),flow_checkpoint_sha256=sha(new),adapter_manifest_path=manifest,adapter_manifest_sha256=sha(manifest),observation_frame=frame,checkpoint_environment_fingerprint=m['environment_fingerprint'],execution_callable=str(P1/'flow_adapter.py'),execution_callable_sha256=sha(P1/'flow_adapter.py'),controller_source_sha256=sha(P1/'flow_adapter.py'),parent_best_controller_uid=base['controller_uid'],checkpoint_role='independent training seed original+10000 best');profiles[name]=dict(controller_uid=uid('ctl',v),payload=v,parent_best_name=n);rows.append(dict(name=name,run=n,original_seed=item['original_seed'],new_seed=item['new_seed'],new_param_sha256=h,old_param_sha256=oh,parameter_L2=float(np.linalg.norm(d)),parameter_relative_L2=float(np.linalg.norm(d)/np.linalg.norm(w)),normalizers_identical=True,architecture_identical=True,forward_reproducible=True,new_checkpoint_sha256=sha(new),jax_enable_x64=False))
 for p in sorted((ROOT/item['dataset']).rglob('*')):
  if p.is_file():data.append(dict(recipe=n,path=str(p),sha256=sha(p)))
assert len(profiles)==14 and len({x['new_param_sha256'] for x in rows})==14
with connect(True) as db:
 for x in rows:assert db.execute('SELECT COUNT(*) FROM rollout r JOIN controller_config c ON r.controller_uid=c.controller_uid WHERE c.flow_checkpoint_sha256=?',(x['new_checkpoint_sha256'],)).fetchone()[0]==0
freeze(HERE/'FINAL_BASE_PROFILES.json',dict(profiles=profiles,eligible_controller_order=list(profiles),full_controller_order=list(profiles),best_controller_order=source_names));write(HERE/'COMPATIBILITY_AUDIT.json',dict(status='PASS',usable=14,checks=rows,success_labels_generated=False,physical_steps=0));write(HERE/'CHECKPOINT_NOVELTY_AUDIT.json',dict(utc=utc(),passed=True,historical_unique_param_hashes=len(existing),historical_paths=sum(map(len,existing.values())),unreadable_nonmatching_schemas=errors,checkpoints=rows));write(HERE/'DATASET_FILE_HASHES.json',data)
print('14 independent checkpoints compatible, novel and no success labels')
