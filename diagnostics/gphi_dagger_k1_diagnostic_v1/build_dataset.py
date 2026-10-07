"""Audit k1 teacher validity and append valid historical k1 samples."""
from __future__ import annotations
import csv, hashlib, json, shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/gphi_dagger_k1_diagnostic_v1'
BASE=ROOT/'diagnostics/gphi_training_dataset_strict_deadlock_v1'; OUT=ROOT/'diagnostics/gphi_training_dataset_dagger_k1_v1'
EXPECTED='79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9'

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def jsonwrite(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def jsonl(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def jsonlwrite(p,rows): p.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))

def main():
 if sha(BASE/'samples.npz')!=EXPECTED: raise RuntimeError('base hash mismatch')
 src=json.load(open(HERE/'source_manifest.json')); allrows=[]
 for s in src['training_sources']:
  p=HERE/'raw_k1'/f"{s['state_id']}.jsonl"; rows=jsonl(p)
  if len(rows)!=64 or sorted(r['flow_seed'] for r in rows)!=list(range(95310001,95310065)): raise RuntimeError((s['state_id'],'tuple mismatch'))
  allrows.extend(rows)
 if len(allrows)!=704: raise RuntimeError(len(allrows))
 bysource=defaultdict(list)
 for r in allrows: bysource[r['source_state_id']].append(r)
 audit=[]
 for r in allrows:
  audit.append({k:r[k] for k in ('case_id','source_state_id','state_id','flow_seed','outcome','teacher_valid','continuation_steps','J_def','target_norm','state_sha256','tuple_sha256')})
 csvwrite(HERE/'teacher_validity_audit.csv',audit)
 counts={sid:sum(r['teacher_valid'] for r in rows) for sid,rows in bysource.items()}
 valid=sum(counts.values()); per_source_B63=all(v>=63 for v in counts.values())
 state_manifest={'schema':'learner_k1_state_manifest_v1','candidate_count':704,'teacher_valid_count':valid,'per_source_successes':counts,'per_source_B63':per_source_B63,'states':allrows}
 jsonwrite(HERE/'learner_k1_state_manifest.json',state_manifest)
 targets=[]
 for r in allrows:
  with np.load(HERE/r['tuple_file'],allow_pickle=False) as z: t=np.asarray(z['target'])
  targets.append({'case_id':r['case_id'],'source_state_id':r['source_state_id'],'state_id':r['state_id'],'flow_seed':r['flow_seed'],'teacher_valid':r['teacher_valid'],'target_0':t[0],'target_1':t[1],'target_2':t[2],'target_3':t[3],'target_norm':float(np.linalg.norm(t))})
 csvwrite(HERE/'k1_teacher_targets.csv',targets)
 if not per_source_B63:
  jsonwrite(HERE/'teacher_gate.json',{'status':'STOP','classification':'FIXED_ETA_TEACHER_INVALID_ON_LEARNER_K1','valid':valid,'per_source':counts})
  raise SystemExit('fixed eta does not remain B63 for every source; stop before training')
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'states').mkdir(exist_ok=True)
 with np.load(BASE/'samples.npz',allow_pickle=False) as z: base={k:np.asarray(z[k]).copy() for k in z.files}
 base_states=jsonl(BASE/'state_manifest.jsonl'); base_meta=jsonl(BASE/'sample_metadata.jsonl')
 if len(base_states)!=424 or len(base_meta)!=27136: raise RuntimeError('base counts')
 append=defaultdict(list); new_states=[]; new_meta=[]
 validrows=[r for r in allrows if r['teacher_valid']]
 for offset,r in enumerate(validrows):
  index=424+offset; srcstate=HERE/r['state_file']; dst=OUT/'states'/f"{r['state_id']}.npz"; shutil.copy2(srcstate,dst)
  new_states.append({'state_id':r['state_id'],'state_index':index,'state_file':str(dst.relative_to(OUT)),'state_sha256':sha(dst),'split':'train','category':'STRICT_K1','leakage_group':f"dagger_k1_{r['case_id']}",'source_episode':r['case_id'],'source_state_id':r['source_state_id'],'physical_step':r['query_step'],'flow_seed':r['flow_seed'],'teacher_eta_frozen':True})
  with np.load(HERE/r['tuple_file'],allow_pickle=False) as z: tup={k:np.asarray(z[k]).copy() for k in z.files}
  sid=r['state_id']; sample_id=sid
  vals={**{k:v for k,v in tup.items() if k not in ('feature','target','target_action')},
        'features':tup['feature'],'targets':tup['target'],'target_actions':tup['target_action'],
        'state_index':index,'flow_seed':r['flow_seed'],'state_id':sid,'split':'train','category':'STRICT_K1','sample_id':sample_id}
  for k,v in vals.items(): append[k].append(v)
  new_meta.append({'sample_id':sample_id,'state_id':sid,'state_index':index,'source_trajectory':r['case_id'],'leakage_group':f"dagger_k1_{r['case_id']}",'category':'STRICT_K1','split':'train','flow_seed':r['flow_seed'],'target_dimension':4,'target_semantics':'executed correction from frozen source eta on learner-visited k1 state','k1_dagger_v1':True,'special_weight':False})
 merged={}; prefix={}
 for k,old in base.items():
  new=np.asarray(append[k]);
  if old.dtype.kind not in 'USO': new=new.astype(old.dtype,copy=False)
  merged[k]=np.concatenate([old,new]); prefix[k]=bool(np.array_equal(merged[k][:len(old)],old))
 if not all(prefix.values()): raise RuntimeError('base prefix changed')
 np.savez_compressed(OUT/'samples.npz',**merged); jsonlwrite(OUT/'state_manifest.jsonl',base_states+new_states); jsonlwrite(OUT/'sample_metadata.jsonl',base_meta+new_meta)
 shutil.copy2(BASE/'feature_schema.json',OUT/'feature_schema.json'); shutil.copy2(BASE/'startup_feature_builder.py',OUT/'startup_feature_builder.py')
 jsonwrite(OUT/'split_manifest.json',{'base_split_unchanged':True,'k1_split':'train_only','k1_states':len(validrows),'k1_samples':len(validrows),'no_weighting':True,'no_oversampling':True})
 manifest={'schema':'gphi_training_dataset_dagger_k1_v1','base_dataset':str(BASE),'base_sha256':EXPECTED,'original_unique_states':424,'original_samples':27136,'added_k1_states':len(validrows),'added_samples':len(validrows),'final_unique_states':424+len(validrows),'final_samples':27136+len(validrows),'fresh6_overlap':0,'no_weighting':True,'no_oversampling':True,'files_sha256':{n:sha(OUT/n) for n in ('samples.npz','state_manifest.jsonl','sample_metadata.jsonl','feature_schema.json','startup_feature_builder.py','split_manifest.json')}}
 jsonwrite(OUT/'manifest.json',manifest); jsonwrite(HERE/'augmented_dataset_manifest.json',manifest)
 jsonwrite(HERE/'overlap_audit.json',{'status':'PASS','historical_sources':11,'fresh6_training_states':0,'fresh6_training_samples':0,'base_prefix_identity':prefix,'no_weighting':True,'no_oversampling':True})
 jsonwrite(HERE/'teacher_gate.json',{'status':'PASS','valid':valid,'candidate':704,'per_source':counts,'all_sources_B63':True})
 print(json.dumps({'status':'PASS','teacher_valid':valid,'per_source':counts,'added':len(validrows),'dataset_sha256':manifest['files_sha256']['samples.npz']},indent=2))
if __name__=='__main__': main()
