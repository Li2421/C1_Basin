"""Materialize only predeclared Q16 panel cells, preserving exact rollout provenance."""
from pathlib import Path
from collections import defaultdict,Counter
import sqlite3,json,hashlib,datetime
import numpy as np
HERE=Path(__file__).resolve().parent;DB=HERE.parents[1]/'shared_rollout_db/rollout.sqlite'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

def seedinfo(k):
 try:z=json.loads(k)
 except Exception:return ('lexicographic',k,'',k)
 if 'future_index' in z and isinstance(z['future_index'],int):kind='future_index';ix=z['future_index']
 elif 'seed' in z and isinstance(z['seed'],int):kind='numeric_seed';ix=z['seed']
 else:kind='lexicographic';ix=k
 root=json.dumps({a:v for a,v in z.items() if a not in ('future_index','seed')},sort_keys=True)
 return kind,ix,root,k

def selected_seedkeys(cells):
 common=set.intersection(*(set(x) for x in cells))
 groups=defaultdict(list)
 for k in common:
  kind,ix,root,_=seedinfo(k);groups[(kind,root)].append((ix,k))
 choices=[]
 for (kind,root),items in groups.items():
  if len({i for i,_ in items})<16:continue
  chosen=[];seen=set()
  for ix,k in sorted(items,key=lambda v:(str(v[0]) if kind=='lexicographic' else v[0],v[1])):
   if ix in seen:continue
   seen.add(ix);chosen.append(k)
   if len(chosen)==16:break
  choices.append((kind,root,chosen,len(items)))
 if not choices:return None,{'reason':'no_16_common_semantic_seed_keys','common_literal_keys':len(common),'seed_groups':[(k,len(v)) for k,v in groups.items()]}
 # Coverage-only choice, independent of success values.
 kind,root,chosen,n=max(choices,key=lambda x:(x[3],x[0]=='future_index',-len(x[1]),x[1]))
 return chosen,{'kind':kind,'root':root,'common_literal_keys':len(common),'selected_index_values':[seedinfo(k)[1] for k in chosen],'seed_groups_available':len(choices)}

def main():
 inv=json.loads((HERE/'inventory.json').read_text());panels=[p for p in inv['panels'] if p['eligible_metadata']];wanted={(p['controller_uid'],s,e) for p in panels for s in p['state_uids'] for e in p['eta_uids']}
 c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA temp_store=MEMORY')
 c.execute('CREATE TEMP TABLE wanted(controller_uid TEXT,state_uid TEXT,eta_uid TEXT,PRIMARY KEY(controller_uid,state_uid,eta_uid)) WITHOUT ROWID')
 c.executemany('INSERT INTO wanted VALUES(?,?,?)',wanted)
 plan=[tuple(r) for r in c.execute('''EXPLAIN QUERY PLAN SELECT r.rollout_uid FROM wanted w CROSS JOIN rollout r INDEXED BY idx_rollout_lookup
     WHERE r.state_uid=w.state_uid AND r.eta_uid=w.eta_uid AND r.controller_uid=w.controller_uid
     AND r.conflict_quarantined=0 AND r.numerical_failure=0''')]
 cells=defaultdict(dict);sources=Counter();experiments=Counter();rows=0
 for r in c.execute('''SELECT r.controller_uid,r.state_uid,r.eta_uid,r.seed_key,r.success,r.rollout_uid,r.experiment_uid,r.original_source_file
     FROM wanted w CROSS JOIN rollout r INDEXED BY idx_rollout_lookup
     WHERE r.state_uid=w.state_uid AND r.eta_uid=w.eta_uid AND r.controller_uid=w.controller_uid
     AND r.conflict_quarantined=0 AND r.numerical_failure=0'''):
  key=(r['controller_uid'],r['state_uid'],r['eta_uid']);assert r['seed_key'] not in cells[key];cells[key][r['seed_key']]=(r['success'],r['rollout_uid'],r['experiment_uid'],r['original_source_file']);rows+=1
 assert len(cells)==len(wanted)
 # n>=16 came from the independent inventory query; ensure raw exact agreement.
 assert all(len(cells[k])>=16 for k in wanted)
 root=HERE/'panel_data';root.mkdir(exist_ok=True);index=[];all_used=set()
 for p in panels:
  ss=p['state_uids'];ee=p['eta_uids'];n,m=len(ss),len(ee);Y=np.zeros((n,m,16),np.uint8);U=np.empty((n,m,16),dtype='<U96');S=np.empty((n,16),dtype='<U256');used=set();policies=[];problem=[]
  for i,s in enumerate(ss):
   keys,info=selected_seedkeys([cells[p['controller_uid'],s,e] for e in ee]);policies.append(info)
   if keys is None:problem.append({'state_uid':s,**info});continue
   S[i]=keys
   for j,e in enumerate(ee):
    for k,seed in enumerate(keys):
     v,roll,ex,path=cells[p['controller_uid'],s,e][seed];Y[i,j,k]=v;U[i,j,k]=roll;used.add(roll);sources[path]+=1;experiments[ex]+=1
  if problem:
   index.append({'panel_uid':p['panel_uid'],'status':'seed_alignment_failed','problems':problem,'n_states':n,'n_eta':m,'scene':p['scene'],'controller_uid':p['controller_uid']});continue
  f=root/(p['panel_uid']+'.npz');np.savez_compressed(f,Y=Y,rollout_uids=U,seed_keys=S,state_uids=np.asarray(ss),eta_uids=np.asarray(ee));all_used|=used
  methods=Counter(x['kind'] for x in policies);roots=Counter(x['root'] for x in policies)
  index.append({'panel_uid':p['panel_uid'],'status':'ready','file':str(f),'sha256':sha(f),'n_states':n,'n_eta':m,'n_seed_per_cell':16,'selected_rollouts':len(used),'scene':p['scene'],'controller_uid':p['controller_uid'],'seed_methods':dict(methods),'seed_root_variants':len(roots),'selection_policy_by_state':policies,'state_source_groups':p['state_source_groups'],'controller_experiment_uids':p['controller_experiment_uids']})
 write(HERE/'PANEL_INDEX.json',index)
 source_rows=[]
 for path,count in sources.items():
  f=Path(path);source_rows.append({'path':path,'used_rows_across_overlapping_panels':count,'exists':f.exists(),'sha256':sha(f) if f.is_file() else None})
 write(HERE/'SOURCE_FILES.json',source_rows)
 write(HERE/'EXTRACTION_AUDIT.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'database_path':str(DB),'sql_query_plan':plan,'selected_Q16_cells':len(wanted),'retrieved_valid_rollouts_before_first16_filter':rows,'structural_eligible_panels':len(panels),'aligned_panels':sum(p['status']=='ready' for p in index),'seed_alignment_failed':sum(p['status']!='ready' for p in index),'unique_selected_rollouts_after_first16':len(all_used),'source_file_paths':len(sources),'source_experiment_counts':dict(experiments),'success_values_preserved_in_npz':True,'numerical_failure_and_conflict_excluded_from_binary_labels':True,'no_new_rollouts':True})
 print({'selected_cells':len(wanted),'rows':rows,'panels':len(index),'ready':sum(p['status']=='ready' for p in index),'unique_selected_rollouts':len(all_used),'source_files':len(sources)},flush=True)
 c.close()
if __name__=='__main__':main()
