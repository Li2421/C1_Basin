"""Autonomous search: exact acceleration -> gate -> discovery -> frozen16 -> confirm."""
from common import *
from scene_search import case
from design import controllers,menu,CONTROL_IDS
import numpy as np,subprocess,fcntl
PY=sys.executable

def log(text):
 print(utc(),text,flush=True)
 with(HERE/'ITERATION_LOG.md').open('a')as f:f.write(f'\n- {utc()} {text}\n')

def merge(stage):
 with(HERE/'MERGER.lock').open('a')as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  p=subprocess.run([PY,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,capture_output=True,text=True)
  write(HERE/f'{stage}_MERGE.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr

def prepare(stage,cases):
 marker=HERE/f'{stage}_PREFLIGHT.receipt.json'
 if marker.exists():return
 write(HERE/'STATUS.json',dict(phase='registration',stage=stage,cases=len(cases),utc=utc()))
 register(stage,cases)
 subprocess.run([PY,'-B','-m','shared_rollout_db.plan','--manifest',str(HERE/f'{stage}_PLANNED_ROLLOUTS.json'),'--output',str(HERE/f'{stage}_OFFICIAL_PREFLIGHT.json')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def stage(stage,cases):
 prepare(stage,cases)
 if not(HERE/f'{stage}_LOCAL_COMPLETE.json').exists():
  subprocess.run([PY,'-B',str(HERE/'launch_local.py'),stage],cwd=HERE,check=True)
 merge(stage)

def result(stage,c):
 p=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
 if p.exists():return read(p)
 with dbm.connect(True)as db:
  rows=db.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid) WHERE rs.rollout_uid=?',(c['rollout_uid'],)).fetchall()
 for row in rows:
  p=Path(row['path'])
  if p.exists():
   with p.open()as f:
    for i,line in enumerate(f,1):
     if i==row['source_line']:return json.loads(line)
 raise FileNotFoundError(c['rollout_uid'])

def labels(stage,scenes,indices):
 cs=read(HERE/f'{stage}_CASES.json');look={}
 for c in cs:look[(c['state_provenance']['scene'],c['state']['index'],c['controller_payload']['nominal']['controller'],c['requested_action'])]=c
 out={}
 for s in scenes:
  n=s['name'];y=np.zeros((3,len(indices),49),np.int16);unknown=np.zeros_like(y);uids=[]
  for ci,ctl in enumerate(CONTROL_IDS):
   for si,i in enumerate(indices):
    for ai,a in enumerate(menu()):
     c=look[(n,i,ctl,a['key'])];r=result(stage,c);y[ci,si,ai]=r['success'];unknown[ci,si,ai]=r['numerical_failure'];uids.append(c['rollout_uid'])
  out[n]=dict(controllers=list(CONTROL_IDS),eta_keys=[a['key']for a in menu()],state_indices=indices,y=y.tolist(),unknown=unknown.tolist(),rollout_uids=uids)
 write(HERE/f'LABEL_TABLES_{stage.upper()}.json',out);return out

def pairs_prepare(scenes):
 chosen=[next(s for s in scenes if s['group']==g)for g in ['obstacle','plus','box','swap']];actions=[a for a in menu()if a['key']in ['sE22','sE05']]
 cs=[case(s,c,st,a)for s in chosen for c in controllers()for st in s['states'][:2]for a in actions];prepare('new_pairs',cs)


def gate(scenes):
 cs=[]
 for s in scenes:
  for ctl in controllers():
   for st in s['states'][:4]:
    for i in range(st['N']):
     solo=dict(st,N=1,parent_N=st['N'],agent=i,parent_positions=st['positions'],positions=[st['positions'][i]],goals=[st['goals'][i]],velocities=[st['velocities'][i]],index=100*st['index']+i)
     for seed in [0,1]:cs.append(case(s,ctl,solo,menu()[0],seed))
 stage('capability',cs);table={}
 for c in read(HERE/'capability_CASES.json'):
  n=c['state_provenance']['scene'];ctl=c['controller_payload']['nominal']['controller'];r=result('capability',c);v=table.setdefault(n,{}).setdefault(ctl,dict(total=0,success=0,unknown=0));v['total']+=1;v['success']+=int(r['success']);v['unknown']+=int(r['numerical_failure'])
 for rows in table.values():
  for v in rows.values():v.update(rate=v['success']/v['total'],passed=v['success']/v['total']>=.95)
 active=[s for s in scenes if all(v['passed']for v in table[s['name']].values())];write(HERE/'CAPABILITY_TABLE.json',dict(threshold=.95,per_scene=table,passed=[s['name']for s in active]));return active

def allcases(scenes,indices):
 return[case(s,c,s['states'][i],a)for s in scenes for c in controllers()for i in indices for a in menu()]

def select(scenes,tables):
 from scipy.optimize import milp,Bounds,LinearConstraint
 # Expected crossvalidated own-rate versus each conservative full-data fixed eta.
 from statistics_search import half_scores
 names=sorted(tables);by={s['name']:s for s in scenes};ys=np.stack([np.asarray(tables[n]['y'],np.int16).sum(0)for n in names]);rng=np.random.default_rng(2026102710)
 own,shared,best,gc,score=half_scores(ys,rng,P=2000);own=own.mean(1)/(3*2);q=ys.mean(1)/3;best_rate=q.max(1)
 eligible=np.flatnonzero((best_rate>=.2)&(best_rate<=.98));names=[names[i]for i in eligible];q=q[eligible];own=own[eligible];M=len(names)
 if M<16:raise RuntimeError(('INSUFFICIENT_INFORMATIVE_SCENES',M))
 regret=own[:,None]-q;mat=np.c_[-regret.T,np.ones(49)];cons=[LinearConstraint(mat,-np.inf,0),LinearConstraint(np.r_[np.ones(M),0][None],16,16)]
 for g,minn in [('obstacle',4),('plus',4),('box',2),('swap',2)]:
  mask=np.array([by[n]['group']==g for n in names],float);cons.append(LinearConstraint(np.r_[mask,0][None],minn,np.inf))
 res=milp(np.r_[np.zeros(M),-1.],integrality=np.r_[np.ones(M),0],bounds=Bounds(np.r_[np.zeros(M),-M],np.r_[np.ones(M),M]),constraints=cons,options=dict(time_limit=30,mip_rel_gap=1e-8))
 if res.x is None:raise RuntimeError(('NO_FEASIBLE_BALANCED_16',res.message,names))
 chosen=[names[i]for i in np.flatnonzero(res.x[:M]>.5)];assert len(chosen)==16
 receipt=dict(utc=utc(),selected_names=chosen,selection_stage='discovery0-3_only',confirmation_labels_accessed=False,selection_seed=2026102710,optimizer_message=res.message,conservative_discovery_gap=float(-res.fun/16),minimum_group_counts=dict(obstacle=4,plus=4,box=2,swap=2),candidates={n:dict(own_half_rate=float(own[i]),best_full_rate=float(q[i].max()),group=by[n]['group'])for i,n in enumerate(names)},source_labels_sha256=sha(HERE/'LABEL_TABLES_DISCOVERY.json'),scene_definitions={n:by[n]['definition_sha256']for n in chosen})
 write(HERE/'TRAIN_SCENES_FROZEN.json',receipt);return[by[n]for n in chosen]


def certificate():
 rows=[read(p)for p in(HERE/'certification').glob('*.json')];assert len(rows)==1000 and all(r['passed']for r in rows)
 checks=sum(r['matrix_checks']for r in rows);assert checks>=10000
 pairs=[read(p)for p in(HERE/'new_pairs/certificates').glob('*.json')];assert len(pairs)==48 and all(p['passed']for p in pairs)
 modules=['runtime_new.py','controllers_new.py','backend/swap_safety.py','fast_projection.py','accel_fold.py','np_threefry.py','accel_mode.py']
 copies=read(HERE/'TOOLING_COPY.receipt.json')['files'];replaced={r['source']:r['source_sha256']for r in copies if Path(r['target']).relative_to(HERE).as_posix()in modules}
 write(HERE/'ACCELERATION_CERTIFICATE.json',dict(passed=True,enabled_changes=[1,2,3,4],archive_replays=1000,new_scene_config_pairs=48,matrix_checks=checks,unit_rng=read(HERE/'RNG_CERTIFICATION.json'),modules_sha256={str(HERE/m):sha(HERE/m)for m in modules},original_modules_sha256=replaced,projection_original_sha256=sha(ROOT/'shared_control/hard_projection.py'),all_archive_records=rows,new_scene_records=pairs,original_elapsed_seconds=sum(r['original_elapsed_seconds']for r in rows),accelerated_elapsed_seconds=sum(r['accelerated_elapsed_seconds']for r in rows),exact_equivalence='shape,dtype,tobytes; all declared outcome fields identical'))
 merge('new_pairs')

def final_audit():
 stages=['new_pairs','capability','discovery','confirmation'];counts={};total=0
 with dbm.connect(True)as db:
  for stage_ in stages:
   cs=read(HERE/f'{stage_}_CASES.json');missing=[];uids=set()
   for c in cs:
    u=c['rollout_uid'];uids.add(u);r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(u,)).fetchone()
    if r is None:missing.append(u);continue
    assert not r['conflict_quarantined'];raw=result(stage_,c);assert raw['rollout_uid']==u and raw['success']==r['success']
    assert sha(raw['trace_path'])==raw['trace_sha256']
   counts[stage_]=dict(requests=len(cs),unique_uids=len(uids),missing=len(missing));assert not missing
   total+=len(uids)
 write(HERE/'DB_POSTFLIGHT.json',dict(passed=True,stages=counts,new_and_reused_records=total,only_append_new_rows=True))
 subprocess.run([PY,'-B',str(HERE/'audit_sources.py'),'end'],check=True)


def main():
 assert not(HERE/'USER_ANALYSIS_ONLY.receipt.json').exists(),'User requested analysis only; pipeline disabled'
 assert read(ROOT/'diagnostics/critic_pool_supplement_v1/COMPLETE.json')['status']=='COMPLETE'
 library=read(HERE/'SCENE_LIBRARY.json');geom=read(HERE/'GEOMETRY_REPORTS.json');scenes=[s for s in library if all(r['eligible']for r in geom[s['name']]['states'])]
 write(HERE/'SCENE_EXCLUSIONS.json',{s['name']:'geometric validity or witnessed direct single-robot route failed; no state replaced'for s in library if s not in scenes})
 log('Only own isolated files will be changed; aggregate CPU governor active.')
 # Archive certification started separately; block until it is complete.
 if not(HERE/'certify_LOCAL_COMPLETE.json').exists():
  while not(HERE/'certify_LOCAL_COMPLETE.json').exists():
   if(HERE/'CERTIFICATION_FAILED.json').exists():raise RuntimeError('certification failed, see receipt')
   time.sleep(10)
 log('1000 archive replays complete.')
 pairs_prepare(scenes)
 if not(HERE/'new_pairs_LOCAL_COMPLETE.json').exists():subprocess.run([PY,'-B',str(HERE/'launch_local.py'),'new_pairs'],cwd=HERE,check=True)
 certificate();log('Four exact acceleration changes certified.')
 active=gate(scenes);log(f'Capability gate passed {len(active)} scenes.')
 cs=allcases(active,list(range(4)));assert len(cs)<=30000;stage('discovery',cs);tables=labels('discovery',active,list(range(4)))
 if(HERE/'TRAIN_SCENES_FROZEN.json').exists():names=read(HERE/'TRAIN_SCENES_FROZEN.json')['selected_names'];chosen=[s for n in names for s in active if s['name']==n]
 else:chosen=select(active,tables)
 log('16 training scenes frozen before reading confirmation labels.')
 cs=allcases(chosen,list(range(8,24)));assert len(cs)<=40000;stage('confirmation',cs);labels('confirmation',chosen,list(range(8,24)))
 from statistics_search import main as statistics_main
 statistics_main();final_audit()
 from deliver_search import deliver
 deliver()
except_marker=HERE/'PIPELINE_FATAL.json'
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  write(except_marker,dict(utc=utc(),traceback=traceback.format_exc()));raise
