from common import *
import stats as st
from metrics import features,collision_geometry,signature_codes
from audit_utils import source_inventory
import csv
G=guard()
ARMS=['nominal only','safety baseline','F10','F13','S300','S500','RHR-fair']
FULL={'DM':'分布式模型预测控制（DM）','AS':'A*规划控制器（AS）','G':'高斯行为克隆策略（G）','V0':'原离线蒸馏学生（V0）','ALL':'分布式黑盒控制器合并表（DEVELOPMENT）'}
AF={'nominal only':'只跑控制器，不加安全层或修正','safety baseline':'只加分布式安全层','F10':'全程第10组修正参数','F13':'全程第13组修正参数','S300':'前300步第13组、之后第10组修正时间表','S500':'前500步第13组、之后第10组修正时间表','RHR-fair':'逐族公平调参的右手规则','A3_full':'冻结的通用critic','A0_full':'不看控制器的critic','G_E':'冻结规则选择','oracle':'四修正臂事后最优（控制器级）'}
def main():
 prereg();p=read(HERE/'main_protocol.json');cap=read(HERE/'CAPABILITY_GATE.json');newchoices=read(HERE/'FROZEN_CHOICES.json')['controllers'];records=[];fm={}
 if not (HERE/'ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json').exists():seal('ANALYSIS_IMPLEMENTATION_FROZEN.receipt.json',[HERE/'analyze.py',HERE/'metrics.py',HERE/'stats.py',HERE/'mechanism_core.py'],before_opening_main_labels=True)
 # Hash-check and reprocess new traces using copied preregistered pure functions.
 metric_path=HERE/'EPISODE_METRICS.jsonl'
 with metric_path.open('w') as outf:
  for c in read(HERE/'main_CASES.json'):
   path=HERE/'main/raw'/f"{c['rollout_uid']}.json";r=read(path);assert sha(r['trace_path'])==r['trace_sha256'];rec=dict(group=c['family'],name=c['base_controller_name'],state_index=c['state_index'],seed=c['seed'],arm=c['arm'],rollout_uid=c['rollout_uid'],success=r['success'] if not r['numerical_failure'] else 0,collision=r['collision'] if not r['numerical_failure'] else 0,timeout=r['timeout'] if not r['numerical_failure'] else 0,unknown=r['numerical_failure'],outcome=r['outcome'],raw_path=str(path),raw_sha256=sha(path),trace_path=r['trace_path'],trace_sha256=r['trace_sha256']);records.append(rec)
   if not rec['unknown']:
    z=np.load(r['trace_path']);m=features(z['positions'],z['actions'],z['goals'],z['velocities']);angles=np.unwrap(np.arctan2(z['positions'][...,1],z['positions'][...,0]),axis=0);sign,_=signature_codes(angles-angles[0]);m['collision_geometry']=collision_geometry(z['positions'],z['actions'],sign);fm[c['rollout_uid']]=m;outf.write(json.dumps(clean(dict(m,rollout_uid=c['rollout_uid'],source_trace_sha256=r['trace_sha256'])),separators=(',',':'))+'\n')
   if len(records)%100==0:print('mechanisms',len(records),flush=True)
 # Historical student results and frozen metrics are read, not rerun.
 old=D/'teacher_student_line_closeout_v1';student=[r for r in read(old/'MANIFEST.json') if r['group']=='V0'];student_names={r['controller_index']:r['teacher_name'] for r in student}
 student_ids={r['rollout_uid'] for r in student}
 for f in sorted((old/'features').glob('shard*.jsonl')):
  for line in f.open():
   m=json.loads(line)
   if m['rollout_uid'] in student_ids:fm[m['rollout_uid']]=m
 for r in student:
  assert sha(r['raw_path'])==r['raw_sha256'];records.append(dict(r,name=r['teacher_name']))
 choices=dict(newchoices)
 for folder in ['teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c']:
  path=D/folder/'FROZEN_CHOICES.json'
  if path.exists():
   for n,c in read(path)['controllers'].items():choices[n]=c
 # Check A3 exactly matches the round30g frozen historical audit.
 for i,n in student_names.items():assert choices[n]['A3_full']==read(old/'FROZEN_CRITIC_REUSE.json')['V0'][str(i)]
 write(HERE/'ANALYSIS_MANIFEST.json',records);write(HERE/'HISTORICAL_FROZEN_CHOICES_REUSE.json',{n:choices[n] for n in student_names.values()})
 groups={f:[n for n in p['controller_order'] if n.startswith(f)] for f in ['DM','AS','G']};groups['V0']=[student_names[i] for i in sorted(student_names)];groups['ALL']=sum([groups[f] for f in ['DM','AS','G','V0']],[])
 outputs={};tables=[];fails=[];mechanisms={};det=[]
 for group,names in groups.items():
  if not names:
   outputs[group]=dict(status='INCONCLUSIVE',controllers=[]);continue
  C=len(names);shape=(C,32,2,len(ARMS));S=np.zeros(shape);U=np.ones(shape);B=np.zeros(shape);T=np.zeros(shape);M={};lookup={};ci={n:i for i,n in enumerate(names)}
  for r in records:
   if r['name'] not in ci:continue
   k=(ci[r['name']],r['state_index'],r['seed'],ARMS.index(r['arm']));assert k not in lookup;lookup[k]=r;S[k]=r['success'];U[k]=r['unknown'];B[k]=r['collision'];T[k]=r['timeout'];M[k]=fm.get(r['rollout_uid'])
  assert len(lookup)==C*32*2*7
  result=dict(full_name=FULL[group],evidence_level='DEVELOPMENT' if group in ['ALL','V0'] else 'CONFIRMATORY',controllers=names,independent_states=32,seeds=[0,1],status='INCONCLUSIVE' if group in ['DM','AS','G'] and C<4 else 'EVALUABLE',arms={},choices={},endpoints={})
  for ai,a in enumerate(ARMS):result['arms'][a]=dict(success=st.outcome_rate(S[...,ai],U[...,ai]),collision=st.outcome_rate(B[...,ai],U[...,ai]),timeout=st.outcome_rate(T[...,ai],U[...,ai]))
  for lab,b in [('one',1),('two',6)]:
   v=st.paired_success(S[...,5],U[...,5],S[...,b],U[...,b]);v['decision']='INCONCLUSIVE' if result['status']=='INCONCLUSIVE' else 'PASS' if v['CI_unknown_envelope'][0]>0 else 'NOT_PASS';result['endpoints'][lab]=v
  ids=[ARMS.index(a) for a in ['F10','F13','S300','S500']];oracle=np.asarray(ids)[S[...,ids].mean((1,2)).argmax(1)];selected={label:np.array([ARMS.index(choices[n][label]) for n in names]) for label in ['A3_full','A0_full','G_E']};selected['oracle']=oracle
  def pick(arr,idx):return np.stack([arr[i,:,:,j] for i,j in enumerate(idx)])
  for label,idx in selected.items():
   ss=pick(S,idx);uu=pick(U,idx);result['choices'][label]=dict(selected={n:ARMS[int(j)] for n,j in zip(names,idx)},success=st.outcome_rate(ss,uu),collision=st.outcome_rate(pick(B,idx),uu),timeout=st.outcome_rate(pick(T,idx),uu),minus_S500=st.paired_success(ss,uu,S[...,5],U[...,5]),minus_F10=st.paired_success(ss,uu,S[...,2],U[...,2]),oracle_regret=st.paired_success(pick(S,oracle),pick(U,oracle),ss,uu))
  outputs[group]=result
  for label,v in list(result['arms'].items())+list(result['choices'].items()):
   tables.append(dict(group=group,full_name=FULL[group],method=label,method_full=AF[label],controllers=C,states=32,episodes=C*64,evidence_level=result['evidence_level'],success_lower=v['success']['lower']['estimate'],success_upper=v['success']['upper']['estimate'],CI_low=v['success']['CI_unknown_envelope'][0],CI_high=v['success']['CI_unknown_envelope'][1],unknown=v['success']['unknown'],collision_lower=v['collision']['lower']['estimate'],collision_upper=v['collision']['upper']['estimate'],timeout_lower=v['timeout']['lower']['estimate'],timeout_upper=v['timeout']['upper']['estimate']))
  # Paired outcome transitions; numerical unknown stays a separate label.
  transitions={};nomT=(T[...,0]>0)&(U[...,0]==0)&(B[...,0]==0);induced=(B[...,0]>0)&(T[...,1]>0)&(U[...,0]==0)&(U[...,1]==0)
  for i in range(C):
   for s in range(32):
    for seed in range(2):
     n=lookup[i,s,seed,0];b=lookup[i,s,seed,1];key=('unknown' if n['unknown'] else n['outcome'])+' -> '+('unknown' if b['unknown'] else b['outcome']);transitions[key]=transitions.get(key,0)+1
  fail=dict(transitions=transitions,own_nominal_stall_total=int(nomT.sum()),own_stall_persisting_under_safety=int((nomT&(T[...,1]>0)).sum()),safety_induced_stall_total=int(induced.sum()),rescues={})
  for ai,a in enumerate(ARMS):
   fail['rescues'][a]=dict(own_nominal_stall=int((nomT&(S[...,ai]>0)).sum()),own_stall_persisting_under_safety=int((nomT&(T[...,1]>0)&(S[...,ai]>0)).sum()),safety_induced_stall=int((induced&(S[...,ai]>0)).sum()),baseline_timeouts=int(((T[...,1]>0)&(S[...,ai]>0)).sum()))
   fails.append(dict(group=group,arm=a,**fail['rescues'][a]))
  write(HERE/f'{group}_FAILURE_COMPOSITION.json',fail)
  mech={}
  for ai,a in enumerate(ARMS):
   vals=[m for k,m in M.items() if k[-1]==ai and m is not None];timeouts=[M[k] for k in M if k[-1]==ai and T[k]>0 and M[k] is not None];collisions=[M[k] for k in M if k[-1]==ai and B[k]>0 and M[k] is not None];cm={t:sum(t in m['collision_geometry']['types'] for m in collisions) for t in ['center','outer','agent']}
   # Sample-level conditional mechanism statistics keep controller and state clusters.
   def array(field,conditional=None):
    x=np.full((C,32,2),np.nan)
    for k,m in M.items():
     if k[-1]!=ai or m is None:continue
     if conditional is not None and not conditional[k]:continue
     x[k[:3]]=field(m)
    return x
   mech[a]=dict(n_known=len(vals),collision_objects_overlapping=cm,collision_geometry_verified=sum(m['collision_geometry']['geometry_verified'] for m in collisions),circulation_counts={str(t):sum(m['mode']==t for m in vals) for t in range(4)},circulation_legend={'0':'静止','1':'顺时针一致','2':'逆时针一致','3':'混合方向'},timeout_count=len(timeouts),timeout_metrics={f:st.mean(array(lambda m:m[f],T>0)) for f in ['moving_no_progress_robot_seconds','stall_robot_seconds','stall_runs','last_progress_seconds','final_unfinished']},timeout_any_strict_stall=st.mean(array(lambda m:int(m['stall_runs']>0),T>0)),timeout_mixed=st.mean(array(lambda m:int(m['mode']==3),T>0)),nominal_failure_time_seconds=st.mean(array(lambda m:m['total_time'],(S==0)&(U==0))))
  for a in ['F10','F13','S300','S500']:
   ai=ARMS.index(a);aa=np.full((C,32,2),np.nan);bb=aa.copy()
   for k,m in M.items():
    if m is None:continue
    if k[-1]==ai:aa[k[:3]]=m['early']['100']['progress']
    if k[-1]==1:bb[k[:3]]=m['early']['100']['progress']
   mech[a]['early_100_progress_minus_safety_m']=st.difference(aa,bb)
  mechanisms[group]=mech
  if group=='AS':
   for i,n in enumerate(names):
    for s in range(32):
     for ai,a in enumerate(ARMS):
      x=lookup[i,s,0,ai];y=lookup[i,s,1,ai];xx=np.load(x['trace_path']);yy=np.load(y['trace_path']);checks={k:xx[k].dtype==yy[k].dtype and xx[k].shape==yy[k].shape and xx[k].tobytes()==yy[k].tobytes() for k in ['positions','velocities','actions']};det.append(dict(name=n,state_index=s,arm=a,passed=all(checks.values()),checks=checks,unknown=x['unknown']+y['unknown']))
  print('analyzed',group,C,flush=True)
 write(HERE/'MECHANISMS.json',mechanisms);write(HERE/'ASTAR_SEED_IDENTITY.json',dict(passed=all(r['passed'] for r in det),pairs=len(det),results=det));write(HERE/'TABLE_DATA.json',tables)
 for filename,rows in [('PAPER_TABLE.csv',tables),('FAILURE_RESCUES.csv',fails)]:
  with (HERE/filename).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for group in groups:
  rows=[r for r in tables if r['group']==group];lines=[f'# {FULL[group]}','', '| 方法全名 | 成功率（未知界） | 配对分簇95%置信区间 | 碰撞率（未知界） | 超时率（未知界） |','|---|---:|---:|---:|---:|']
  for r in rows:
   pct=lambda x:f'{100*x:.2f}%';lines.append(f"| {r['method_full']}（{r['method']}） | {pct(r['success_lower'])}–{pct(r['success_upper'])} | {pct(r['CI_low'])}–{pct(r['CI_high'])} | {pct(r['collision_lower'])}–{pct(r['collision_upper'])} | {pct(r['timeout_lower'])}–{pct(r['timeout_upper'])} |")
  (HERE/f'PAPER_TABLE_{group}.md').write_text('\n'.join(lines)+'\n')
 write(HERE/'results.json',dict(utc=utc(),status='ANALYZED',groups=outputs,capability=cap,rhr_selected=read(HERE/'RHR_SELECTED.json'),mechanisms_path=str(HERE/'MECHANISMS.json'),bootstrap=dict(draws=5000,seed=2026101707,scheme='controller then state; paired, seed cluster retained'),source_manifest_sha256=sha(HERE/'ANALYSIS_MANIFEST.json'),unknown_retained=True,guard=guard()))
 source_inventory('imports_analysis.json');seal('ANALYSIS_COMPLETE.receipt.json',[HERE/'results.json',HERE/'MECHANISMS.json',HERE/'TABLE_DATA.json',HERE/'ANALYSIS_MANIFEST.json',HERE/'ASTAR_SEED_IDENTITY.json'],records=len(records),guard=guard())
if __name__=='__main__':main()
