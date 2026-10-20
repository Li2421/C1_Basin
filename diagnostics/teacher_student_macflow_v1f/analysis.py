from core import *
from guard import guard
from audit_utils import source_inventory
from shared_rollout_db.src.rollout_db import connect
from design import ARMS
G=guard()
FULL={'V0':'原学生（离线蒸馏、各自噪声；V0）','V1':'闭环重标注学生（各自噪声；V1）','V3':'闭环重标注加共享随机种子学生（V3）'}
def boot_indices(n):
 rng=np.random.default_rng(2026103032);return rng.integers(0,n,(5000,n)),rng.integers(0,32,(5000,n,32))
def summary(x):
 c,s=boot_indices(len(x));draw=x.mean(2)[c[:,:,None],s].mean((1,2));return dict(mean=float(x.mean()),CI95=np.quantile(draw,[.025,.975]).tolist())
def paired(a,b):
 if len(a)==0:return dict(status='INCONCLUSIVE',reason='No eligible paired teachers',n=0)
 low=(a==1).astype(float)-(b!=0).astype(float);high=(a!=0).astype(float)-(b==1).astype(float);lo=summary(low);hi=summary(high);return dict(n=len(a),lower=lo,upper=hi,CI_unknown_envelope=[lo['CI95'][0],hi['CI95'][1]],passes=lo['CI95'][0]>0)
def main():
 assert (HERE/'CHOICES_FROZEN.receipt.json').exists() and (HERE/'main_MERGE_RECEIPT.json').exists()
 ts=read(HERE/'TEACHERS.json');names=[r['name'] for r in ts];vlist=['V0','V1','V3'];shape=(14,32,2,7);data={v:{k:np.full(shape,-1,np.int8) for k in ['success','collision','timeout']} for v in vlist};teachers={k:np.full(shape,-1,np.int8) for k in ['success','collision','timeout']};index=[];eligible={'V0':list(range(14)),'V1':[],'V3':[]};choices={}
 for folder in ['teacher_student_macflow_v1','teacher_student_macflow_v1b','teacher_student_macflow_v1c']:
  path=D/folder/'FROZEN_CHOICES.json'
  if path.exists():choices.update(read(path)['controllers'])
 newchoices=read(HERE/'FROZEN_CHOICES.json')['controllers']
 with connect(True) as db:
  def fill(c,group,arm=None):
   arm=arm or c['arm']
   if arm not in ARMS:return
   n=c['base_controller_name'];i=names.index(n);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row and not row['conflict_quarantined'] and row['compatibility_quality']=='EXACT_REUSE';ix=(i,c['state_index'],c['seed'],ARMS.index(arm))
   for k in group:group[k][ix]=-1 if row['numerical_failure'] else row[k]
   index.append(dict(case=c,row=dict(row)))
  for c in read(D/'teacher_student_macflow_v1c/main_CASES.json'):
   if c['arm'] in ARMS:fill(c,data['V0'])
  for c in read(D/'teacher_student_macflow_v1d/fair_CASES.json'):
   if c.get('kind')=='student':fill(c,data['V0'])
  for c in read(HERE/'main_CASES.json'):
   i=c['student_index'];v=c['variant'];n=dict(c,base_controller_name=names[i]);fill(n,data[v]);eligible[v].append(i)
  for c in read(D/'teacher_student_macflow_v1/teacher_CASES.json'):fill(c,teachers)
  for c in read(D/'decentralized_pipeline_v1/main_CASES.json'):
   if c['pipeline']=='local' and c['arm'] in ['F10','F13','S300','S500']:fill(c,teachers)
 eligible={v:sorted(set(indices)) for v,indices in eligible.items()};table={};failures={};endpoints={};critic={};gaps={};lines=['| 学生全名 | 臂 | 成功率下界 [95% CI] | 成功率上界 | 碰撞率 | 超时率 | unknown |','|---|---|---|---|---|---|---|'];flines=['| 学生全名 | 类型 | 数量 | 占全部配对局 | 占两类停滞 | 各臂救回数量 |','|---|---|---:|---:|---:|---|']
 for v in vlist:
  ids=eligible[v];table[v]={};s=data[v]['success'][ids];col=data[v]['collision'][ids];tim=data[v]['timeout'][ids]
  if not ids:gaps[v]='没有通过两个门并完成主评估的学生';continue
  for j,a in enumerate(ARMS):
   low=summary((s[...,j]==1).astype(float));upper=summary((s[...,j]!=0).astype(float));r=dict(success_lower=low,success_upper=upper,collision=summary((col[...,j]==1).astype(float)),timeout=summary((tim[...,j]==1).astype(float)),unknown=int((s[...,j]<0).sum()),episodes=int(s[...,j].size));table[v][a]=r;lines.append(f"| {FULL[v]} | {a} | {low['mean']:.2%} {low['CI95']} | {upper['mean']:.2%} | {r['collision']['mean']:.2%} | {r['timeout']['mean']:.2%} | {r['unknown']} |")
  own=(tim[...,0]==1)&(col[...,0]==0);induced=(col[...,0]==1)&(tim[...,1]==1);failures[v]={};den=int(own.sum()+induced.sum())
  for kind,mask in [('自身停滞',own),('安全层导致的停滞',induced)]:
   rescue={ARMS[j]:int((mask&(s[...,j]==1)).sum()) for j in range(2,7)};r=dict(count=int(mask.sum()),fraction_all=float(mask.mean()),fraction_two_stall_types=float(mask.sum()/den) if den else None,rescued=rescue);failures[v][kind]=r;flines.append(f"| {FULL[v]} | {kind} | {r['count']} | {r['fraction_all']:.2%} | {r['fraction_two_stall_types']} | {rescue} |")
  if v!='V0':
   endpoints[v+'_deadlock_one']=paired(s[...,5],s[...,1]);endpoints[v+'_deadlock_two']=paired(s[...,5],s[...,6])
  # Descriptive best schedule minus best fixed, recomputed within every resample.
  low=(s==1).astype(float);upper=(s!=0).astype(float);ci,si=boot_indices(len(s))
  def bestdelta(a):
   r=a.mean(2)[ci[:,:,None],si].mean(2);return (r[...,4:6].max(-1)-r[...,2:4].max(-1)).mean(1)
  draw=bestdelta(low);endpoints[v+'_schedule']=dict(mean=float((low.mean((1,2))[:,4:6].max(-1)-low.mean((1,2))[:,2:4].max(-1)).mean()),CI95=np.quantile(draw,[.025,.975]).tolist(),upper_CI95=np.quantile(bestdelta(upper),[.025,.975]).tolist(),descriptive=True)
  critic[v]={};rate=low.mean((1,2));oracle=rate[:,2:6].max(-1)
  for method in ['A3_full','A0_full','G_E','S500']:
   picked=[]
   for i in ids:
    ch=choices[names[i]] if v=='V0' else newchoices[f'{v}__{i}'];picked.append(ARMS.index(ch[method]))
   selected=np.stack([s[k,...,j] for k,j in enumerate(picked)]);critic[v][method]=dict(mean_regret=float((oracle-(selected==1).mean((1,2))).mean()),always_S500_minus_selected=paired(s[...,5],selected))
  gaps[v+'_teacher_gap']={a:paired(s[...,j],teachers['success'][ids,...,j]) if np.all(teachers['success'][ids,...,j]>=0) else dict(status='GAP',reason='No complete known teacher evidence for this arm; no new teacher rollout') for j,a in enumerate(ARMS)}
 for v,b in [('V1','V0'),('V3','V1')]:
  ids=sorted(set(eligible[v])&set(eligible[b]));endpoints[v+'_competence']=paired(data[v]['success'][ids,...,0],data[b]['success'][ids,...,0]);endpoints[v+'_collision_difference']=paired(data[v]['collision'][ids,...,0],data[b]['collision'][ids,...,0])
 write(HERE/'REUSED_AND_NEW_DB_ROWS.json',index);write(HERE/'results.json',dict(utc=utc(),status='COMPLETE_WITH_DECLARED_GAPS' if gaps else 'COMPLETE',evidence={'V0':'DEVELOPMENT','teachers':'DEVELOPMENT','V1':'CONFIRMATORY','V3':'CONFIRMATORY'},eligible=eligible,table=table,endpoints=endpoints,failure_composition=failures,critic=critic,gaps=gaps,unknown_rule='preserved; lower0 upper1; paired conservative differences',resource_exception=read(HERE/'RESOURCE_EXCEPTION.json'),guard=guard()))
 (HERE/'PAPER_TABLE.md').write_text('\n'.join(lines)+'\n');(HERE/'FAILURE_COMPOSITION.md').write_text('\n'.join(flines)+'\n');np.savez_compressed(HERE/'analysis_arrays.npz',**{v+'_'+k:a for v,d in data.items() for k,a in d.items()},**{'teacher_'+k:a for k,a in teachers.items()},controller_names=names,arms=ARMS);source_inventory('imports_analysis.json')
if __name__=='__main__':main()
