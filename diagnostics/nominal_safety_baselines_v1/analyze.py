from core import *
from design import localdb
localdb()
from shared_rollout_db.src.rollout_db import connect
from shared_rollout_db.src.planner import preflight
import csv
METHODS=['nominal only','safety baseline','F10','F13','best fixed oracle','A3_full','S500','full menu oracle']
ARMS=['no_safety','hard_safety','F10','F13','F2','S200','S300','S500']
def ci(x):return np.quantile(x,[.025,.975],axis=0)
def policies(v,picks):
 # v [..., controller, arm], picks [..., controller]
 a3=np.take_along_axis(v,picks[...,None],axis=-1)[...,0]
 return np.stack([v[...,0],v[...,1],v[...,2],v[...,3],v[...,2:5].max(-1),a3,v[...,7],v[...,2:].max(-1)],axis=-1)
def csvwrite(name,rows):
 with (HERE/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def main():
 prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed'];p=read(HERE/'protocol.json');names=p['controller_order'];nc=len(names);shape=(nc,32,8,2)
 arrays={k:np.zeros(shape,np.int8) for k in ['success','unknown','collision','timeout','stagnation']};seen=np.zeros(shape,np.int8);unknown=[];mins=[]
 with connect(True) as db:
  for phase,folder,casefile in [('old',OLD,'main_CASES.json'),('new',HERE,'main_CASES.json')]:
   for c in read(folder/casefile):
    path=folder/'main/raw'/f"{c['rollout_uid']}.json"
    if not path.exists():
     record=db.execute('SELECT original_source_file FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert record;path=Path(record[0])
    r=read(path);i=names.index(c['base_controller_name']);s=c['state_index'];a=ARMS.index(c['arm']);seed=c['seed'];ix=(i,s,a,seed);seen[ix]+=1
    for field,key in [('success','success'),('unknown','numerical_failure'),('collision','collision'),('timeout','timeout')]:arrays[field][ix]=r[key]
    if phase=='new':
     assert r['case']==c and not r['jax_enable_x64'] and r['protocol_sha256']==sha(HERE/'protocol.json');assert sha(r['trace_path'])==r['trace_sha256'];rr=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert rr and not rr['conflict_quarantined'] and all(rr[k]==r[k] for k in ['success','collision','timeout','numerical_failure','episode_length'])
    z=np.load(r['trace_path']);positions=z['positions'];errors=np.linalg.norm(positions-z['goals'],axis=-1).sum(-1);arrays['stagnation'][ix]=len(errors)>100 and errors[-101]-errors[-1]<.05 and r['outcome']!='success'
    if len(z['swept_clearance']):mins.append(float(z['swept_clearance'].min()))
    if r['numerical_failure']:unknown.append(dict(source=phase,case=c,error=r['error']))
    if r['success']:assert not r['collision'] and (len(z['swept_clearance'])==0 or z['swept_clearance'].min()>=-1e-9)
 assert np.all(seen==1)
 legacy=np.load(OLD/'LABELS.npz');assert np.array_equal(arrays['success'][:,:,2:,:],legacy['success']) and np.array_equal(arrays['unknown'][:,:,2:,:],legacy['unknown'])
 np.savez_compressed(HERE/'LABELS.npz',**arrays,controller_names=names,state_uids=[s['state_uid'] for s in p['states']],arms=ARMS)
 write(HERE/'UNKNOWN_CASES.json',unknown)
 post=preflight(HERE/'main_PLANNED_ROLLOUTS.json');write(HERE/'POSTFLIGHT_OFFICIAL_CACHE.json',post)
 missing={(v['state_uid'],v['eta_uid'],v['controller_uid'],seed) for v in post['details'] for seed in v['missing_seeds']};expected={(r['case']['state_uid'],r['case']['eta_uid'],r['case']['controller_uid'],r['case']['seed_key']) for r in unknown if r['source']=='new'}
 assert missing==expected and post['summary']['ambiguous']==0
 write(HERE/'POSTFLIGHT.json',dict(utc=utc(),status='PASS',physical_records=2048,gate_records=8,reused_old_labels=6144,unknown_new=len(expected),missing_physical_records=0,planner_missing_are_retained_unknown=len(missing),local_DB_only=True,global_DB_unmodified=True))
 Y=arrays['success'].mean(-1);U=arrays['unknown'].mean(-1);choices=read(OLD/'FROZEN_CHOICES.json')['controllers'];picks=np.array([ARMS.index(choices[n]['A3_full']) for n in names]);rates=Y.mean(1);upper=(Y+U).mean(1)
 groups={'M':[i for i,n in enumerate(names) if n.startswith('M')],'P':[i for i,n in enumerate(names) if n.startswith('P')],'Flow':[i for i,n in enumerate(names) if n.startswith('base_')]}
 groups.update({n:[i] for i,n in enumerate(names)});results={};table=[];details=[]
 oldresults=read(OLD/'results.json')
 for gi,(g,ix) in enumerate(groups.items()):
  ix=np.asarray(ix);r=rates[ix];hi=upper[ix];pick=picks[ix];v=policies(r,pick);vh=policies(hi,pick);point=v.mean(0);point_hi=vh.mean(0)
  # Diagnostic rates use the same controller-level selected oracle arm as point.
  sel=np.stack([np.zeros(len(ix),int),np.ones(len(ix),int),np.full(len(ix),2),np.full(len(ix),3),2+r[:,2:5].argmax(-1),pick,np.full(len(ix),7),2+r[:,2:].argmax(-1)],axis=-1)
  metric={k:np.take_along_axis(x.mean((1,3))[ix],sel,axis=-1).mean(0) for k,x in arrays.items()}
  known_success=np.divide(metric['success'],1-metric['unknown'],out=np.full(8,np.nan),where=metric['unknown']<1)
  rng=np.random.default_rng(202610152901+gi);B=5000;boot=np.empty((B,8));bhi=boot.copy();diff=np.empty((B,2));dlo=diff.copy();dhi=diff.copy()
  for start in range(0,B,100):
   n=min(100,B-start);cc=rng.integers(0,len(ix),(n,len(ix))) if len(ix)>1 else np.zeros((n,1),int);original=ix[cc];ss=rng.integers(0,32,(n,len(ix),32));bv=Y[original[:,:,None],ss].mean(2);bh=(Y+U)[original[:,:,None],ss].mean(2);a=policies(bv,picks[original]);b=policies(bh,picks[original]);boot[start:start+n]=a.mean(1);bhi[start:start+n]=b.mean(1)
   diff[start:start+n]=(a[...,5,None]-a[...,:2]).mean(1);dlo[start:start+n]=(a[...,5,None]-b[...,:2]).mean(1);dhi[start:start+n]=(b[...,5,None]-a[...,:2]).mean(1)
  cis=ci(boot).T;ucis=ci(bhi).T;contrasts={}
  for j,baseline in enumerate(METHODS[:2]):contrasts[baseline]=dict(difference=float(point[5]-point[j]),CI=ci(diff)[:,j],identified_interval=[float(point[5]-point_hi[j]),float(point_hi[5]-point[j])],unknown_CI_envelope=[float(ci(dlo)[0,j]),float(ci(dhi)[1,j])])
  vals=[]
  for j,m in enumerate(METHODS):
   row=dict(group=g,method=m,confirmed_success_fraction=float(point[j]),CI_low=float(cis[j,0]),CI_high=float(cis[j,1]),identified_success_upper=float(point_hi[j]),unknown_CI_envelope_low=float(cis[j,0]),unknown_CI_envelope_high=float(ucis[j,1]),known_only_success=float(known_success[j]),unknown_rate=float(metric['unknown'][j]),collision_rate=float(metric['collision'][j]),timeout_rate=float(metric['timeout'][j]),stagnation_rate=float(metric['stagnation'][j]));vals.append(row);details.append(row)
  results[g]=dict(controllers=len(ix),methods=vals,A3_minus_baseline=contrasts,selected_arms={names[i]:{METHODS[j]:ARMS[sel[k,j]] for j in range(8)} for k,i in enumerate(ix)})
  if g in ['M','P','Flow']:
   old=oldresults['groups'][g];assert np.allclose(r[:,2:].mean(0),old['mean_rates'],atol=1e-14);assert abs(point[5]-old['policy_metrics']['A3_full']['mean_success'])<1e-14;assert abs(point[6]-old['policy_metrics']['S500']['mean_success'])<1e-14
   table.append(dict(category=g,**{m:f'{100*point[j]:.2f} [{100*cis[j,0]:.2f}, {100*cis[j,1]:.2f}]' for j,m in enumerate(METHODS)}))
 write(HERE/'results.json',dict(utc=utc(),groups=results,methods=METHODS,baseline_rollouts=2048,reused_round27=6144,gate_passed=True,unknown_new=sum(r['source']=='new' for r in unknown),unknown_old=sum(r['source']=='old' for r in unknown),bootstrap_replicates=5000,estimator='Confirmed success fraction is lower identification bound, unknown kept distinct; see upper bounds, known-only rates and unknown CI envelopes.',minimum_swept_clearance=min(mins),DB_scope='Local snapshot only; global index not updated.'))
 csvwrite('PAPER_TABLE.csv',table);csvwrite('METHOD_METRICS.csv',details)
 first=[]
 for g,label in [('M','MPC'),('P','势场'),('Flow','Flow')]:
  v=results[g]['A3_minus_baseline'];parts=[]
  for m in METHODS[:2]:
   r=v[m];parts.append(f"相对 {m} {100*r['difference']:+.2f}pp（95% CI [{100*r['CI'][0]:.2f}, {100*r['CI'][1]:.2f}]）")
  first.append(label+'：'+'，'.join(parts))
 text='；'.join(first)+'。以上是与 round27 一致的已确认成功占比差；unknown 单列，完整成功率的识别范围及 CI 包络见 results.json。\n'
 lines=[text,'','| 类别 | '+' | '.join(METHODS)+' |','| --- | '+' | '.join(['---']*8)+' |']
 for row in table:lines.append('| '+row['category']+' | '+' | '.join(row[m] for m in METHODS)+' |')
 lines+=['','表内为百分比 [95% CI]。5000 次 controller→state 配对分层 bootstrap，两 seed 留在同一 state 簇内；oracle 在每次重抽后重新取 controller 聚合成功率最大臂。最佳固定为 F10/F13/F2，完整菜单为全部六臂；oracle 是描述性事后上界，不是可部署策略。','',
 '碰撞率、超时率、数值失败（unknown）率、描述性停滞及逐 controller 八方法结果见 METHOD_METRICS.csv。unknown 未重跑、未归为已知失败；给出 confirmed-success 下界、完整成功率上界、known-only 比率与 unknown bootstrap 包络。timeout 不称 deadlock。两 seed 的确定性势场重复不视为两份独立未来随机性证据。','',
 '一致性门 8/8：覆盖 MPC、势场、全部 Flow；终止、步数与首动作 dtype/shape/bytes 完全相同，附加全位置轨迹核对见 CONSISTENCY_GATE.json。冻结 controller 直接加载并验证原 V2 收据和参数/checkpoint hash；未重新实现或调参。导入后及每次构造后恢复 x64=False，逐步断言。','',
 '历史审计、预注册 UTC/sha256 收据、2048 manifest 与 preflight/postflight 均保留。第3–8方法复用 round27 原标签，类别六臂与 A3/S500 点值已逐项断言一致；表中 CI 使用本轮统一 bootstrap 重新计算，未改写 round27 原报告。连续运动 swept clearance、完整位置/动作与真实终止保存在 main/traces；collision-free success 由原 simulator 判定。纯 Q 评估按预注册免视频。','',
 '**DB 写入范围限制：** 用户只授权新目录写文件，因此 authoritative shared DB 只读备份到本目录 rollout.sqlite；官方 preflight、append-only journal、单进程官方 merger、postflight 在副本运行。全局共享索引未更新；不得声称已完成全局 DB ingestion。没有修改历史资产，没有 git commit。','',
 '停滞只用于描述：未成功且最后100个实际步的总目标距离减少<0.05；正常等待或后退也可能满足，不提供死锁认证。A3 只沿用 round27 冻结选择，未使用本轮 baseline 标签训练、选择或调整。']
 diagnostic_lines=['碰撞率 / 超时率（%）：','','| 类别 | '+' | '.join(METHODS)+' |','| --- | '+' | '.join(['---']*8)+' |']
 for g in ['M','P','Flow']:
  diagnostic_lines.append('| '+g+' | '+' | '.join(f"{100*r['collision_rate']:.2f} / {100*r['timeout_rate']:.2f}" for r in results[g]['methods'])+' |')
 lines+=['']+diagnostic_lines
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'PAPER_TABLE.md').write_text('\n'.join(lines[2:2+2+3])+ '\n\n'+ '\n'.join(diagnostic_lines)+'\n\n成功率为已确认成功占比（%）[95% CI]；unknown 单列、不归为已知失败。完整识别区间与 CI 包络见 results.json。5000 次 controller→state 配对 bootstrap；两个 oracle 为每 controller 聚合事后最大值。\n')
 (HERE/'HANDOFF.md').write_text('Round29 nominal/safety baseline evaluation\n\n'+text+'\nAuthority: docs/prompt_round29_nominal_safety_baselines_20261015.md. Read PREREGISTRATION.md/receipt, FROZEN_INPUTS.json and HISTORY_AUDIT.json. Frozen controller code lives unchanged in ../other_controller_classes_v1; runtime.py loads it directly. 8-case consistency gate must PASS before run.py main. 2048 baseline records, exact round27 states/seed keys, no_safety/native nominal and hard_safety/native one projection. x64=False after imports and lazy constructions.\n\nEvidence: main/raw + main/traces; UNKNOWN_CASES.json retained without reruns. All 6144 legacy labels/choices reused, original class values asserted. Analyze without rollout: python -B analyze.py from this directory (use specified .venv-c1 Python). Outputs results.json, PAPER_TABLE.md/csv, METHOD_METRICS.csv. Bootstrap is paired controller then state, 5000; point fractions are confirmed success lower bounds to match round27, unknown envelopes and known-only rates included.\n\nOnly this directory was writable. rollout.sqlite is a read-only-source SQLite backup with new exact chain identities and official local preflight/journal/single-process merger/postflight. Global shared DB remains unchanged and requires separately authorized ingestion. All journal records and hashes are retained here for that purpose. No training/video/commit. Oracle is per-controller aggregate retrospective max, not per-rollout best and not deployable.\n')
 write(HERE/'FINAL_AUDIT.json',dict(utc=utc(),status='PASS_LOCAL_DELIVERY',global_DB_ingestion='not performed: new-directory-only scope',rollouts=2048,unknown_new=len(expected),round27_values_verified=True,report_sha256=sha(HERE/'REPORT.md'),results_sha256=sha(HERE/'results.json')))
 print(text,flush=True)
if __name__=='__main__':main()
