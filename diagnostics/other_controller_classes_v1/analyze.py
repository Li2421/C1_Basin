from core import *
from collections import Counter,defaultdict
from shared_rollout_db.src.rollout_db import connect
from shared_rollout_db.src.planner import preflight
import csv
sys.path.insert(0,str(D/'controller_loco_real_flow_v1/plot_vendor'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
qci=lambda x:np.quantile(x,[.025,.975]).tolist()
def verdict(lo,hi,positive=False):
 if positive:return 'SUPPORTED' if lo>0 else 'REJECTED' if hi<=0 else 'INCONCLUSIVE'
 return 'SUPPORTED' if hi<0 else 'REJECTED' if lo>=0 else 'INCONCLUSIVE'
def main():
 prereg();p=read(HERE/'main_protocol.json');names=p['controller_order'];arms=p['arms'];nc=len(names);cases=read(HERE/'main_CASES.json');Y=np.zeros((nc,32,6,2),np.int8);U=Y.copy();fail=defaultdict(list);unknown=[];clearances=[]
 with connect(True) as db:
  for phase in ['gate','pilot','main']:
   pp=read(HERE/f'{phase}_protocol.json')
   for c in read(HERE/f'{phase}_CASES.json'):
    r=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json");assert r['case']==c and r['protocol_sha256']==sha(HERE/f'{phase}_protocol.json') and not r['jax_enable_x64'];assert sha(r['trace_path'])==r['trace_sha256'];row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row and not row['conflict_quarantined'] and all(row[k]==r[k] for k in ['success','deadlock','timeout','collision','numerical_failure','episode_length'])
    if phase=='main':
     i=names.index(c['base_controller_name']);s=c['state_index'];a=arms.index(c['arm']);seed=c['seed'];Y[i,s,a,seed]=r['success'];U[i,s,a,seed]=r['numerical_failure'];z=np.load(r['trace_path']);assert len(z['positions'])==r['physical_steps']+1
     if len(z['swept_clearance']):clearances.append(float(z['swept_clearance'].min()))
     if r['numerical_failure']:unknown.append(dict(case=c,error=r['error']))
     elif not r['success']:fail[names[i]].append(dict(arm=c['arm'],outcome=r['outcome'],episode_length=r['episode_length'],state_uid=c['state_uid'],seed=seed))
 assert len(cases)==nc*32*6*2;assert sum(read(f)['new'] for f in (HERE/'main').glob('worker_*.json'))==len(cases)
 np.savez_compressed(HERE/'LABELS.npz',success=Y,unknown=U,controller_names=names,arms=arms,state_uids=[s['state_uid'] for s in p['states']]);write(HERE/'UNKNOWN_CASES.json',unknown);write(HERE/'FAILURE_RECORDS.json',fail)
 official=preflight(HERE/'main_PLANNED_ROLLOUTS.json');write(HERE/'POSTFLIGHT_OFFICIAL_CACHE.json',official);missing={(v['state_uid'],v['eta_uid'],v['controller_uid'],seed) for v in official['details'] for seed in v['missing_seeds']};expected={(v['case']['state_uid'],v['case']['eta_uid'],v['case']['controller_uid'],v['case']['seed_key']) for v in unknown};assert missing==expected and official['summary']['ambiguous']==0
 write(HERE/'POSTFLIGHT.json',dict(utc=utc(),status='PASS',main_rows=len(cases),gate_rows=16,pilot_rows=384,unique_task_records=16+384+len(cases),main_physical_executions=len(cases),gate_independent_repeats=16,unknown_main=int(U.sum()),raw_trace_DB_exact=True,actual_record_missing=0,official_planner_missing_are_retained_unknown=len(missing)))
 choice=read(HERE/'FROZEN_CHOICES.json')['controllers'];o4=read(HERE/'O4_CHOICES.json')['controllers'];policies=['A3_full','A3_fixed','A0_full','A0_fixed','G_E','S500','F10','O4'];selected=np.zeros((nc,len(policies)),int)
 for i,n in enumerate(names):
  for j,m in enumerate(policies):selected[i,j]=arms.index(o4[n]['action'] if m=='O4' and n in o4 else 'S500' if m=='O4' else choice[n][m])
 lo=Y.mean(-1);hi=(Y+U).mean(-1);rates=lo.mean(1);rateshi=hi.mean(1);groups={'M':[i for i,n in enumerate(names) if n.startswith('M') and n[1:].isdigit()],'P':[i for i,n in enumerate(names) if n.startswith('P') and n[1:].isdigit()],'Flow':[i for i,n in enumerate(names) if n not in o4]};groups['NEW']=groups['M']+groups['P'];results={};rows=[];bootstrap={}
 for gi,(group,ix) in enumerate(groups.items()):
  if not ix:results[group]=dict(status='INCONCLUSIVE',controllers=0);continue
  r=rates[ix];rh=rateshi[ix];fixed=r[:,:3].max(1);sched=r[:,3:].max(1);gain=sched-fixed;dom=sched>=fixed-.01-1e-14;domcertain=sched>=rh[:,:3].max(1)-.01-1e-14;dompossible=rh[:,3:].max(1)>=fixed-.01-1e-14;s500=r[:,arms.index('S500')];endpoint=r[:,:2].max(1);o2=s500>=endpoint-.01-1e-14;o2certain=s500>=rh[:,:2].max(1)-.01-1e-14;o2possible=rh[:,arms.index('S500')]>=endpoint-.01-1e-14;pref=np.sign(r[:,1]-r[:,0]).astype(int);prefcounts=Counter(pref.tolist());policy=r[np.arange(len(ix))[:,None],selected[ix]];policyhi=rh[np.arange(len(ix))[:,None],selected[ix]];reg=r.max(1)[:,None]-policy;rng=np.random.default_rng(202610152721+gi);B=5000;bgain=np.empty(B);bworst=bgain.copy();bbest=bgain.copy();breg=np.empty((B,len(policies)));bvalue=breg.copy();bO3=np.empty((B,3));bO4=np.empty((B,3))
  for start in range(0,B,100):
   n=min(100,B-start);cc=rng.integers(0,len(ix),(n,len(ix)));original=np.asarray(ix)[cc];ss=rng.integers(0,32,(n,len(ix),32));v=lo[original[:,:,None],ss].mean(2);vh=hi[original[:,:,None],ss].mean(2);bf=v[:,:,:3].max(2);bs=v[:,:,3:].max(2);bgain[start:start+n]=(bs-bf).mean(1);bworst[start:start+n]=(bs-vh[:,:,:3].max(2)).mean(1);bbest[start:start+n]=(vh[:,:,3:].max(2)-bf).mean(1);pick=selected[original];pv=np.take_along_axis(v,pick,axis=2);ph=np.take_along_axis(vh,pick,axis=2);bvalue[start:start+n]=pv.mean(1);breg[start:start+n]=(v.max(2)[:,:,None]-pv).mean(1)
   for output,a,b in [(bO3,policies.index('A3_full'),policies.index('A0_full')),(bO4,policies.index('O4'),policies.index('S500'))]:
    coef=np.eye(6)[pick[:,:,b]]-np.eye(6)[pick[:,:,a]];center=(coef*v).sum(2);gap=vh-v;output[start:start+n,0]=center.mean(1);output[start:start+n,1]=(center+(np.minimum(coef,0)*gap).sum(2)).mean(1);output[start:start+n,2]=(center+(np.maximum(coef,0)*gap).sum(2)).mean(1)
  bootstrap[group]=dict(gain=bgain,worst=bworst,best=bbest,O3=bO3,O4=bO4);rateci=[]
  # Mean arm CI from the same hierarchy, deterministic independent stream.
  arng=np.random.default_rng(202610152721+gi+100);armboot=[]
  for _ in range(5000):
   cc=arng.integers(0,len(ix),len(ix));ss=arng.integers(0,32,(len(ix),32));armboot.append(lo[np.asarray(ix)[cc,None],ss].mean((0,1)))
  for a in range(6):rateci.append(qci(np.asarray(armboot)[:,a]))
  res=dict(controllers=len(ix),names=[names[i] for i in ix],mean_rates=r.mean(0),mean_rates_upper=rh.mean(0),mean_rates_CI=rateci,mean_best_schedule_minus_fixed=float(gain.mean()),gain_CI=qci(bgain),gain_unknown_CI_envelope=[float(np.quantile(bworst,.025)),float(np.quantile(bbest,.975))],gain_unknown_lower_CI_bound_range=[float(np.quantile(bworst,.025)),float(np.quantile(bbest,.025))],dominance_count=int(dom.sum()),dominance_fraction=float(dom.mean()),dominance_unknown_bounds=[int(domcertain.sum()),int(dompossible.sum())],O2_count=int(o2.sum()),O2_fraction=float(o2.mean()),O2_unknown_bounds=[int(o2certain.sum()),int(o2possible.sum())],fixed_preference_counts=dict(prefcounts),fixed_preference_consistency=max(prefcounts.values())/len(ix),unknown=int(U[ix].sum()),policy_metrics={})
  for j,m in enumerate(policies):
   if m=='O4' and group=='Flow':continue
   res['policy_metrics'][m]=dict(mean_regret=float(reg[:,j].mean()),maximum_regret=float(reg[:,j].max()),mean_success=float(policy[:,j].mean()),success_upper=float(policyhi[:,j].mean()),mean_regret_CI=qci(breg[:,j]),success_CI=qci(bvalue[:,j]));rows.append(dict(group=group,method=m,**res['policy_metrics'][m]))
  res['A3_minus_A0']=dict(regret_difference=float((policy[:,policies.index('A0_full')]-policy[:,policies.index('A3_full')]).mean()),CI=qci(bO3[:,0]),unknown_CI_envelope=[float(np.quantile(bO3[:,1],.025)),float(np.quantile(bO3[:,2],.975))],unknown_upper_CI_bound_range=[float(np.quantile(bO3[:,1],.975)),float(np.quantile(bO3[:,2],.975))]);res['O4_minus_S500']=dict(regret_difference=float((policy[:,policies.index('S500')]-policy[:,policies.index('O4')]).mean()),CI=qci(bO4[:,0]));results[group]=res
 o1='INCONCLUSIVE';o2='INCONCLUSIVE';o3='INCONCLUSIVE';adequate=all(len(groups[k])>=4 for k in ['M','P'])
 if adequate:
  guaranteed_dom=all(results[k]['dominance_unknown_bounds'][0]/len(groups[k])>=.7 for k in ['M','P']);possible_dom=all(results[k]['dominance_unknown_bounds'][1]/len(groups[k])>=.7 for k in ['M','P']);ci=results['NEW']['gain_unknown_lower_CI_bound_range'];o1='SUPPORTED' if guaranteed_dom and ci[0]>0 else 'REJECTED' if not possible_dom or ci[1]<=0 else 'INCONCLUSIVE';guaranteed2=all(results[k]['O2_unknown_bounds'][0]/len(groups[k])>=.7 for k in ['M','P']);possible2=all(results[k]['O2_unknown_bounds'][1]/len(groups[k])>=.7 for k in ['M','P']);o2='SUPPORTED' if guaranteed2 else 'REJECTED' if not possible2 else 'INCONCLUSIVE';env=results['NEW']['A3_minus_A0']['unknown_upper_CI_bound_range'];o3='SUPPORTED' if env[1]<0 else 'REJECTED' if env[0]>=0 else 'INCONCLUSIVE'
 controller_rows=[]
 for i,n in enumerate(names):controller_rows.append(dict(controller=n,controller_class=p['base_profiles'][n]['payload']['controller_class'],rates={a:rates[i,j] for j,a in enumerate(arms)},rates_upper={a:rateshi[i,j] for j,a in enumerate(arms)},best_schedule_minus_fixed=float(rates[i,3:].max()-rates[i,:3].max()),O2=bool(rates[i,arms.index('S500')]>=rates[i,:2].max()-.01-1e-14),choices={m:arms[selected[i,j]] for j,m in enumerate(policies) if m!='O4' or n in o4}))
 # Historical FF evidence is only read after TT choices and all labels freeze.
 ff=np.load(D/'schedule_phenomenon_consolidated_v1/R10.npz');fn=ff['names'].tolist();fa=ff['arms'].tolist();comparison=[]
 for i in groups['Flow']:
  n=names[i];j=fn.index(n);fr=ff['success'][j].mean((0,2));frhi=(ff['success'][j]+ff['unknown'][j]).mean((0,2));av=ff['available'][j];fixed=[a for a,x in enumerate(fa) if x.startswith('F') and av[a]];sch=[a for a,x in enumerate(fa) if x.startswith('S') and av[a]];sharedfixed=['F10','F13'];sharedsched=['S300','S500'];tt=rates[i];comparison.append(dict(controller=n,FF_own_menu_gain=float(fr[sch].max()-fr[fixed].max()),TT_own_menu_gain=float(tt[3:].max()-tt[:3].max()),FF_O2=bool(fr[fa.index('S500')]>=max(fr[fa.index('F10')],fr[fa.index('F13')])-.01-1e-14),TT_O2=controller_rows[i]['O2'],FF_common_menu_gain=float(max(fr[fa.index(a)] for a in sharedsched)-max(fr[fa.index(a)] for a in sharedfixed)),TT_common_menu_gain=float(max(tt[arms.index(a)] for a in sharedsched)-max(tt[arms.index(a)] for a in sharedfixed)),same_checkpoint=True,FF_states=64,TT_states=32,FF_Q=4,TT_Q=2,paired_states=False))
 write(HERE/'FLOW_FF_TT_COMPARISON.json',comparison);result=dict(utc=utc(),O1=o1,O2=o2,O3=o3,groups=results,controllers=controller_rows,main_rollouts=len(cases),pilot_rollouts=384,unknown=int(U.sum()),failure_types={n:dict(Counter(v['outcome'] for v in vv)) for n,vv in fail.items()},first_numerical_error_steps=[v['error']['step'] for v in unknown],minimum_observed_swept_clearance=min(clearances) if clearances else None,evidence='Preregistered new TT evaluation; declared state-interval correction before any evaluation labels. O4 DEVELOPMENT LOCO.',state_draw_deviation='Original planned evaluation interval overlapped pilot by6; dedup failed before any evaluation run. Supplement preregistered deterministic corrected start pilot_start+16. Original no-redraw clause explicitly amended; no controller/threshold changes.',physical_dedup_scope=read(HERE/'main_DEDUP_AUDIT.json'),O2_scope='Forward schedule endpoint dominance proxy; reverse direction not in requested menu, no direct bidirectional direction test',no_video='Pure Q exception preregistered')
 write(HERE/'results.json',result)
 with (HERE/'POLICY_METRICS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
 write(HERE/'CONTROLLER_RATES.json',controller_rows)
 fig,axs=plt.subplots(1,3,figsize=(15,4))
 for ax,kind in zip(axs,['M','P','Flow']):
  rr=results[kind];ax.bar(range(6),rr['mean_rates']);ax.set_xticks(range(6),arms,rotation=30);ax.set_ylim(0,1.05);ax.set_title(kind+f" (n={rr['controllers']})")
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(HERE/f'class_arm_success.{ext}',dpi=180)
 plt.close(fig);fig,axs=plt.subplots(1,2,figsize=(11,4))
 for kind in ['M','P','Flow']:
  ix=groups[kind];axs[0].scatter([kind]*len(ix),[100*controller_rows[i]['best_schedule_minus_fixed'] for i in ix]);axs[1].scatter([kind]*len(ix),[float(controller_rows[i]['O2']) for i in ix])
 axs[0].axhline(0,color='black');axs[0].set_ylabel('best schedule minus best fixed (pp)');axs[1].set_ylabel('S500 endpoint dominance with1pp margin');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(HERE/f'O1_O2_distribution.{ext}',dpi=180)
 plt.close(fig);fig,ax=plt.subplots(figsize=(10,5));ms=['A3_full','A0_full','S500','G_E','O4'];x=np.arange(len(ms))
 for offset,kind in [(-.2,'M'),(.2,'P')]:ax.bar(x+offset,[100*results[kind]['policy_metrics'][m]['mean_regret'] for m in ms],.4,label=kind)
 ax.set_xticks(x,ms);ax.set_ylabel('mean regret pp');ax.legend();fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(HERE/f'cross_class_regret.{ext}',dpi=180)
 plt.close(fig)
 first=f'O1 **{o1}**，O2 **{o2}**，O3 **{o3}**。'+('在本轮Ring/TT及所测参数实例内，支持时间表现象跨非Flow类别。' if o1==o2=='SUPPORTED' else '本轮不能将Flow上的时间表优势推广为跨类别的一般结论。')+('Flow训练的A3显著优于A0。' if o3=='SUPPORTED' else 'Flow训练的A3跨类零样本优势未获支持。')
 lines=['# Round27 other controller classes','',first,'',f'能力门：M {len(groups["M"])}、P {len(groups["P"])}个实例保留，无替换、无性能调参；4个原Flow TT参照。主评估{len(cases)}条，unknown {int(U.sum())}保留未重跑。', '', '**预注册偏差披露：** 原pilot/eval seed区间误重叠6个state，去重门在任何评估执行前失败。已保留失败审计，并在补充预注册中以pilot_start+16修正评估起点，重新通过去重；这是对原“重复即停、不重抽”条款的明确修订，不能称原计划无偏差通过。controller参数、能力门、阈值及臂均未变。', '', '| class | best schedule−fixed pp | 95%CI pp | dominance | O2 | A3−A0 regret pp | O4−S500 regret pp |','| --- | ---: | --- | --- | --- | ---: | ---: |']
 for k in ['M','P','NEW','Flow']:
  v=results[k];lines.append(f"| {k} | {100*v['mean_best_schedule_minus_fixed']:.2f} | {[round(100*x,2) for x in v['gain_CI']]} | {v['dominance_count']}/{v['controllers']} | {v['O2_count']}/{v['controllers']} | {100*v['A3_minus_A0']['regret_difference']:.2f} | {format(100*v['O4_minus_S500']['regret_difference'], '.2f') if k!='Flow' else 'NA'} |")
 lines+=['', 'O1–O3最终判定包含unknown兼容范围；逐类/合并CI、点估计与边界见results.json。每类少于4实例时应记未定，本轮两类均6实例。O2按任务定义检验前向S500对两个端点的支配；未运行反方向，不能宣称直接证明双向排序共享。', '', 'O4为类内controller留一的DEVELOPMENT描述（仅对新M/P类，Flow的O4为NA）：5个同类source训练，R16选定step和原A3超参数固定，目标controller不用于训练、step或预处理；每折3fit，所有预测/checkpoint在o4目录。它不替代跨类零样本确认；若各controller都选同一臂，不能仅凭优于S500便声称学到了controller条件关系。', '', 'Flow参照与历史FF对照见FLOW_FF_TT_COMPARISON.json；既报各自菜单，又报共同F10/F13/S300/S500菜单。checkpoint相同但state与Q不同，不能把差值解释为同state配对的修正位置因果效应。', '', '已知失败与终止步见FAILURE_RECORDS；unknown含首个数值错误步，未记作已知失败。timeout不直接称deadlock。完整位置/动作/连续运动swept clearance证据保存在main/traces。', '', f"去重：所有{result['physical_dedup_scope']['DB_rows']}个DB state核对UID/content；物理三元组缺失/不匹配4机器人形状而无法额外核对者{result['physical_dedup_scope']['unresolved_physical']}，不能声称跨表示物理去重全覆盖。", '', 'TT一致性门16案例：本轮执行与独立历史rollout_tt参照的首动作、native终止/长度和全轨迹逐字节一致。原生v10入口及配置schema构造问题在pilot成功标签前修复并留收据；控制器参数不变。共享DB只执行新身份登记、preflight/journal/单进程merge/postflight；probe仅登记source_file，无伪造task标签。纯Q按预注册免视频，未改历史代码/资产，未commit。']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'HANDOFF.md').write_text('# Handoff\n\n'+first+'\n\nAuthority docs/prompt_round27_other_controller_classes_20261015.md. Read original prereg/receipt AND STATE_INTERVAL_CORRECTION supplemental prereg/receipt: original overlapping candidate interval was rejected before eval; corrected32 states disjoint. No controller grid/threshold changes. TT uses native Runtime.action(old), nominal only replaced. Code and parameters undercontrollers.py; implementationV2 and constructor/native-dispatch receipts explain pre-pilot engineering fixes. Capability12/12 retained. R5 and crossclass selections froze before all6144main labels. Officialcache missing identities exactly retainedunknown, not absent physical records; no reruns.\nReproduce withdesign/gate/pilot/gates/probes/freeze_choices/main/merge/o4/analyze scripts in this directory. O4 uses source-selected R16step,36LOCO fits, Q2 BCE normalization2, no held-controller training/selection. Raw/traces plus sharedDB exactidentity indexed. No videos pureQ exception; no historical assets edited. Reports include no-direct-reverse-direction and FFvsTT independentstates/menu caveats.\n');write(HERE/'FINAL_AUDIT.json',dict(utc=utc(),status='PASS',main_rows=len(cases),unknown=int(U.sum()),no_duplicate_main_execution=True,eval_labels_after_probe_and_choices_freeze=True,declared_preregistration_state_interval_correction=True,results_sha256=sha(HERE/'results.json'),report_sha256=sha(HERE/'REPORT.md')));print(first,flush=True)
if __name__=='__main__':main()
