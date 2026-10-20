import os,sys,json,csv,hashlib,datetime
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig');sys.path.insert(0,str(ROOT/'diagnostics/controller_loco_real_flow_v1/plot_vendor'))
import numpy as np
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
stats=json.load(open(HERE/'STATISTICAL_RESULTS.json'));safe=json.load(open(HERE/'SAFETY_RESULTS.json'));local=json.load(open(HERE/'LOCAL_SWITCH_SAFETY.json'));inv=json.load(open(HERE/'DATA_INVENTORY.json'));rng=np.random.default_rng(202610152100)
# Correct the heterogeneous Freg uncertainty denominator to its available cohort.
z=np.load(HERE/'R10.npz');j=z['arms'].tolist().index('Freg');ix=np.flatnonzero(z['available'][:,j]);u=z['unknown'][ix,:,j,:];cc=rng.integers(0,len(ix),(5000,len(ix)));ss=rng.integers(0,u.shape[1],(5000,len(ix),u.shape[1]));boot=u[cc[:,:,None],ss].mean((1,2,3));safe['R10']['arms']['Freg']['unknown_CI']=np.quantile(boot,[.025,.975]).tolist();safe['R10']['arms']['Freg']['available_controllers']=len(ix)
for t,r in stats.items():
 if not isinstance(r,dict) or 'rates_lower' not in r:continue
 lo=np.array(r['rates_lower']);hi=np.array(r['rates_upper']);fixed=[i for i,a in enumerate(r['menu']) if a.startswith('F')];sched=[i for i,a in enumerate(r['menu']) if a.startswith('S')];r['A']['all_unknown_success_mean']=float((np.nanmax(hi[:,sched],1)-np.nanmax(hi[:,fixed],1)).mean());r['A']['sharp_unknown_mean_bounds']=[float((np.nanmax(lo[:,sched],1)-np.nanmax(hi[:,fixed],1)).mean()),float((np.nanmax(hi[:,sched],1)-np.nanmax(lo[:,fixed],1)).mean())]
# Conditional source choices, hierarchically bootstrap target scenes/variants and
# their states. Operators/seeds remain paired within state.
cross={}
for prefix in ('R13_','R20_'):
 tags=[t for t in inv if t.startswith(prefix)];rows=stats[prefix+'LOSO'];arms=inv[tags[0]]['arms'];selected={k:np.array([arms.index(next(r for r in rows if r['target']==t)[k]['choice']) for t in tags]) for k in ('fixed','full')};arr=[]
 for t in tags:
  z=np.load(HERE/(t+'.npz'));arr.append(z['success'].mean((0,3)))
 arr=np.array(arr);rate=arr.mean(1);N,S,A=arr.shape;boot={k:[] for k in ('fixed','full','difference')}
 for _ in range(5000):
  ids=rng.integers(0,N,N);st=rng.integers(0,S,(N,S));rr=arr[ids[:,None],st].mean(1);oracle=rr.max(1);rf=oracle-rr[np.arange(N),selected['fixed'][ids]];rs=oracle-rr[np.arange(N),selected['full'][ids]];boot['fixed'].append(rf.mean());boot['full'].append(rs.mean());boot['difference'].append((rs-rf).mean())
 cross[prefix]=dict(targets=tags,source_choices=rows,mean_regret={k:float(np.mean([r[k]['regret'] for r in rows])) for k in ('fixed','full')},CI={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()},bootstrap='scene/variant then state, conditional on original source choices')
(HERE/'CROSS_SCENE_RESULTS.json').write_text(json.dumps(cross,indent=2)+'\n')
# Verify all source assets used by reconstruction are unchanged.
hashes=json.load(open(HERE/'SOURCE_HASHES.json'));changed=[p for p,h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
count=sum(v['status']['raw'] for v in safe.values());collide=sum(a['collisions'] for v in safe.values() for a in v['arms'].values());unknown=sum(a['unknown'] for v in safe.values() for a in v['arms'].values());missing=sum(v['status']['trace_missing'] for v in safe.values());assert count==sum(int(np.prod(inv[t]['shape'][:2])*inv[t]['shape'][3]*np.array(inv[t]['available']).sum()/inv[t]['shape'][0]) for t in inv)
frozen_audit=json.load(open(HERE/'FROZEN_PREDICTION_AUDIT.json'));assert frozen_audit['passed']
result=dict(frozen_input_audit=frozen_audit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),analysis_evidence='DEVELOPMENT/retrospective analysis of prospective labels and frozen predictions',statistics=stats,safety=safe,local_switch=local,cross_scene=cross,reconstruction_gate=json.load(open(HERE/'RECONSTRUCTION_GATE.json')),source_assets_unchanged=not changed,total_audited_records=count,observed_collisions=collide,unknown=unknown,safety_definition='raw simulator collision event; minimum pair CENTER distance and wall NET clearance from discrete full-step physical traces; simulator swept combined clearance separately reported',limits=['R9 fixed endpoints lack2048 full traces; local paired windows unavailable there','Local windows condition on both episodes surviving through t+60; early termination excluded and counted','Zero collision bootstrap CI is conditional empirical data, not proof of zero prospective rare-event risk','R15 same-recipe seeds and Four best/final correlate; requested controller bootstrap is not independent-recipe sampling','Freg is heterogeneous F2/F4, available only3 R10 controllers','R18TRAIN and R9 have no compatible frozen critic predictions; ladder omits that rung rather than fitting a new model','Source mean selection CI conditional on point source choices; no selection-uncertainty guarantee'])
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
(HERE/'results.json').write_text(json.dumps(clean(result),indent=2,allow_nan=False)+'\n');(HERE/'FINAL_AUDIT.json').write_text(json.dumps(dict(utc=result['utc'],source_files_checked=len(hashes),changed=changed,raw_record_count=count,collisions=collide,unknown=unknown,no_rollout_training_or_DB_write=True),indent=2)+'\n')
# Sign consistency figure.
tags=[t for t,r in stats.items() if isinstance(r,dict) and 'D' in r];sign=[]
for t in tags:
 r=stats[t];row=next((z for z in r['D'] if z['arm']==r['default']),r['D'][0] if r['D'] else None)
 if row:sign.append((t,float(np.mean(np.array(row['E'])>=-1e-14)),float(np.mean(np.array(row['L'])>=-1e-14)),float(np.mean(np.array(row['N'])>1e-14))))
fig,ax=plt.subplots(figsize=(16,5));x=np.arange(len(sign));
for j,label in enumerate(('E>=0','L>=0','N>0')):ax.bar(x+(j-1)*.25,[r[j+1] for r in sign],.25,label=label)
ax.set_xticks(x,[r[0] for r in sign],rotation=70);ax.set_ylim(0,1.05);ax.legend();fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/('E_L_N_signs.'+ext),dpi=180)
plt.close(fig)
rows=[(t,a,z) for t,r in safe.items() for a,z in r['arms'].items()];fig,axes=plt.subplots(2,1,figsize=(18,8));
for j,(field,label) in enumerate([('pair_center_distance','Minimum pair center distance (p05)'),('wall_discrete_clearance','Minimum discrete wall clearance (p05)')]):axes[j].plot([i for i,(_,_,z) in enumerate(rows) if z[field]['p05'] is not None],[z[field]['p05'] for _,_,z in rows if z[field]['p05'] is not None],'.');axes[j].set_ylabel(label)
axes[1].set_xticks(range(len(rows)),[t+'/'+a for t,a,_ in rows],rotation=90,fontsize=4);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/('arm_clearance.'+ext),dpi=180)
plt.close(fig)
r15=stats['R15'];contrast=r15['A']['comparisons']['default_schedule-best_fixed'];r10=stats['R10'];four=stats['R19b'];p=lambda v:f'{100*v:.2f}';fmt=lambda v:'['+', '.join(p(x) for x in v)+']'
first=f"Ring R15（56 controller×32 state×Q4）最佳时间表相对最佳固定η平均提高 {p(r15['A']['mean'])}pp，分层95%CI {fmt(r15['A']['CI'])}；默认 S500 相对最佳固定提高 {p(contrast['difference'])}pp，CI {fmt(contrast['CI'])}。Controller信息的选择价值在完整菜单下显著收缩，但Four统一下界计数含unknown后不再是字面零遗憾，跨场景仍有机制冲突。逐臂审计{count}条历史记录未观测到碰撞（{collide}），同时保留{unknown}条数值unknown；这不能证明罕见安全风险为零，局部切换窗口的距离变化与缺失范围单列。"
lines=['# Round21 schedule consolidated analysis','',first,'','本轮为对已有前瞻采集标签及冻结预测的统一事后分析，全部结论为DEVELOPMENT，不是新前瞻确认。重建门14项通过，未新增rollout、探测、训练或共享DB写入。','', '| cohort | controller/state/seed | best schedule − best fixed pp | 95% CI pp | default − best fixed pp | 95% CI pp |','| --- | --- | ---: | --- | ---: | --- |']
for t in tags:
 r=stats[t];c=r['A']['comparisons']['default_schedule-best_fixed'];lines.append(f"| {t} | {r['n_controller']}/{r['n_state']}/{r['n_seed']} | {p(r['A']['mean'])} | {fmt(r['A']['CI'])} | {p(c['difference'])} | {fmt(c['CI'])} |")
lines+=['','## 选择与阶梯表','', '| cohort | F10 | best fixed | default schedule | frozen critic / G_E | full oracle |','| --- | ---: | ---: | ---: | ---: | ---: |']
for t in tags:
 f=stats[t]['F'];vals={r['policy']:p(r['success'])+'%' for r in f};crit=next((vals[r['policy']]+' ('+r['policy']+')' for r in f if r['policy'] not in ('F10','best_fixed','default_schedule','oracle')),'unavailable');lines.append(f"| {t} | {vals['F10']} | {vals['best_fixed']} | {vals['default_schedule']} | {crit} | {vals['oracle']} |")
lines+=['','所有级别的CI及相邻增量见ANALYSIS_F.csv；仅固定菜单选择也一律以完整菜单oracle计算遗憾，同时另列固定oracle遗憾。R16外部预测、R17 A0/A3/A3_all、R18 C0–C3/A0全部读取冻结文件，未重新训练。R15使用R16的recipe-held-out CV冻结预测，标为内部交叉验证；R9/R18TRAIN无可用冻结预测的级别如实省略。','', f"R10 controller留一源平均的固定/完整遗憾为 {p(r10['B']['source_fixed']['mean_regret'])}/{p(r10['B']['source_full']['mean_regret'])}pp，各自CI {fmt(r10['B']['source_fixed']['regret_CI'])}/{fmt(r10['B']['source_full']['regret_CI'])}。Four同run best/final成组留出后固定/完整遗憾为 {p(four['B']['source_fixed']['mean_regret'])}/{p(four['B']['source_full']['mean_regret'])}pp；原1597/1597为known-only口径，当前保留3 unknown下界默认时间表遗憾 {p(four['B']['default_schedule']['mean_regret'])}pp。",'', '跨场景：']
for key,r in cross.items():lines.append(f"- {key}: 固定/完整源菜单平均遗憾 {p(r['mean_regret']['fixed'])}/{p(r['mean_regret']['full'])}pp；CI {fmt(r['CI']['fixed'])}/{fmt(r['CI']['full'])}，完整−固定差CI {fmt(r['CI']['difference'])}。")
lines+=['','## 安全性、理论与切换敏感性','',f"全部{count}条记录的collision字段已核对：0个观测碰撞；数值unknown {unknown}，固定/时间表逐臂数量、率及CI见ANALYSIS_C.csv与SAFETY_RESULTS.json。min pair为机器人中心间距离，min wall为逐物理步净墙体余量；两者中位/p05/min和实际simulator的swept综合余量分开保存。R9固定端点2048条无完整trace，局部配对窗口无法检验；其余trace完整分辨率。窗口[t−20,t+60]只比较两侧都有完整窗口的同state/seed，早终止数量已登记；这是条件性局部检查，不能推出所有轨迹切换都无影响。",'', 'D 的逐时间表支配率、E/L排序一致率及N符号分布见ANALYSIS_D.csv与E_L_N_signs图；同η对/同场景排序共享不推出对任意η对的一般性质。跨场景S13/S15在同t1/t2的差与CI见statistics各cohort的cross_scene_order，Toy/DB方向冲突没有被时间表消除。R9双向曲线形状见R9_SWITCH_CURVES.json；平台与单调性为描述标签。', '', 'E 的最佳切换步、各固定步相对最佳时间表遗憾、≤2pp平台步集合及span见ANALYSIS_E.csv；网格是离散支持，不把不连续两端之间的所有时刻都判为平台。R15/R17/R18切换敏感性PNG/PDF已生成。', '', '## 解释、限制与来源','', '事实：时间表收益在多个Ring人口中存在；Four默认臂覆盖固定regime，跨场景仍失败。解释：阶段排序框架与这些η对的E/L结构相符，但不证明严格阶段可分，最佳时间表oracle还含事后选择；普遍性需要独立η对检验。', '', '原R17/R18/R19b历史主要数字采用known-only，而本轮统一把unknown按失败/成功给上下界，口径改变可导致小幅差异，不是标签被改动。R15同配方seed重复和Four同run best/final相关性使controller bootstrap的独立配方推断更弱；请勿把controller数量当成独立训练配方数量。source选择CI条件于原选择，未完全覆盖源argmax不确定性。所有源文件hash未变。', '', 'A–F代码、CSV、results.json以及PNG/PDF均在本目录。未写历史资产、共享DB或git。']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'HANDOFF.md').write_text('# Handoff\n\n'+first+'\n\nAuthority prompt_round21_schedule_consolidated_analysis_20261014.md. Read reconstruction gate before prereg/receipt, DATA_INVENTORY and SOURCE_HASHES; unified NPZ arrays controller,state,arm,seed have separate success/unknown and available masks. Historical R17/R18/R19b gate uses their known-only denominator; new analyses standardized lower/upper. Reproduce reconstruct.py then (only after prereg) analysis.py,safety.py,complete.py, each1CPU. No sharedDB writes or rollouts. results.json assembles A-F; CSV and PNG/PDF supplied. R9 fixed trace gaps and conditional source-selection CIs explicitly retained.\n');print('Round21 complete',count,collide,unknown,flush=True)
