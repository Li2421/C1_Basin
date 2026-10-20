from core import *
import csv
sys.path.insert(0,str(D/'controller_loco_real_flow_v1/plot_vendor'))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
TAGS=['T1','T2','T3','T4','T5']
def boot(pack,seed,contrasts,methods):
 lo,hi,W,av,fm,ns=pack;nc=len(lo);rng=np.random.default_rng(seed);nmethods=W.shape[1];B=5000;br=np.empty((B,nmethods));bv=br.copy();bvh=br.copy();bm=br.copy();bc={a+"__"+b:np.empty((B,2)) for a,b in contrasts}
 for start in range(0,B,50):
  n=min(50,B-start);cc=rng.integers(0,nc,(n,nc));ss=(rng.random((n,nc,lo.shape[1]))*ns[cc][:,:,None]).astype(int);mask=np.arange(lo.shape[1])[None,None,:]<ns[cc][:,:,None];rates=(lo[cc[:,:,None],ss]*mask[:,:,:,None]).sum(2)/ns[cc][:,:,None];rateshi=(hi[cc[:,:,None],ss]*mask[:,:,:,None]).sum(2)/ns[cc][:,:,None];value=np.einsum('bnma,bna->bnm',W[cc],rates);valuehi=np.einsum('bnma,bna->bnm',W[cc],rateshi);oracle=np.where(av[cc],rates,-np.inf).max(2);regret=oracle[:,:,None]-value;br[start:start+n]=regret.mean(1);bm[start:start+n]=regret.max(1);bv[start:start+n]=value.mean(1);bvh[start:start+n]=valuehi.mean(1)
  for a,b in contrasts:
   coef=W[cc][:,:,methods.index(b),:]-W[cc][:,:,methods.index(a),:];center=(coef*rates).sum(2);gap=rateshi-rates;bc[a+"__"+b][start:start+n,0]=(center+(np.minimum(coef,0)*gap).sum(2)).mean(1);bc[a+"__"+b][start:start+n,1]=(center+(np.maximum(coef,0)*gap).sum(2)).mean(1)
 return br,bm,bv,bvh,bc
def combine(zs):
 nc=sum(len(z['lo']) for z in zs);na=max(z['lo'].shape[2] for z in zs);smax=max(z['lo'].shape[1] for z in zs);nm=zs[0]['weights'].shape[1];lo=np.zeros((nc,smax,na));hi=lo.copy();W=np.zeros((nc,nm,na));av=np.zeros((nc,na),bool);fm=av.copy();ns=np.zeros(nc,int);offset=0
 for z in zs:
  n,s,a=z['lo'].shape;sl=slice(offset,offset+n);lo[sl,:s,:a]=z['lo'];hi[sl,:s,:a]=z['hi'];W[sl,:,:a]=z['weights'];av[sl,:a]=z['available'];fm[sl,:a]=z['fixedmask'];ns[sl]=s;offset+=n
 return lo,hi,W,av,fm,ns
qci=lambda x:np.quantile(x,[.025,.975]).tolist()
def main():
 prereg();assert read(HERE/'RECONSTRUCTION_GATE.json')['passed'];zs={t:dict(np.load(HERE/f'{t}_ANALYSIS.npz')) for t in TAGS};methods=zs['T1']['methods'].tolist();assert all(z['methods'].tolist()==methods for z in zs.values());points=read(HERE/'POINT_RESULTS.json');rows=[];boots={};comparisons={};allz=list(zs.values());contrasts=[('M4_full',b) for b in ['M0','M1','M2','M3_fixed','M3_full','M5']]+[(m+'_full',m+'_fixed') for m in ['M3','M4']]+[(m+f'_full_k{k}',m+f'_fixed_k{k}') for m in ['M6_E','M6_Thompson'] for k in [1,2,4,8,16]]+[('M4_fixed','M1')]
 for ti,t in enumerate(TAGS+['POOLED']):
  pack=combine([zs[t]]) if t!='POOLED' else combine(allz);lo,hi,W,av,fm,ns=pack;br,bm,bv,bvh,bc=boot(pack,202610152601+ti,contrasts,methods);boots[t]=(br,bm,bv,bvh);rates=lo.sum(1)/ns[:,None];rateshi=hi.sum(1)/ns[:,None];value=np.einsum('nma,na->nm',W,rates);valuehi=np.einsum('nma,na->nm',W,rateshi);oracle=np.where(av,rates,-np.inf).max(1);oraclehi=np.where(av,rateshi,-np.inf).max(1);fo=np.where(fm,rates,-np.inf).max(1)
  for j,m in enumerate(methods):
   existing=next((r for r in points if r['test']==t and r['method']==m),None)
   if existing is None:
    pp=[r for r in points if r['method']==m];weights=np.array([len(zs[r['test']]['names']) for r in pp]);reg=oracle-value[:,j];existing=dict(test=t,method=m,mean_regret=float(reg.mean()),maximum_regret=float(reg.max()),mean_success=float(value[:,j].mean()),mean_success_upper=float(valuehi[:,j].mean()),regret_unknown_bounds=[float((oracle-valuehi[:,j]).mean()),float((oraclehi-value[:,j]).mean())],catastrophe_count=sum(r['catastrophe_count'] for r in pp),expected_per_repeat_catastrophes=sum(r['expected_per_repeat_catastrophes'] for r in pp),target_labels_min=min(r['target_labels_min'] for r in pp),target_labels_max=max(r['target_labels_max'] for r in pp),target_labels_mean=float(np.average([r['target_labels_mean'] for r in pp],weights=weights)),fixed_oracle_mean_regret=float((fo-value[:,j]).mean()))
   row=dict(existing,mean_regret_CI=qci(br[:,j]),maximum_regret_CI=qci(bm[:,j]),mean_success_CI=qci(bv[:,j]),controllers=len(lo));rows.append(row)
  comp={}
  def contrast(a,b):
   ia=methods.index(a);ib=methods.index(b);delta=br[:,ia]-br[:,ib];unknown=[float(np.quantile(bc[a+'__'+b][:,0],.025)),float(np.quantile(bc[a+'__'+b][:,1],.975))];return dict(regret_difference=float((value[:,ib]-value[:,ia]).mean()),CI=qci(delta),unknown_CI_envelope=unknown,unknown_upper_CI_bound_range=[float(np.quantile(bc[a+'__'+b][:,0],.975)),float(np.quantile(bc[a+'__'+b][:,1],.975))])
  comp['B1']={b:contrast('M4_full',b) for b in ['M0','M1','M2','M3_fixed','M3_full','M5']};comp['B2']={m:contrast(m+'_full',m+'_fixed') for m in ['M3','M4']}
  for m in ['M6_E','M6_Thompson']:
   for k in [1,2,4,8,16]:comp['B2'][m+f'_k{k}']=contrast(m+f'_full_k{k}',m+f'_fixed_k{k}')
  comp['B2']['M4fixed_minus_M1']=contrast('M4_fixed','M1');comp['B3']={}
  target=next(r['mean_regret'] for r in rows if r['test']==t and r['method']=='M4_full')
  for m in ['M6_E','M6_Thompson','M7']:
   for menu in ['fixed','full']:
    candidates=[next(r for r in rows if r['test']==t and r['method']==f'{m}_{menu}_k{k}') for k in [1,2,4,8,16]];match=next((r for r in candidates if r['mean_regret']<=target+1e-14),None);comp['B3'][m+'_'+menu]=dict(k=int(match['method'].split('_k')[1]) if match else None,total_labels_min=match['target_labels_min'] if match else None,total_labels_max=match['target_labels_max'] if match else None,total_labels_mean=match['target_labels_mean'] if match else None,not_reached_lower_budget_bound=candidates[-1]['target_labels_mean'] if match is None else None,last_tested_label_range=[candidates[-1]['target_labels_min'],candidates[-1]['target_labels_max']],target_M4_full_regret=target,statement='matches M4 within tested grid' if match else 'not reached by16 per eligible arm')
  comparisons[t]=comp
 b1=comparisons['POOLED']['B1'];pointpass=all(v['CI'][1]<0 for v in b1.values());guaranteed=all(v['unknown_upper_CI_bound_range'][1]<0 for v in b1.values());possible=all(v['unknown_upper_CI_bound_range'][0]<0 for v in b1.values());status='SUPPORTED' if guaranteed else 'REJECTED' if not possible else 'INCONCLUSIVE';result=dict(utc=utc(),evidence='DEVELOPMENT',B1=status,B1_lower_point_CI_pass=pointpass,metrics=rows,comparisons=comparisons,bootstrap=dict(repetitions=5000,seed=202610152601,paired_controller_then_within_controller_state=True,conditional_on_observed_calibration_pool=True),menus='Labeled oracle uses all available arms. M3/M4/M7 restricted to R15 head support; T2 Freg mapped to F2 where exact eta2 (two controllers), while eta4 lacks a learned head; all labeled arms retained for M6 and oracle. Actual budgets differ and are reported.',Thompson='balanced k-per-arm collection followed by Beta posterior-sampling final recommendation; no adaptive allocation',unknown_policy='lower/upper retained, calibration unknown excluded, no reruns')
 write(HERE/'results.json',result)
 with (HERE/'METHOD_METRICS.csv').open('w') as f:
  keys=list(rows[0]);w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 zeromethod=['M0','M1','M2','M3_fixed','M3_full','M4_fixed','M4_full','M5'];paper_methods=zeromethod+[m+'_'+menu+'_k16' for m in ['M6_E','M6_Thompson','M7'] for menu in ['fixed','full']];grid=np.array([[next(r['mean_regret']*100 for r in rows if r['test']==t and r['method']==m) for t in TAGS] for m in zeromethod]);fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(grid,cmap='viridis');ax.set_xticks(range(5),TAGS);ax.set_yticks(range(8),zeromethod)
 for i in range(8):
  for j in range(5):ax.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',color='white' if grid[i,j]>grid.max()/2 else 'black')
 fig.colorbar(im,ax=ax,label='mean regret (pp)');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(HERE/f'method_regret_heatmap.{ext}',dpi=180)
 plt.close(fig);fig,axs=plt.subplots(2,3,figsize=(15,8))
 for ax,t in zip(axs.flat,TAGS+['POOLED']):
  for m in ['M6_E','M6_Thompson','M7']:
   for menu,style in [('fixed','--'),('full','-')]:
    rr=[next(r for r in rows if r['test']==t and r['method']==f'{m}_{menu}_k{k}') for k in [1,2,4,8,16]];ax.plot([r['target_labels_mean'] for r in rr],[100*r['mean_regret'] for r in rr],style,marker='o',label=m+' '+menu)
  m4=next(r for r in rows if r['test']==t and r['method']=='M4_full');ax.axhline(100*m4['mean_regret'],color='black',ls=':',label='M4 zero-shot');ax.set_title(t);ax.set_xlabel('target labels (mean/controller)');ax.set_ylabel('regret pp')
 axs.flat[-1].legend(fontsize=7);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(HERE/f'fewshot_curves.{ext}',dpi=180)
 plt.close(fig);fig,axs=plt.subplots(1,3,figsize=(14,4))
 for ax,m in zip(axs,['M3','M4','M6_E']):
  for menu in ['fixed','full']:
   key=m+'_'+menu+('_k16' if m.startswith('M6') else '');ax.plot(TAGS,[100*next(r['mean_regret'] for r in rows if r['test']==t and r['method']==key) for t in TAGS],marker='o',label=menu)
  ax.set_title(m);ax.set_ylabel('regret pp');ax.legend()
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(HERE/f'fixed_full_comparison.{ext}',dpi=180)
 plt.close(fig)
 lines=['# Round26 method benchmark','','| method | T1 regret pp | T2 | T3 | T4 | T5 | pooled | pooled95%CI |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
 for m in paper_methods:
  rr=[next(r for r in rows if r['test']==t and r['method']==m) for t in TAGS+['POOLED']];lines.append('| '+m+' | '+' | '.join(f"{100*r['mean_regret']:.2f}" for r in rr)+' | '+str([round(100*x,2) for x in rr[-1]['mean_regret_CI']])+' |')
 lines+=['','| pooled method | success % | max regret pp | catastrophes | labels/controller min–max |','| --- | ---: | ---: | ---: | --- |']
 for m in paper_methods:
  r=next(r for r in rows if r['test']=='POOLED' and r['method']==m);lines.append(f"| {m} | {100*r['mean_success']:.2f} | {100*r['maximum_regret']:.2f} | {r['catastrophe_count']} | {r['target_labels_min']}–{r['target_labels_max']} |")
 lines+=['',f'B1 **{status}**；下界点估计配对CI全通过={pointpass}。不满足严格优于M4比较的基线：'+str([b for b,v in b1.items() if v['CI'][1]>=0])+'。逐测试集及unknown敏感性见results.json。', '', 'M0固定参数；M1源平均固定策略；M2源平均增益调度；M3静态任务描述条件价值函数；M4交互上下文辨识式critic（类似PEARL的范式类比，不宣称实现PEARL）；M5结构化手工辨识规则；M6逐实例经验调参/后验采样；M7模型先验加快速残差适应。', '', '## 动作空间与方法（B2）','', '| test | M3 full−fixed pp | M4 full−fixed pp | fixed M4−M1 pp |','| --- | --- | --- | --- |']
 for t in TAGS+['POOLED']:
  def fmt(v):return f"{100*v['regret_difference']:.2f} {str([round(100*x,2) for x in v['CI']])}"
  b=comparisons[t]['B2'];lines.append(f"| {t} | {fmt(b['M3'])} | {fmt(b['M4'])} | {fmt(b['M4fixed_minus_M1'])} |")
 pooled=comparisons['POOLED']['B2'];lines+=['',f"合并结果中，M4完整菜单相对固定菜单的遗憾差为{100*pooled['M4']['regret_difference']:.2f}pp，95%CI {[round(100*x,2) for x in pooled['M4']['CI']]}；固定菜单内M4相对M1为{100*pooled['M4fixed_minus_M1']['regret_difference']:.2f}pp，CI {[round(100*x,2) for x in pooled['M4fixed_minus_M1']['CI']]}。动作菜单的改善幅度明显大于固定菜单内方法差异；这是描述性比较，不是因果可加分解。"]
 lines+=['', '## 达到M4零样本水平的最小标签预算（B3）','','| test | method/menu | k | total target labels min–max |','| --- | --- | ---: | --- |']
 for t in TAGS+['POOLED']:
  for m,v in comparisons[t]['B3'].items():lines.append(f"| {t} | {m} | {v['k'] if v['k'] is not None else '>16'} | {str(v['total_labels_min'])+'–'+str(v['total_labels_max']) if v['k'] is not None else '>16/arm; last tested '+str(v['last_tested_label_range'])} |")
 lines+=['','逐测试集未能以95%CI证明M4严格优于的基线：']
 for t in TAGS:
  lines.append(t+': '+', '.join(b for b,v in comparisons[t]['B1'].items() if v['CI'][1]>=0))
 lines+=['','**T2输入身份修正：** R10的FRESH_STATES.json是旧64state抽样，与实际main_CASES/protocol的64state集合不相交。最初盲预测与原R16加载逻辑使用了该旧输入；R16 DATA_AUDIT中记录的FRESH文件hash与当前旧文件一致。统一比较的身份门拦截此错配后，本轮保持M3/R16模型与所有超参数不变，仅对实际标签对应state重做前向推理，保留原冻结预测及T2_ALIGNMENT_CORRECTION收据。此输入修正发生在重建已读取标签之后，但不使用成功/unknown决定输入、拟合或规则。历史T2逐state指标需要这一限定；本轮T2数值使用正确输入。']
 lines+=['', '证据等级DEVELOPMENT：所有标签此前均已打开，本轮不构成新的前瞻确认。M3及方法超参数、R3和盲预测先冻结，再读取测试标签并复现R21全部14项重建门；未新增rollout、探测或共享DB写入。', '', '各测试controller按state_uid前半校准/后半评估；零样本方法只读输入，M6/M7只读校准标签。CI对controller和评估state分层配对重抽5000次，条件于既有校准池和1000次推荐分布，不估计新校准池总体不确定性。不同配方/同run兄弟作为controller单位，CI不能解释为独立配方总体区间。', '', 'T2的Freg仅3个controller有标签：两个实际为η2，已按执行身份映射至冻结F2预测头；另一个为η4，R15没有该训练头，仅保留在完整oracle及M6菜单，明确限制了此处菜单完全同构性。所有实际预算和可用臂见逐controller表。Thompson严格使用每臂k条平衡采集后做后验采样推荐，不是自适应分配的bandit轨迹。', '', '事实与解释分开：B2的配对差给出动作空间贡献与固定菜单内方法贡献的相对尺度；B3只是在已测离散预算中达到同一评估集M4点遗憾的最小预算，不是新部署保证。unknown上下界、最大遗憾、灾难数和标签预算均保留于主CSV/JSON。']
 (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'HANDOFF.md').write_text('# Handoff\n\nRound26 complete. Read REPORT/results/METHOD_METRICS/METHOD_CHOICES; evidence DEVELOPMENT. No new rollout/probe/DB writes. Reproduce: prepare_inputs.py -> static_features.py array6 -> train_m3.py CV7x3/full -> predict.py -> reconstruction_gate.py -> analyze.py -> summarize.py. Model and all hyperparameters froze before test outcomes; selected step '+str(read(HERE/'M3_SELECTED_STEP.json')['selected_step'])+'. Reconstruct gate14 exact; source-only R3 PCA and R16 A3 optimizer. T2 per-controller Freg=F2 aliases resolved; eta4 model-support gap and balanced posterior-sampling convention disclosed. Bootstrap conditional on calibration pool; all explicit seeds in PREREGISTRATION.\n');write(HERE/'FINAL_AUDIT.json',dict(utc=utc(),status='PASS',reconstruction_checks=14,tests=5,controllers=67,bootstrap=5000,simulations=1000,no_rollout_probe_or_shared_DB_write=True));print('Round26 complete B1',status,flush=True)
if __name__=='__main__':main()
