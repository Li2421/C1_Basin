import os,sys,json,hashlib,datetime,csv
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];os.environ['MPLCONFIGDIR']=str(HERE/'mplconfig');sys.path.insert(0,str(ROOT/'diagnostics/controller_loco_real_flow_v1/plot_vendor'))
import numpy as np
from scipy.special import betaln,logsumexp
from functools import lru_cache
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
assert hashlib.sha256((HERE/'PREREGISTRATION.md').read_bytes()).hexdigest()==json.load(open(HERE/'PREREGISTRATION.receipt.json'))['sha256'];assert json.load(open(HERE/'RECONSTRUCTION_GATE.json'))['passed']
@lru_cache(None)
def beta_gt(s,n,t,m):
 a=s+1;b=n-s+1;c=t+1;d=m-t+1;i=np.arange(a);return float(np.exp(logsumexp(betaln(c+i,b+d)-np.log(b+i)-betaln(1+i,b)-betaln(c,d))))
assert abs(beta_gt(0,0,0,0)-.5)<1e-12 and abs(beta_gt(1,1,0,0)-2/3)<1e-12
rng=np.random.default_rng(202610152300);ks=[1,2,4,8,16];results={};choices_store={};table=[]
for menu_name,menu in [('M',['F10','F15','S13_t1','S15_t2']),('all7',None)]:
 units=[]
 for f in sorted(HERE.glob('R*.npz')):
  z=np.load(f,allow_pickle=False);arms=z['arms'].tolist();names=z['names'].tolist();state=z['states'].tolist();order=np.argsort(state);assert len(order)==40;mi=[arms.index(a) for a in (menu or arms)];baseline=mi.index(arms.index('S15_t2'))
  for ci,name in enumerate(names):
   yy=z['success'][ci][:,mi];uu=z['unknown'][ci][:,mi];eval_y=yy[order[20:]];eval_u=uu[order[20:]];rates=eval_y.mean((0,2));upper=(eval_y+eval_u).mean((0,2));pools=[yy[order[:20],a,:][uu[order[:20],a,:]==0].flatten() for a in range(len(mi))];assert all(len(pool)>=16 for pool in pools);unit=f.stem+' / '+name;stats={};draws={};oracle=round(float(rates.max()),14);oracle_upper=float(upper.max());den=eval_y.shape[0]*eval_y.shape[2];success_count=eval_y.sum((0,2));bestcounts=int(success_count.max())
   for k in ks:
    # Independent per-arm without-replacement episode pools, shared across E/S.
    permutations=np.array([[rng.permutation(len(pool)) for pool in pools] for _ in range(1000)],dtype=object);wins=np.empty((1000,len(mi)),int)
    for j,pool in enumerate(pools):wins[:,j]=[int(pool[np.asarray(permutations[b,j],int)[:k]].sum()) for b in range(1000)]
    chosen={'D0':np.full(1000,baseline,int)};emp=wins.argmax(1);ties=(wins==wins.max(1)[:,None]).sum(1)>1;emp[ties]=baseline;chosen['E']=emp
    safe=np.full(1000,baseline,int)
    for b in range(1000):
     qualified=[j for j in range(len(mi)) if j!=baseline and beta_gt(int(wins[b,j]),k,int(wins[b,baseline]),k)>=.9]
     if qualified:safe[b]=max(qualified,key=lambda j:(wins[b,j],-j))
    chosen['S']=safe;se=np.empty(1000,int);exhaustions=0
    for b in range(1000):
     active=set(range(len(mi)));counts=np.zeros(len(mi),int);success=np.zeros(len(mi),int)
     for t in range(len(mi)*k):
      eligible=[j for j in active if counts[j]<len(pools[j])]
      if not eligible:eligible=[j for j in range(len(mi)) if counts[j]<len(pools[j])];exhaustions+=1
      j=min(eligible,key=lambda j:(counts[j],j));ix=int(permutations[b,j][counts[j]]);success[j]+=pools[j][ix];counts[j]+=1
      if all(counts[j]>0 for j in active):
       best=max(active,key=lambda j:((success[j]+1)/(counts[j]+2),j==baseline,-j));drop=[j for j in active if j!=best and beta_gt(int(success[j]),int(counts[j]),int(success[best]),int(counts[best]))<.1];active-=set(drop)
     se[b]=max(active,key=lambda j:((success[j]+1)/(counts[j]+2),j==baseline,-j))
    chosen['SE']=se;stats[str(k)]={};draws[str(k)]={}
    for method,ch in chosen.items():
     regret=(bestcounts-success_count[ch])/den;boundslo=np.maximum(0,oracle-upper[ch]);boundshi=np.maximum(0,oracle_upper-rates[ch]);ok=(bestcounts-success_count[ch])*100<=3*den;isc=success_count[ch]==bestcounts;row=dict(mean_regret=float(regret.mean()),p90_regret=float(np.quantile(regret,.9)),prob_regret_le3pp=float(ok.mean()),prob_oracle=float(isc.mean()),mean_unknown_bounds=[float(boundslo.mean()),float(boundshi.mean())],p90_unknown_bounds=[float(np.quantile(boundslo,.9)),float(np.quantile(boundshi,.9))]);stats[str(k)][method]=row;draws[str(k)][method]=ch.tolist();table.append(dict(menu=menu_name,unit=unit,k=k,method=method,**{key:row[key] for key in ('mean_regret','p90_regret','prob_regret_le3pp','prob_oracle')}))
   units.append(dict(unit=unit,names=name,source=f.stem,menu=[arms[i] for i in mi],calibration_state_uids=[state[i] for i in order[:20]],evaluation_state_uids=[state[i] for i in order[20:]],evaluation_rates=rates.tolist(),evaluation_upper=upper.tolist(),calibration_known_counts=[len(x) for x in pools],evaluation_unknown=int(eval_u.sum()),stats=stats));choices_store[menu_name+'/'+unit]=draws;print(menu_name,unit,flush=True)
 results[menu_name]=dict(units=units)
 for method in ('E','S','SE'):
  star=next((k for k in ks if all(u['stats'][str(k)][method]['prob_regret_le3pp']>=.9 for u in units)),None);results[menu_name][method+'_kstar']=star
 bad=[dict(unit=u['unit'],k=k,p90=u['stats'][str(k)]['S']['p90_regret']) for u in units for k in ks if u['stats'][str(k)]['S']['p90_regret']>.1];star=results[menu_name]['S_kstar'];contrastk=star or 16;diff=np.array([u['stats'][str(contrastk)]['S']['mean_regret']-u['stats'][str(contrastk)]['D0']['mean_regret'] for u in units]);br=np.random.default_rng(202610152301);boot=diff[br.integers(0,30,(5000,30))].mean(1);ci=np.quantile(boot,[.025,.975]).tolist();results[menu_name].update(F1='SUPPORTED' if star is not None and star<=8 else 'REJECTED',F2='SUPPORTED' if not bad else 'REJECTED',F3=('SUPPORTED' if ci[1]<0 else 'REJECTED') if star is not None else 'INCONCLUSIVE',F2_violations=bad,contrast=dict(k=contrastk,registered_kstar_available=star is not None,mean_difference=float(diff.mean()),CI=ci),hardest_units=[u['unit'] for u in units if u['stats']['16']['S']['prob_regret_le3pp']<.9])
 for metric,file,title in [('mean_regret','regret_curves','Mean evaluation regret'),('prob_regret_le3pp','probability_curves','P(regret <= 3pp)')]:
  fig,axes=plt.subplots(5,6,figsize=(23,17),sharex=True)
  for ax,u in zip(axes.flat,units):
   for method in ('D0','E','S','SE'):ax.plot(ks,[u['stats'][str(k)][method][metric] for k in ks],'-o',label=method,ms=3)
   ax.set_title(u['unit'].replace('flow_',''),fontsize=7);ax.set_xscale('log',base=2);ax.grid(alpha=.2)
   if metric.startswith('prob'):ax.axhline(.9,color='gray',ls='--');ax.set_ylim(0,1.05)
  axes.flat[0].legend(fontsize=8);fig.suptitle(title+' / '+menu_name);fig.tight_layout()
  for ext in ('png','pdf'):fig.savefig(HERE/(file+'_'+menu_name+'.'+ext),dpi=160)
  plt.close(fig)
(HERE/'results.json').write_text(json.dumps(results,indent=2)+'\n');(HERE/'SIMULATION_CHOICES.json').write_text(json.dumps(choices_store,separators=(',',':'))+'\n')
with (HERE/'UNIT_METRICS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
m=results['M'];star=m['S_kstar'];first=f"主菜单的安全方法 S 在 k∈{{1,2,4,8,16}} 中{'需要 '+str(4*star)+' 条有标签 episode（k*='+str(star)+'）' if star else '没有达到全部30单元可靠性门，故 k*>16、所需标签数>64，精确值未识别'}；F1 {m['F1']}，F2 {m['F2']}（所有单元/预算的 p90≤10pp），F3 {m['F3']}。这是已有前瞻标签的 DEVELOPMENT 模拟，不是新的实地部署确认。"
lines=['# Round23 cross-scene few-shot', '',first,'',f"F2 超阈值项 {len(m['F2_violations'])}；最难单元 {m['hardest_units']}。",f"S−D0 在 k={m['contrast']['k']} 的平均遗憾差 {100*m['contrast']['mean_difference']:.3f}pp，单元配对95%CI {[round(100*x,3) for x in m['contrast']['CI']]}；k*不存在时仅作描述。",'', '全部单位按 state_uid 排序前20校准/后20评估，无交叉；校准不采 unknown，评估同时保存上下界。次菜单、E与SE结果和逐单位概率见 results.json；两种菜单各有30面板PNG/PDF。', '', '相对 round8 的128条标签，仅能比较研究情境和效应量：这里场景族之间差异大，但所有30单元的统一可靠性门可能受低效应量单元限制，不能从平均表现推出统一预算。', '', '事实：F1/F2/F3 与逐单位指标见结果。解释：Beta安全门保留D0可减少低信息切换，但D0本身并非所有场景中的安全选择。建议更保守策略需要新预注册，本轮未实施。']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n');(HERE/'HANDOFF.md').write_text('# Handoff\n\n'+first+'\n\nAuthority prompt_round23_cross_scene_fewshot_20261015.md. Read PREREGISTRATION.md, reconstruction gate, results.json and REPORT.md. Inputs are ten immutable reconstructed cohorts copied from Round21; exact hashes in gate. No rollout/sharedDB/old asset writes. Reproduce with specified Python -B analyze.py, 1 CPU. SE exhaustion spends residual budget on remaining discarded candidates while final choice remains surviving set; no evaluation labels enter selection.\n');print(first,flush=True)
