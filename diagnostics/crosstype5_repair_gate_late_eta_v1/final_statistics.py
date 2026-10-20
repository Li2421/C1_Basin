"""Development G1--G6, paired type/controller/state bootstrap and all 19 axis rows."""
from common import *
from candidate_menu import METHODS
from catalog_names import full_type
import csv

def summary(values):return dict(point=float(values[0]*100),CI95=np.quantile(values[1:]*100,[.025,.975]))
def type_mean(values,types,mask):
    tt=sorted(set(types[mask].tolist()));return float(np.mean([values[mask&(types==t)].mean() for t in tt])) if tt else None

def bootstrap(arrays,types,mask,rng):
    tt=sorted(set(types[mask].tolist()));output=[]
    if not tt:return None
    for rep in range(5001):
        means=[]
        for t in tt if rep==0 else rng.choice(tt,len(tt),replace=True):
            pool=np.flatnonzero(mask&(types==t));cc=pool if rep==0 else rng.choice(pool,len(pool),replace=True);v=[]
            for ci in cc:
                ss=np.arange(arrays.shape[1]) if rep==0 else rng.integers(arrays.shape[1],size=arrays.shape[1]);v.append(arrays[ci,ss].mean(0))
            means.append(np.mean(v,0))
        output.append(np.mean(means,0))
    return np.asarray(output)

def main():
    with np.load(HERE/'NEW_EVAL_BUNDLE.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    label=d['labels'];uids=d['uids'];types=d['types'];names=d['names'];assert d['methods'].tolist()==list(METHODS)
    lower=(label==1).mean(-1);upper=(label!=0).mean(-1);safety=lower[:,:,6].mean(1);main_rate=lower[:,:,0].mean(1);high=safety>=.98;low=safety<.9
    # Preserve exact UID cancellation in all ordinary paired unknown bounds.
    bound=[];comparators=[6,1,2,3,4,5]
    for b in comparators:
        same=uids[:,:,0]==uids[:,:,b]
        ll=np.where(same,0,(label[:,:,0]==1).astype(float)-(label[:,:,b]!=0).astype(float)).mean(-1)
        uu=np.where(same,0,(label[:,:,0]!=0).astype(float)-(label[:,:,b]==1).astype(float)).mean(-1)
        bound.extend([ll,uu])
    arrays=np.concatenate([lower,upper,np.stack(bound,-1)],-1);rng=np.random.default_rng(202610195106)
    choices=read(HERE/'NEW_EVAL_CHOSEN_ACTIONS.json')['states'];fractions={m:{k:np.zeros((74,24)) for k in ('no_repair','progress','冻结提议对','时间表生成器提议对','切换时重查')} for m in METHODS[:6]}
    for row in choices:
        i=list(names).index(row['controller']);j=row['state_index']
        for method in METHODS[:6]:
            a=row['methods'][method];meta=row['selected_metadata'][method];zero=bool(np.all(np.asarray(a['eta_a'])==0) and np.all(np.asarray(a['eta_b'])==0));fractions[method]['no_repair'][i,j]=zero;fractions[method]['progress'][i,j]=a['switch_mode']=='progress'
            for source in ('冻结提议对','时间表生成器提议对','切换时重查'):fractions[method][source][i,j]=source in meta['origins'] and not zero
    groups={t:types==t for t in sorted(set(types.tolist()))};groups.update({'类型等权合并':np.ones(74,bool),'只加安全投影 ≥ 98% 的 controller':high,'只加安全投影 < 90% 的 controller':low});results={}
    for group,mask in groups.items():
        print('Fresh bootstrap',group,int(mask.sum()),flush=True);samples=bootstrap(arrays,types,mask,rng)
        if samples is None:results[group]=dict(controllers=0,gap='Empty observed subgroup',methods={},comparisons={});continue
        row=dict(controllers=int(mask.sum()),types=len(set(types[mask].tolist())),methods={},comparisons={},events={},choice_fractions={})
        for i,m in enumerate(METHODS):
            row['methods'][m]=dict(**summary(samples[:,i]),unknown_upper=summary(samples[:,9+i]))
            row['events'][m]={k:100*type_mean(d[k][:,:,i].mean((1,2)),types,mask) for k in ('collision','timeout','unknown')}
        for index,b in enumerate(comparators):
            row['comparisons'][METHODS[b]]=dict(**summary(samples[:,0]-samples[:,b]),unknown_bounds=[float(samples[0,18+2*index]*100),float(samples[0,19+2*index]*100)],unknown_lower_CI95=np.quantile(samples[1:,18+2*index]*100,[.025,.975]),unknown_upper_CI95=np.quantile(samples[1:,19+2*index]*100,[.025,.975]),identical_UID_uncertainty_cancelled=True)
        row['comparisons']['较好的人定双 η 时间表']=dict(**summary(samples[:,0]-np.maximum(samples[:,7],samples[:,8])),best_human_reselected_each_bootstrap=True,unknown_bounds_conservative=[100*float(samples[0,0]-max(samples[0,16],samples[0,17])),100*float(samples[0,9]-max(samples[0,7],samples[0,8]))])
        row['harm_count']=int(np.sum(mask&(safety-main_rate>=.1-1e-12)))
        row['choice_fractions']={m:{k:type_mean(v.mean(1),types,mask) for k,v in stats.items()} for m,stats in fractions.items()}
        # Conditional low-Safety subgroup is a descriptive warning, not a claim
        # that every no-repair decision in that group was certainly incorrect.
        mistaken=fractions[METHODS[0]]['no_repair']*(lower[:,:,1]>lower[:,:,6])
        row['mistaken_no_repair_fraction_vs_observed_direct_choice']=type_mean(mistaken.mean(1),types,mask)
        results[group]=row
    hh=results['只加安全投影 ≥ 98% 的 controller'];ll=results['只加安全投影 < 90% 的 controller'];allr=results['类型等权合并'];coverage=read(HERE/'COVERAGE_RESULTS.json')
    goals=dict(G1=dict(available=hh['controllers']>0,controllers=hh['controllers'],gain=hh['comparisons'].get(METHODS[6]),harm_count=hh.get('harm_count'),no_repair_fraction=hh.get('choice_fractions',{}).get(METHODS[0],{}).get('no_repair'),achieved=hh['controllers']>0 and hh['comparisons'][METHODS[6]]['point']>=-1 and hh['harm_count']==0),G2=dict(available=ll['controllers']>0,controllers=ll['controllers'],gain=ll['comparisons'].get(METHODS[1]),no_repair_fraction=ll.get('choice_fractions',{}).get(METHODS[0],{}).get('no_repair'),mistaken_no_repair_fraction=ll.get('mistaken_no_repair_fraction_vs_observed_direct_choice'),achieved=ll['controllers']>0 and ll['comparisons'][METHODS[1]]['CI95'][0]>-1),G3={m:ll['comparisons'].get(m) for m in METHODS[2:5]},G4=allr['comparisons']['较好的人定双 η 时间表'],G5={g:results[g]['comparisons'].get(METHODS[5]) for g in ('类型等权合并','只加安全投影 ≥ 98% 的 controller','只加安全投影 < 90% 的 controller')},G6=coverage['groups']['类型等权合并']['comparisons'])
    goals['G4']['achieved']=bool(goals['G4']['CI95'][0]>-1)
    for comparison in goals['G6'].values():comparison['CI_lower_positive']=bool(comparison['CI95'][0]>0)
    result=dict(utc=utc(),evidence='DEVELOPMENT',bootstrap=5000,seed=202610195106,groups=results,goals=goals,observed_safety_subgroup_membership_fixed=True,low_group_mistake_definition='No-repair chosen and evaluated direct-argmax choice beats Safety for that state, empirical seed0/1; unknown not asserted false',no_unseen_confirmatory_claim=True)
    write(HERE/'FINAL_RESULTS.json',result);axis_tables(result)
    with (HERE/'CONTROLLER_RESULTS.csv').open('w') as f:
        fields=['controller','type',*METHODS,'harm_ge10pp','no_repair_fraction'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,n in enumerate(names):w.writerow(dict(controller=str(n),type=full_type(types[i]),**{m:100*float(lower[i,:,j].mean()) for j,m in enumerate(METHODS)},harm_ge10pp=int(safety[i]-main_rate[i]>=.1-1e-12),no_repair_fraction=float(fractions[METHODS[0]]['no_repair'][i].mean())))
    development_report()
    seal('FINAL_STATISTICS.receipt.json',[HERE/n for n in ('FINAL_RESULTS.json','AXIS_TABLE.md','AXIS_TABLE.csv','EVENT_AND_CHOICES.json','CONTROLLER_RESULTS.csv','DEVELOPMENT_VALIDATION.json','final_statistics.py','NEW_EVAL_BUNDLE.receipt.json','COVERAGE_STATISTICS.receipt.json')],all_development=True)

def axis_tables(result):
    base=read(HERE/'BASE_PROTOCOL.json');all_types=sorted(set(base['types'].values()));groups=[*all_types,'类型等权合并','只加安全投影 ≥ 98% 的 controller','只加安全投影 < 90% 的 controller'];lines=['# 19个已使用类型的开发检验','','本轮主方法 = '+METHODS[0]+'。成功率 % [95% CI]；另8个学习型类型按prompt不做新state评估，N/A。','','| 类型 / 分组 | '+' | '.join(METHODS)+' |','|---|'+'---|'*9]
    with (HERE/'AXIS_TABLE.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['类型 / 分组',*METHODS,'证据等级']);writer.writeheader()
        for label in groups:
            row=result['groups'].get(label,{});record={'类型 / 分组':full_type(label),'证据等级':'DEVELOPMENT'};cells=[]
            for m in METHODS:
                r=row.get('methods',{}).get(m);cell='N/A' if r is None else f"{r['point']:.2f} [{r['CI95'][0]:.2f}, {r['CI95'][1]:.2f}]";record[m]=cell;cells.append(cell)
            writer.writerow(record);lines.append('| '+full_type(label)+' | '+' | '.join(cells)+' |')
    lines+=['','碰撞率、超时率、unknown、伤害数、判为不修正比例、各η来源与按进度切换选择比例见 EVENT_AND_CHOICES.json；每controller结果见 CONTROLLER_RESULTS.csv。','', 'Transformer行为克隆策略、环形交换场景流匹配策略与集中式模型预测路径积分控制均按本轮74个controller的冻结身份清单评估，分别列出。另8个未在新state清单中的学习型类型标N/A，不混入旧state结果。']
    (HERE/'AXIS_TABLE.md').write_text('\n'.join(lines)+'\n');write(HERE/'EVENT_AND_CHOICES.json',{k:{v:r.get(v) for v in ('controllers','events','harm_count','choice_fractions','mistaken_no_repair_fraction_vs_observed_direct_choice')} for k,r in result['groups'].items()})

def development_report():
    folds={};allrows=[]
    for path in sorted((HERE/'gate_fits').glob('*/DEVELOPMENT.json')):
        rows=read(path);typ=path.parent.name;allrows.extend(rows);confusion={}
        for method in (METHODS[0],METHODS[5]):
            matrix=np.zeros((2,2),int);unknown=0
            for r in rows:
                truth=r['truth_no_repair']
                if truth<0:unknown+=1
                else:matrix[truth,int(not r['details'][method]['repair'])]+=1
            confusion[method]=dict(truth_rows='0=repair,1=no repair',prediction_columns='0=repair,1=no repair',matrix=matrix,accuracy=float(np.trace(matrix)/matrix.sum()) if matrix.sum() else None,supported_states=int(matrix.sum()),unsupported=unknown)
        success={}
        for m in METHODS[:6]:
            per=[];scored=0
            for name in sorted({r['controller'] for r in rows}):
                value=[r['known_success'][m] for r in rows if r['controller']==name and r['known_success'][m] is not None];scored+=len(value)
                if value:per.append(float(np.mean(value)))
            success[m]=dict(success=float(np.mean(per)) if per else None,scored_states=scored,unscored_fraction=1-scored/len(rows))
        folds[typ]=dict(confusion=confusion,chosen_success=success)
    with np.load(HERE/'DEVELOPMENT_PROBABILITIES.npz',allow_pickle=False) as z:p=z['full']
    from train_critic import load_data
    from critic_model import metrics
    d=load_data();ranking=metrics(p,d['labels'],d['types'],sorted(set(d['types'].tolist())),d['candidate_indices'])
    # Coverage proposal ranking is measured only after model selection, never fed back.
    with np.load(HERE/'COVERAGE_BUNDLE.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
    rank={}
    for t in sorted(set(c['types'].tolist())):
        mask=c['types']==t;y=c['lower'][mask,57:66];known=c['lower'][mask,57:66]==c['upper'][mask,57:66];pred=c['critic_probability'][mask,:,:9].mean(1);valid=(y[:,:,None]-y[:,None,:]>=.25)&known[:,:,None]&known[:,None,:];correct=(pred[:,:,None]>pred[:,None,:])+.5*(pred[:,:,None]==pred[:,None,:]);den=valid.sum((1,2));values=(correct*valid).sum((1,2))/np.maximum(den,1)
        # Keep controller weighting equal even if states have no identifiable pair.
        cc=[]
        for n in sorted(set(c['names'][mask].tolist())):
            nm=c['names'][mask]==n;support=nm&(den>0)
            if support.any():cc.append(float(values[support].mean()))
        rank[t]=dict(accuracy=float(np.mean(cc)) if cc else None,supported_controllers=len(cc),identifiable_pairs=int(den.sum()))
    write(HERE/'DEVELOPMENT_VALIDATION.json',dict(utc=utc(),folds=folds,ranking=ranking,coverage_generator_proposal_ranking=rank,generator_curves='DEVELOPMENT_OUTER_FOLDS.json generator_selection; each fit HISTORY.json retains omitted-type positive NLL',all_development=True,outer_labels_used_for_selection=False))
if __name__=='__main__':main()
