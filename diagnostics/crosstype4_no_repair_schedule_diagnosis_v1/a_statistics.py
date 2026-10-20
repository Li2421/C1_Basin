"""Paired hierarchy bootstrap with oracle reselection inside each draw."""
from common import *
from actions import TIMES
import csv
from a_uncertainty import full_bounds,coverage_bounds
TYPE_NAMES={'Flow':'环形交换场景流匹配策略','transformer_bc':'Transformer 行为克隆策略','conflict_based_search':'冲突搜索多机规划','centralized_mppi':'集中式模型预测路径积分控制','distributed_cross_entropy':'分布式交叉熵方法模型预测控制'}
FULL=['总差距','切换方式','切换步选择','η 对（同为按进度切换）']+[f'人定 η 对的切换步敏感性：{t}' for t in TIMES]
COVER=['总差距','切换方式','critic 选 η 对','生成器覆盖','样本内提议最佳（上界）']

def summarize(x):
    return dict(point=float(x[0]*100),CI95=np.quantile(x[1:]*100,[.025,.975]).tolist())

def oracle_prediction(labels,draw):
    sampled=labels[draw];totals=sampled.sum(0);counts=np.bincount(draw,minlength=len(labels));result=[];degenerate=0
    for sid in draw:
        support=len(draw)-counts[sid]
        scores=totals-counts[sid]*labels[sid]
        if support==0:pick=0;degenerate+=1
        else:pick=int(np.argmax(scores))
        result.append(labels[sid,pick])
    return float(np.mean(result)),degenerate

def full_values(d,draw):
    old=d['old_selected'][draw];prog=d['selected_progress'][draw];human=d['human_progress'][draw]
    fixed=np.stack([d['selected_'+str(t)][draw] for t in TIMES],axis=-1)
    # For each controller, select one time on all resampled states of the
    # opposite seed. The chosen time is recomputed for every bootstrap draw.
    chosen=np.argmax(fixed.mean(0),axis=-1);best=np.stack([fixed[:,q,chosen[1-q]] for q in (0,1)],axis=-1)
    return np.asarray([(human-old).mean(),(prog-old).mean(),(best-old).mean(),(human-prog).mean(),*[(human-d['human_'+str(t)][draw]).mean() for t in TIMES]])

def coverage_values(d,draw):
    # Coverage always uses seed 0, including old choices and human references.
    old=d['old_selected'][draw,0].mean();prog=d['selected_progress'][draw,0].mean();human=d['human_progress'][draw,0].mean();best,n=oracle_prediction(d['proposal'],draw)
    out=np.asarray([human-old,prog-old,best-prog,human-best,d['proposal'][draw].max(-1).mean()])
    assert np.isclose(out[0],out[1:4].sum(),rtol=0,atol=1e-12)
    return out,n

def load():
    links=read(HERE/'A_ACTION_LINKS.json');aliases=read(HERE/'A_ALL_LOCAL_EQUIVALENCE.json');data={};outcomes={};detail=HERE/'A_ALL_ACTION_OUTCOMES.csv'
    with rollout_db.connect(True) as db,detail.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['controller','type','state_uid','seed','method_or_pair','method_name','selected_single_eta','rollout_uid','evidence_kind','success_lower','success_upper','collision','timeout','unknown']);writer.writeheader()
        def outcome(rid):
            if rid in outcomes:return outcomes[rid]
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
            if row is None:r=aliases[rid]['outcome'];kind='提前结束等价复用'
            else:r=dict(row);assert not r['conflict_quarantined'];kind='真实执行（含历史精确缓存）'
            unknown=bool(r['numerical_failure']);v=dict(lower=0. if unknown else float(r['success']),upper=1. if unknown else float(r['success']),unknown=unknown,collision=r['collision'],timeout=r['timeout'],kind=kind);outcomes[rid]=v;return v
        for link in links:
            name=link['controller'];s=link['state_uid'];entry=data.setdefault(name,dict(type=link['type'],states={}))['states'].setdefault(s,dict(diagonal=link['diagonal'],coverage=link['coverage'],methods={},pairs={}))
            for group,key in [('methods','methods'),('proposal_pairs','pairs')]:
                for method,rid in link[group].items():
                    v=outcome(rid);entry[key].setdefault(method,{})[link['seed']]=(v['lower'],v['upper'],rid)
                    display={'old_selected':'学习双 η·开局选切换步','selected_progress':'选中 η 对·按进度切换','human_progress':'人定双 η·按进度切换'}
                    method_name=display.get(method,('选中 η 对·第 '+method.split('_')[-1]+' 步切换') if method.startswith('selected_') else ('人定双 η·第 '+method.split('_')[-1]+' 步切换') if method.startswith('human_') else '提议 η 对 '+method+'·按进度切换')
                    if link['diagonal'] and method.startswith('selected_'):method_name='选中单 η（不适用切换实验）'
                    writer.writerow(dict(selected_single_eta=int(link['diagonal']),method_name=method_name,controller=name,type=TYPE_NAMES[link['type']],state_uid=s,seed=link['seed'],method_or_pair=method,rollout_uid=rid,evidence_kind=v['kind'],success_lower=v['lower'],success_upper=v['upper'],collision=v['collision'],timeout=v['timeout'],unknown=int(v['unknown'])))
    full={};coverage={};diagonal={};identity={}
    for name,d in data.items():
        ss=list(d['states'].values());diagonal[name]=sum(s['diagonal'] for s in ss)
        for bound in (0,1):
            target={k:np.asarray([[s['methods'][k][q][bound] for q in (0,1)] for s in ss]) for k in ss[0]['methods']}
            full.setdefault(name,dict(type=d['type']))[bound]=target
            full[name]['diagonal_mask']=np.asarray([s['diagonal'] for s in ss],bool)
            full[name]['uids']={k:np.asarray([[s['methods'][k][q][2] for q in (0,1)] for s in ss]) for k in ss[0]['methods']}
            cs=[s for s in ss if s['coverage']]
            if cs:
                cdata={k:np.asarray([[s['methods'][k][q][bound] for q in (0,1)] for s in cs]) for k in cs[0]['methods']}
                keys=sorted(cs[0]['pairs'],key=lambda x:tuple(map(int,x.split(','))));assert len(keys)==56
                cdata['proposal']=np.asarray([[s['pairs'][k][0][bound] for k in keys] for s in cs]);coverage.setdefault(name,dict(type=d['type']))[bound]=cdata
                coverage[name]['uids']={k:np.asarray([[s['methods'][k][q][2] for q in (0,1)] for s in cs]) for k in cs[0]['methods']}
                coverage[name]['uids']['proposal']=np.asarray([[s['pairs'][k][0][2] for k in keys] for s in cs])
    return full,coverage,diagonal,outcomes

def calculate(full,cover):
    types=sorted({v['type'] for v in full.values()});rng=np.random.default_rng(202610194501);draws=5000;store={b:{scope:{t:[] for t in [*types,'类型等权合并']} for scope in ('full','coverage')} for b in (0,1,2,3)};degeneracies=0
    for rep in range(draws+1):
        per={b:{'full':{},'coverage':{}} for b in (0,1,2,3)}
        for typ in types:
            for scope,dataset in [('full',full),('coverage',cover)]:
                names=[n for n,d in dataset.items() if d['type']==typ];selected=names if rep==0 else rng.choice(names,len(names),replace=True).tolist();vals={0:[],1:[],2:[],3:[]}
                for name in selected:
                    ns=len(dataset[name][0]['old_selected']);idx=np.arange(ns) if rep==0 else rng.integers(ns,size=ns)
                    for bound in (0,1):
                        if scope=='full':v=full_values(dataset[name][bound],idx)
                        else:
                            v,n=coverage_values(dataset[name][bound],idx)
                            if bound==0:degeneracies+=n
                        vals[bound].append(v)
                    bounds=full_bounds(dataset[name],idx) if scope=='full' else coverage_bounds(dataset[name],idx)
                    vals[2].append(bounds[0]);vals[3].append(bounds[1])
                for bound in (0,1,2,3):
                    value=np.mean(vals[bound],axis=0);store[bound][scope][typ].append(value);per[bound][scope][typ]=value
        type_draw=types if rep==0 else rng.choice(types,len(types),replace=True).tolist()
        # Each occurrence of a resampled type gets independent nested draws.
        # Sharing one already-resampled type mean across duplicate occurrences
        # would inflate the conditional variance of the hierarchy bootstrap.
        for scope,dataset in [('full',full),('coverage',cover)]:
            combined={b:[] for b in (0,1,2,3)}
            for typ in type_draw:
                if rep==0:
                    for b in combined:combined[b].append(per[b][scope][typ])
                    continue
                names=[n for n,d in dataset.items() if d['type']==typ];values={b:[] for b in combined}
                for name in rng.choice(names,len(names),replace=True):
                    ns=len(dataset[name][0]['old_selected']);idx=rng.integers(ns,size=ns)
                    for b in (0,1):
                        v=full_values(dataset[name][b],idx) if scope=='full' else coverage_values(dataset[name][b],idx)[0];values[b].append(v)
                    bounds=full_bounds(dataset[name],idx) if scope=='full' else coverage_bounds(dataset[name],idx)
                    values[2].append(bounds[0]);values[3].append(bounds[1])
                for b in combined:combined[b].append(np.mean(values[b],axis=0))
            for b in combined:store[b][scope]['类型等权合并'].append(np.mean(combined[b],axis=0))
        if rep%500==0:print('A bootstrap',rep,flush=True)
    for b in store:
        for scope in store[b]:
            for t in store[b][scope]:store[b][scope][t]=np.asarray(store[b][scope][t])
    return store,degeneracies

def main():
    assert (HERE/'A_COMPLETE.receipt.json').exists();full,cover,diag,outcomes=load();store,degens=calculate(full,cover);summary={};ratios={}
    for scope,metrics in [('full',FULL),('coverage',COVER)]:
        summary[scope]={}
        for typ,values in store[0][scope].items():
            summary[scope][typ]={m:summarize(values[:,i]) for i,m in enumerate(metrics)}
            for i,m in enumerate(metrics):
                summary[scope][typ][m]['all_unknown_success_analysis']=summarize(store[1][scope][typ][:,i])
                summary[scope][typ][m]['unknown_bounds']=[float(store[2][scope][typ][0,i]*100),float(store[3][scope][typ][0,i]*100)]
                summary[scope][typ][m]['unknown_bounds_CI95']=[float(np.quantile(store[2][scope][typ][1:,i]*100,.025)),float(np.quantile(store[3][scope][typ][1:,i]*100,.975))]
            if scope=='coverage':ratios[typ]={metrics[i]:(float(values[0,i]/values[0,0]) if values[0,0]!=0 else None) for i in (1,2,3)}
    combined=summary['coverage']['类型等权合并'];causes=[]
    for key,label in [('切换方式','切换时间学得不好'),('critic 选 η 对','critic 没从提议中挑到好的 η 对'),('生成器覆盖','生成器没提出好的 η 对')]:
        if combined[key]['CI95'][0]>0:causes.append(dict(cause=label,component=key,point=combined[key]['point'],fraction=ratios['类型等权合并'][key]))
    if summary['full']['类型等权合并']['切换方式']['CI95'][0]>0 and not any(c['cause']=='切换时间学得不好' for c in causes):causes.append(dict(cause='切换时间学得不好',component='全部对象切换方式',point=summary['full']['类型等权合并']['切换方式']['point'],fraction=None))
    if summary['full']['类型等权合并']['切换步选择']['CI95'][0]>0 and not any(c['cause']=='切换时间学得不好' for c in causes):causes.append(dict(cause='切换时间学得不好',component='切换步选择',point=summary['full']['类型等权合并']['切换步选择']['point'],fraction=None))
    causes.sort(key=lambda c:c['fraction'] if c['fraction'] is not None else -np.inf,reverse=True)
    result=dict(utc=utc(),evidence='DEVELOPMENT',bootstrap=5000,seed=202610194501,full=summary['full'],coverage=summary['coverage'],fractions=ratios,causes=causes,conclusion='；'.join(c['cause'] for c in causes) if causes else '本轮数据不足以定位',selected_diagonal_states=diag,bootstrap_degenerate_predictions=degens,unknown_unique_rollouts=sum(v['unknown'] for v in outcomes.values()),unknown_policy='lower selects and scores; paired lower/upper intervals freeze lower-label oracle selections and cancel identical UIDs exactly; separate all-upper re-selection sensitivity',fixed_time_crossfit='one time per controller, selected on all sampled states of opposite seed; reselect within bootstrap')
    result['non_diagonal_conditional']={}
    for typ in sorted({d['type'] for d in full.values()}):
        values=[];states=0
        for d in full.values():
            if d['type']!=typ:continue
            draw=np.flatnonzero(~d['diagonal_mask']);states+=len(draw)
            if len(draw):values.append(full_values(d[0],draw))
        result['non_diagonal_conditional'][typ]=dict(controllers=len(values),controller_states=states,point_pp=dict(zip(FULL,np.mean(values,axis=0)*100)) if values else None)
    write(HERE/'A_DIAGNOSIS_RESULTS.json',result)
    with (HERE/'A_UNCERTAINTY.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['范围','类型','分量','下界标签点估计pp','配对未知下界pp','配对未知上界pp','未知区间CI下端pp','未知区间CI上端pp']);writer.writeheader()
        for scope,types in summary.items():
            for typ,components in types.items():
                for m,r in components.items():writer.writerow({'范围':scope,'类型':TYPE_NAMES.get(typ,typ),'分量':m,'下界标签点估计pp':r['point'],'配对未知下界pp':r['unknown_bounds'][0],'配对未知上界pp':r['unknown_bounds'][1],'未知区间CI下端pp':r['unknown_bounds_CI95'][0],'未知区间CI上端pp':r['unknown_bounds_CI95'][1]})
    def fmt(r):return f"{r['point']:+.2f} [{r['CI95'][0]:+.2f}, {r['CI95'][1]:+.2f}]"
    lines=['# 双 η 差距诊断（开发证据）','',result['conclusion']+'。本轮不重训、不重新选择已有 η 对。','', '差值单位为百分点，括号为95%配对分层 bootstrap 区间。固定步最佳值按 controller 用另一 seed 的全部状态选择；每次重抽重新选择。','']
    for scope,metrics,title in [('full',FULL,'全部对象'),('coverage',COVER,'提议覆盖子集')]:
        lines.extend(['## '+title,'','| 类型 | '+' | '.join(metrics)+' |','|---|'+'---|'*len(metrics)])
        for typ,values in summary[scope].items():lines.append('| '+TYPE_NAMES.get(typ,typ)+' | '+' | '.join(fmt(values[m]) for m in metrics)+' |')
        lines.append('')
    lines.extend(['覆盖子集逐次验证：总差距 = 切换方式 + critic 选 η 对 + 生成器覆盖。','', '分量占比：'+json.dumps({TYPE_NAMES.get(t,t):v for t,v in ratios.items()},ensure_ascii=False),'',f'选中单 η 共 {sum(diag.values())} 个 controller-state；没有实际切换，其切换效应记零，物理标签复用同一 UID。未知数值失败保留下界，JSON另列全上界重算。','',f'交叉拟合重抽中只剩一个唯一状态的预测数：{degens}；此时不读取留出标签，以最低提议序号对回退。','', '全部动作、seed、真实执行或提前结束等价复用与 unknown 明细见 A_ALL_ACTION_OUTCOMES.csv；每项配对未知上下界及其区间见 A_UNCERTAINTY.csv。'])
    (HERE/'DIAGNOSIS.md').write_text('\n'.join(lines)+'\n')
    np.savez_compressed(HERE/'A_BOOTSTRAP.npz',**{f'{scope}_{typ}_{bound}':v for bound,s in store.items() for scope,ts in s.items() for typ,v in ts.items()})
    from a_plots import make
    figures=make(full,cover,result)
    seal('A_DIAGNOSIS.receipt.json',figures+[HERE/n for n in ('A_DIAGNOSIS_RESULTS.json','DIAGNOSIS.md','A_ALL_ACTION_OUTCOMES.csv','A_UNCERTAINTY.csv','A_BOOTSTRAP.npz','a_statistics.py','a_uncertainty.py','a_plots.py','A_COMPLETE.receipt.json')],oracle_reselected_each_bootstrap=True)
if __name__=='__main__':main()
