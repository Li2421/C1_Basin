"""Frozen-search coverage, paired fixed-point statistics, and evidence gaps."""
from core import *
from reporting import failure_type,longest
from collections import Counter
import datetime
from functools import lru_cache

NAMES=dict(A='通道单向放行·三列版',B='通道单向放行·合成一列',C='只让等待方减速·一列',D='定时交替放行',E='整队让行',F='预约到达时间',G='去等候点')

@lru_cache(maxsize=1)
def scenes():return {(e['type_id'],e['level']):e['scene']for e in read(HERE/'SCENES.json')['entries']}
def old_key(g):return (g['type_id'],g['level'],g['controller'],g['state_index'])
def identity(g):return tuple(g['identity'])if'identity'in g else old_key(g)
def result(g):return g.get('solution',{}).get('result',g.get('solution',{}))
def covered(g,k=64,mult=6):
    if not(g['success']and g['first_success']<=k):return False
    return mult==6 or result(g)['episode_length']<=scenes()[identity(g)[:2]]['time_limits'][str(mult)]['max_steps']

def timing(records,scene_map):
    records=[(key,r)for key,r in records if r is not None]
    times=[r.get('completion_time',r['episode_length']*scene_map[tuple(key[:2])]['config']['dt'])for key,r in records if r['success']]
    truncated=[(r.get('completion_time',r['episode_length']*scene_map[tuple(key[:2])]['config']['dt'])if r['success']else scene_map[tuple(key[:2])]['config']['max_steps']*scene_map[tuple(key[:2])]['config']['dt'])for key,r in records]
    return dict(observed=len(records),success=len(times),success_mean_seconds=float(np.mean(times))if times else None,whole_cohort_truncated_mean_seconds=float(np.mean(truncated))if truncated else None,outcomes=dict(Counter(r.get('outcome','unknown')for key,r in records)),wall_seconds_sum=float(sum(r.get('wall_seconds',0.)for key,r in records)),wall_seconds_available=sum('wall_seconds'in r for key,r in records))

def paired(a,b,scene_map):
    intervals=[];rng=np.random.default_rng(2026102902);difficult={8,10,11,15,16}
    for start,end in [(0,8),(8,12)]:
        for label,predicate in [('全部',lambda t:True),('困难五类',lambda t:t in difficult),('其余类型',lambda t:t not in difficult)]:
            keys=[key for key in a if a[key]is not None and b.get(key)is not None and start<=key[3]<end and predicate(key[0])]
            blocks={}
            for key in keys:blocks.setdefault((key[0],key[1],key[3]),[]).append(int(b[key]['success'])-int(a[key]['success']))
            vals=np.array([np.mean(v)for v in blocks.values()if len(v)==2]);boot=vals[rng.integers(0,len(vals),(10000,len(vals)))].mean(axis=1)if len(vals)else[]
            common=[key for key in keys if a[key]['success']and b[key]['success']]
            time_a=timing([(key,a[key])for key in common],scene_map);time_b=timing([(key,b[key])for key in common],scene_map)
            intervals.append(dict(scope=label,states=[start,end-1],physical_blocks=len(vals),difference=float(vals.mean())if len(vals)else None,ci95=np.quantile(boot,[.025,.975]).tolist()if len(vals)else None,common_success_count=len(common),common_success_baseline_mean_seconds=time_a['success_mean_seconds'],common_success_new_mean_seconds=time_b['success_mean_seconds']))
    return dict(records=intervals,bootstrap_samples=10000,seed=2026102902,unit='physical scene/state; both nominal controllers in one resampled block')

def classify(r,scene,scheme):
    if scheme!='E':return failure_type(r,scene)
    if not r.get('timeout'):return dict(category='数值异常或碰撞',outcome=r.get('outcome'))
    with np.load(r['trace'])as tr:
        history=json.loads(str(tr['admission_json']));dt=scene['config']['dt'];window=round(2/dt)
        # Count each persistent ordered role separately. A series of unrelated
        # brief encounters must not masquerade as one two-second encounter.
        pairs={tuple(m['pair'])for step in history for m in step if m.get('pair')is not None}
        durations=[longest([any(tuple(m['pair'])==pair and m.get('headon',False)for m in step if m.get('pair')is not None)for step in history])*dt for pair in pairs]
        breach=max(durations,default=0)>=2.;remaining=tr['route_remaining']
        moving=bool(len(remaining)>window and remaining[-window-1].sum()-remaining[-1].sum()>.05)
    return dict(category='迎面对超过2秒未分开'if breach else'仍在推进的超时'if moving else'整队让行后仍拥堵或其他',max_head_on_seconds=max(durations,default=0),flags=dict(head_on_persistent=breach,exit_blocked=None,progressing_timeout=moving),trace=r['trace'])

def summarize(state):
    groups=state['groups'];success=[g['first_success']for g in groups if g['success']]
    by_type=[]
    for t in sorted({g['identity'][0]for g in groups}):
        rr=[g for g in groups if g['identity'][0]==t];ss=[g['first_success']for g in rr if g['success']]
        by_type.append(dict(type_id=t,cases=len(rr),coverage={str(k):sum(covered(g,k)for g in rr)for k in(16,32,64)},first_candidate_success=sum(bool(g['queries'])and g['queries'][0]['result']['success']for g in rr),median_first_success=float(np.median(ss))if ss else None))
    return dict(scheme=state['scheme'],cases=len(groups),coverage={str(k):sum(covered(g,k)for g in groups)for k in(16,32,64)},first_candidate_success=sum(bool(g['queries'])and g['queries'][0]['result']['success']for g in groups),mean_first_success=float(np.mean(success))if success else None,median_first_success=float(np.median(success))if success else None,mean_all_queries=float(np.mean([g['searched']for g in groups]))if groups else None,complete=state.get('complete',False),queries=sum(g['searched']for g in groups),unknown_queries=sum(q['result'].get('numerical_failure',0)for g in groups for q in g['queries']),by_type=by_type,qualifies=bool(groups)and 10*len(success)>=9*len(groups))

def allscene(family):
    scheme=family['scheme'];state=read(HERE/f'FOLLOWUP_{scheme}_all.json');fixed=read(HERE/f'FOLLOWUP_{scheme}_fixed.json');scene_map=scenes()
    old=read(HERE/'inputs/final_v3/SEARCH_STATE.json');four={old_key(g):g for g in old['groups']if g['family']==4};safe={old_key(g):g for g in old['baseline']}
    new={tuple(g['identity']):g for g in state['groups']};a={tuple(r['identity']):r['result']for r in fixed['methods']['four_column_fixed']['records']};b={tuple(r['identity']):r['result']for r in fixed['methods']['scheme_fixed']['records']}
    rows=[]
    for (t,l),sc in sorted(scene_map.items()):
        keys=[key for key in new if key[:2]==(t,l)];rr=[new[key]for key in keys];oo=[four[key]for key in keys]
        row=dict(type_id=t,level=l,name=f"{sc['title']}·参数={sc['knob']}，{sc['config']['num_agents']}台",cases=len(keys),safety_only={str(m):sum(safe[key]['success']and safe[key]['episode_length']<=sc['time_limits'][str(m)]['max_steps']for key in keys)for m in(4,6)},four_column={str(m):{str(k):sum(covered(g,k,m)for g in oo)for k in(16,64)}for m in(4,6)},new={str(m):{str(k):sum(covered(g,k,m)for g in rr)for k in(16,64)}for m in(4,6)},fixed={})
        row['oracle_success_time']=dict(four_column=timing([(key,result(four[key]))for key in keys if four[key]['success']],scene_map),new=timing([(key,result(new[key]))for key in keys if new[key]['success']],scene_map),safety_only=timing([(key,safe[key])for key in keys],scene_map))
        for start,end in [(0,8),(8,12)]:
            subset=[key for key in a if key[:2]==(t,l)and start<=key[3]<end]
            row['fixed'][str(start)]=dict(cases=len(subset),four_column=timing([(key,a[key])for key in subset],scene_map),new=timing([(key,b.get(key))for key in subset],scene_map),success_4x={method:sum(bool(mapping.get(key)and mapping[key]['success']and mapping[key]['episode_length']<=sc['time_limits']['4']['max_steps'])for key in subset)for method,mapping in [('four_column',a),('new',b)]})
        rows.append(row)
    types=[]
    for t in sorted({r['type_id']for r in rows}):
        rr=[r for r in rows if r['type_id']==t]
        types.append(dict(type_id=t,scenes=len(rr),cases=sum(r['cases']for r in rr),safety_only=sum(r['safety_only']['6']for r in rr),four_column={str(k):sum(r['four_column']['6'][str(k)]for r in rr)for k in(16,64)},new={str(k):sum(r['new']['6'][str(k)]for r in rr)for k in(16,64)}))
    losses=[dict(identity=list(key),old_first_success=four[key]['first_success'],new_searched=new[key]['searched'],gap=new[key]['gap'])for key in new if four[key]['success']and not new[key]['success']]
    gains=[dict(identity=list(key),new_first_success=new[key]['first_success'])for key in new if not four[key]['success']and new[key]['success']]
    data=dict(scheme=scheme,name=NAMES[scheme],complete=state['complete']and all(x['complete']for x in fixed['methods'].values()),scenes=rows,by_type=types,coverage=dict(four_column=sum(g['success']for g in four.values()),new=sum(g['success']for g in new.values()),cases=784),regressions=losses,gains=gains,fixed_paired=paired(a,b,scene_map),fixed_totals={str(start):{method:timing([(key,mapping[key])for key in mapping if start<=key[3]<end],scene_map)for method,mapping in [('four_column',a),('new',b)]}for start,end in[(0,8),(8,12)]},horizon_policy='4x is truncation of executed 6x first-success candidates; later candidates skipped after first 6x success are not searched for a faster 4x success.')
    dump(HERE/f'ALLSCENE_{scheme}.json',data);return data

def write_report():
    original=HERE/'ORIGINAL_REPORT.md'
    if not original.exists():original.write_text((HERE/'REPORT.md').read_text())
    summaries=[];full=[]
    for scheme in NAMES:
        path=HERE/f'FOLLOWUP_{scheme}_hard.json'
        if path.exists():summaries.append(summarize(read(path)))
        path=HERE/f'ALLSCENE_{scheme}.json'
        if path.exists():full.append(read(path))
    dump(HERE/'FOLLOWUP_SUMMARY.json',dict(families=summaries,full_schemes=[x['scheme']for x in full]))
    qualifiers=[NAMES[s['scheme']]for s in summaries if s['qualifies']]
    lines=['\n# 续接：冻结协调方案筛查\n',f"当前达到困难案例90%门槛：{'、'.join(qualifiers)if qualifiers else'尚无已验证方案'}。未跑完项不算失败。"]
    for x in full:
        totals=x['fixed_totals'];lines.append(f"{x['name']}：全场景覆盖{x['coverage']['new']}/784，原四列{x['coverage']['four_column']}/784；固定参数状态0–7成功{totals['0']['new']['success']}/{totals['0']['new']['observed']}，原四列{totals['0']['four_column']['success']}/{totals['0']['four_column']['observed']}；状态8–11成功{totals['8']['new']['success']}/{totals['8']['new']['observed']}，原四列{totals['8']['four_column']['success']}/{totals['8']['four_column']['observed']}。")
    lines+=['\n| 方案 | 案例数 | 覆盖16/32/64 | 第1组成功 | 成功所需组数均值/中位数 | 数值异常候选 | 完成 |','|---|---:|---|---:|---|---:|---|']
    for s in summaries:lines.append(f"| {NAMES[s['scheme']]} | {s['cases']} | {'/'.join(str(s['coverage'][str(k)])for k in(16,32,64))} | {s['first_candidate_success']} | {s['mean_first_success']}/{s['median_first_success']} | {s['unknown_queries']} | {s['complete']} |")
    lines+=['\n首次成功即停；所有方案使用同一个四列候选序列与匹配随机流。Oracle是直接搜索冻结候选参数，不使用生成器。均值只对成功案例计算，所有案例的平均实际搜索次数另见FOLLOWUP_SUMMARY.json。','\n乙、丙为原四列上增加一个协调系数；它们仍有状态和乘性作用，不应混称为固定矩阵下对全部系数线性的五维族。未搜索的连续参数不能判为不可能成功。']
    scene_map=scenes()
    for s in summaries:
        lines += [f"\n## {NAMES[s['scheme']]}：困难案例分型\n",'| 场景类型 | 案例 | 覆盖16/32/64 | 第1组成功 | 成功序号中位数 |','|---|---:|---|---:|---:|']
        for t in s['by_type']:
            title=next(sc['title']for key,sc in scene_map.items()if key[0]==t['type_id'])
            lines.append(f"| {title} | {t['cases']} | {'/'.join(str(t['coverage'][str(k)])for k in(16,32,64))} | {t['first_candidate_success']} | {t['median_first_success']} |")
        fail=HERE/f"FOLLOWUP_{s['scheme']}_FAILURES.json"
        if fail.exists():lines.append('\n失败分型（首次候选与搜索耗尽的末次候选分别统计）：'+json.dumps(read(fail)['counts'],ensure_ascii=False)+'。')
    for data in full:
        lines += [f"\n## {data['name']}：49场景对照\n",'表中Oracle为6倍时限；固定参数依次为状态0–7 / 状态8–11。时间为成功样本条件均值（秒），不得与配对成功样本均值或全队列截断时间混用。','| 场景 | 仅安全 | 四列16/64 | 新方案16/64 | 四列固定 | 新方案固定 | 四列/新方案成功均时 |','|---|---:|---|---|---|---|---|']
        for r in data['scenes']:
            ft=lambda method:'/'.join(str(r['fixed'][str(k)][method]['success'])for k in(0,8))
            tm=lambda method:r['oracle_success_time'][method]['success_mean_seconds']
            lines.append(f"| {r['name']} | {r['safety_only']['6']} | {r['four_column']['6']['16']}/{r['four_column']['6']['64']} | {r['new']['6']['16']}/{r['new']['6']['64']} | {ft('four_column')} | {ft('new')} | {tm('four_column')}/{tm('new')} |")
        lines.append(f"\n四列覆盖而该方案未覆盖{len(data['regressions'])}例；新增覆盖{len(data['gains'])}例。逐条清单、17类汇总、4倍/6倍时限与固定参数时间见[完整表](ALLSCENE_{data['scheme']}.json)。")
        lines+=['\n| 固定参数子集 | 状态 | 配对成功率差 | 分块自助法95%区间 | 配对均成功时间·四列/新方案 |','|---|---|---|---|---|']
        for r in data['fixed_paired']['records']:lines.append(f"| {r['scope']} | {r['states']} | {r['difference']} | {r['ci95']} | {r['common_success_baseline_mean_seconds']}/{r['common_success_new_mean_seconds']} |")
    lines+=['\n4倍口径只截取实际执行的6倍首次成功轨迹，不搜索已跳过候选；该口径对所有方案一致。超时与数学死锁严格区分。','\n下一步建议以已完成方案距90%门槛的差值及失败分型为依据；本轮不会自行改为全局单行道网络或联合排程框架。']
    (HERE/'REPORT.md').write_text(original.read_text()+'\n'+'\n'.join(lines).replace('None','—')+'\n')

def finish(progress):
    scene_map=scenes();summaries=[];gaps=[]
    for scheme in NAMES:
        path=HERE/f'FOLLOWUP_{scheme}_hard.json'
        if not path.exists():gaps.append(NAMES[scheme]+'未运行。');continue
        state=read(path);summary=summarize(state);summaries.append(summary)
        if not state['complete']:gaps.append(NAMES[scheme]+'存在尚未搜索满64次且未成功的案例。')
        if summary['qualifies']and scheme not in progress['full_complete']:gaps.append(NAMES[scheme]+'达到门槛，全场景验证尚未完成。')
        failures=[]
        for g in state['groups']:
            if not g['queries']:continue
            for label,q in [('first_candidate',g['queries'][0])]+([('exhausted_final_candidate',g['queries'][-1])]if not g['success']and g['searched']==64 else[]):
                r=q['result']
                if not r['success']:failures.append(dict(identity=g['identity'],scope=label,index=q['index'],**classify(r,scene_map[tuple(g['identity'][:2])],scheme)))
        dump(HERE/f'FOLLOWUP_{scheme}_FAILURES.json',dict(records=failures,counts={scope:dict(Counter(x['category']for x in failures if x['scope']==scope))for scope in ['first_candidate','exhausted_final_candidate']}))
    from finish_extras import integrity
    passed=integrity()
    if not passed:gaps.append('完整性核验未通过，见INTEGRITY_END.json。')
    if progress['budget_exhausted']:gaps.append('续接35000局硬预算耗尽，未完成请求保存在对应pending frontier，不算控制失败。')
    original=read(HERE/'ORIGINAL_COMPLETE.json');budget=read(HERE/'BUDGET.json')
    dump(HERE/'COMPLETE.json',dict(complete=not gaps and original['complete'],original_complete=original['complete'],followup_complete=not gaps,integrity_passed=passed,actual_attempts=budget,progress=progress,finished=datetime.datetime.now().astimezone().isoformat(),report=str(HERE/'REPORT.md')))
    (HERE/'GAPS.md').write_text('# 证据缺口\n\n'+('\n'.join('- '+g for g in gaps)if gaps else'规定实验与交付已完成。有限候选失败不构成连续参数不可解证明。')+'\n')
    write_report()
    (HERE/'STATUS.md').write_text('# 当前状态\n\n'+('任务完成，见REPORT.md。'if not gaps else'已保存结果与可恢复状态；存在证据缺口，见GAPS.md。')+'\n')
