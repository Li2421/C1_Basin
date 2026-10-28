"""Evidence-only reports: finite coverage, query count, times, paper ports."""
import collections,csv,hashlib,json,math,random,statistics,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'scene_types_basis_dimension_v1'
def read(p):return json.loads(Path(p).read_text())
def dump(name,v):(HERE/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def mean(xs):return statistics.mean(xs) if xs else None
def fmt(v):return '—' if v is None else f'{v:.3f}'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv_write(name,rows):
    with (HERE/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def group_path(g):return HERE/'oracle64_groups'/f"{g['family']}_{g['type_id']}_{g['controller']}_{g['state_index']}.json"
def support(sp):return [k for k,x in zip(('G','P','R'),sp['eta']) if x!=0.]+[k for k,x in sp.get('coefficients',{}).items() if x!=0.]
def wilson(k,n):
    z=1.959963984540054;a=k/n+z*z/(2*n);b=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n));d=1+z*z/n
    return [(a-b)/d,(a+b)/d]
def block_ci(rows,field,reference=None):
    blocks=collections.defaultdict(list)
    for r in rows:
        value=r[field]-(r[reference] if reference else 0)
        blocks[r['type_id'],r['state_index']].append(value)
    values=[mean(xs)for xs in blocks.values()];rng=random.Random(2026106401)
    draws=sorted(mean(rng.choices(values,k=len(values))) for _ in range(3000))
    return [draws[74],draws[2924]]

def selected_papers():
    mapping=[('Wang 2017 Consistent Perturbation','wang2017_consistent_perturbation_di_tracking'),('Wang 2017完整Deadlock Resolution','wang2017_full_di_tracking'),('Chen 2023 Dynamic Priority+RHR+Safe Corridor','chen_impc_ob_si_tracking'),('Zhang ICRA2025 Adaptive CLF–CBF','zhang2025_risk_si__development_selected'),('Zhang T-RO2025 DAC','dac_author_si__development_selected'),('Lee IROS2025 MGR','mgr_author_unicycle_tracking__development_selected')]
    stats=read(OLD/'PAPER_RESULTS.json');pools=collections.defaultdict(list)
    for c in read(OLD/'paper_holdout_planned.json')['exact_configs']:
        r=read(OLD/'raw'/f"{c['rollout_uid']}.json")
        pools[r['controller']].append(r)
    result=[]
    for title,name in mapping:
        rr={r['rollout_uid']:r for r in pools[name]}
        assert len(rr)==60 and sum(r['success']for r in rr.values())==stats[name]['overall']['success']
        result.append(dict(method=title,profile=name,statistics=stats[name]['overall'],rows=list(rr.values())))
    return result

def main(partial=False):
    protocol=read(HERE/'ORACLE64_FROZEN.json');groups=read(HERE/'ORACLE64_GROUPS.json')['groups']
    finished=[g for g in groups if group_path(g).exists()]
    if partial:
        dump('ORACLE64_PROGRESS_SUMMARY.json',dict(completed=len(finished),total=len(groups),new_raw=len(list((HERE/'raw').glob('*.json')))))
        print('completed',len(finished),'/',len(groups));return
    assert len(finished)==len(groups)==960
    assert read(HERE/'ORACLE64_EXECUTION.json')['complete']
    cache={};case_records=[];candidate_rows=[];rescues=[]
    def row(point):
        if point['uid'] not in cache:cache[point['uid']]=read(point['path'])
        return cache[point['uid']]
    for g in groups:
        done=read(group_path(g));visited=done['visited'];assert visited
        points={x['candidate']:x for x in g['candidates']}
        index=done['first_success'];selected=row(points[index or 1])
        if index is None:assert len(visited)==64
        else:
            assert selected['success'] and visited[-1]['candidate']==index
            assert all(not row(points[x['candidate']])['success'] for x in visited[:-1])
        out=dict(family=g['family'],type_id=g['type_id'],controller=g['controller'],state_index=g['state_index'],first_success=index,success16=int(index is not None and index<=16),success32=int(index is not None and index<=32),success64=int(index is not None),search_count=index or 64,completion_time=selected['completion_time'] if index else None,truncated_time=selected['completion_time'] if index else selected['horizon_time'],selected_uid=selected['rollout_uid'],selected_spec=selected['spec'],actual_new_rollouts=sum(points[x['candidate']]['source']=='new_missing' for x in visited),candidate_outcomes=dict(collections.Counter(x['outcome']for x in visited)),visited_unknown=sum(x['outcome']=='unknown'for x in visited))
        case_records.append(out)
        if index is not None and index>16:rescues.append(out)
        for v in visited:
            p=points[v['candidate']];r=row(p)
            candidate_rows.append(dict(family=g['family'],type_id=g['type_id'],controller=g['controller'],state_index=g['state_index'],candidate=v['candidate'],uid=v['uid'],outcome=r['outcome'],completion_time=r.get('completion_time'),wall_seconds=r['wall_seconds'],source=p['source'],path=p['path'],spec=json.dumps(r['spec'],sort_keys=True)))
    dump('ORACLE64_CASE_RESULTS.json',case_records);dump('ORACLE64_NEW_RESCUES.json',rescues)
    csv_write('oracle64_visited_candidates.csv',candidate_rows)
    summaries=[];type_rows=[];nominal_rows=[]
    def summary(rr):
        success=[r for r in rr if r['success64']];oldfailed=[r for r in rr if not r['success16']]
        return dict(n=len(rr),success16=sum(r['success16']for r in rr),success32=sum(r['success32']for r in rr),success64=len(success),new_rescues_after16=sum(r['success64']and not r['success16']for r in rr),first_success_index_mean=mean([r['first_success']for r in success]),all_cohort_restricted_query_mean=mean([r['search_count']for r in rr]),original_failed_cohort_n=len(oldfailed),original_failed_cohort_first_success_mean=mean([r['first_success']for r in oldfailed if r['success64']]),success_mean_completion_time=mean([r['completion_time']for r in success]),all_cohort_truncated_task_time=mean([r['truncated_time']for r in rr]),new_rollouts_logged=sum(r['actual_new_rollouts']for r in rr),failed_at64=len(rr)-len(success),cases_with_visited_unknown=sum(r['visited_unknown']>0 for r in rr))
    for f in protocol['families']:
        rr=[r for r in case_records if r['family']==f['index']];s=summary(rr)
        s.update(family=f['index'],columns=f['columns'],dimension=f['dimension'],physical_block_success_rate_bootstrap95=block_ci(rr,'success64'),physical_block_improvement16_to64_bootstrap95=block_ci(rr,'success64','success16'))
        s['enough_types64']=sum(sum(r['success64'] for r in rr if r['type_id']==t)>=6 for t in {r['type_id'] for r in rr})
        summaries.append(s)
        for t in sorted({r['type_id']for r in rr}):type_rows.append(dict(family=f['index'],columns='+'.join(f['columns']),type_id=t,**summary([r for r in rr if r['type_id']==t])))
        for ctl in sorted({r['controller']for r in rr}):nominal_rows.append(dict(family=f['index'],columns='+'.join(f['columns']),controller=ctl,**summary([r for r in rr if r['controller']==ctl])))
    dump('ORACLE64_RESULTS.json',summaries);csv_write('oracle64_per_type.csv',type_rows);csv_write('oracle64_per_controller.csv',nominal_rows)
    primary=[s for s in summaries if s['family']!='B3_wide']
    maximum=max(s['enough_types64']for s in primary)
    eligible=[s for s in primary if s['enough_types64']>=maximum-1]
    chosen=min(eligible,key=lambda s:(s['dimension'],-s['enough_types64'],-s['success64']))
    dump('ORACLE64_DIMENSION_VERDICT.json',dict(recommended=chosen,M=maximum,threshold=maximum-1,rule='Posthoc application of original smallest-dimension rule: type enough>=6/8; primary family enough-types>=M-1. Widened-G control separately reported, not mixed into original primary box.',scope='Finite eight-pool screen, not continuous minimum dimension or new independent validation.'))
    csv_write('oracle64_budget_curve.csv',[dict(family=f['index'],K=k,covered=sum(r['family']==f['index']and r['first_success']is not None and r['first_success']<=k for r in case_records),n=120)for f in protocol['families']for k in range(1,65)])
    paper=selected_papers();dump('PAPER_BASELINES.json',[{k:v for k,v in p.items()if k!='rows'}for p in paper])
    paired=[]
    fixedfamilies=read(OLD/'FAMILIES_FROZEN.json')['families'];labels=read(OLD/'HOLDOUT_LABELS.json')
    for fidx in ('0','4'):
        fixed={}
        for x in labels[fidx]:
            if x['candidate']==0:
                r=read(OLD/'raw'/f"{x['uid']}.json");assert r['spec']==fixedfamilies[int(fidx)]['fixed'];fixed[r['type_id'],r['controller'],r['state_index']]=r
        for ctl in ('centralized_sampling','gradient_mpc'):
            rr=[r for r in case_records if r['family']==fidx and r['controller']==ctl]
            for p in paper:
                common={(r['type_id'],r['state_index']):r for r in p['rows']}
                for mode in ('fixed','oracle64'):
                    pairs=[]
                    for r in rr:
                        a=fixed[r['type_id'],ctl,r['state_index']] if mode=='fixed' else dict(success=r['success64'],completion_time=r['completion_time'])
                        b=common[r['type_id'],r['state_index']]
                        if a['success']and b['success']:pairs.append((a['completion_time'],b['completion_time']))
                    paired.append(dict(family=fidx,controller=ctl,mode=mode,paper_method=p['profile'],paired_success_n=len(pairs),C1_paired_mean_time=mean([a for a,b in pairs]),paper_paired_mean_time=mean([b for a,b in pairs]),paired_time_difference=mean([a-b for a,b in pairs])))
    dump('PAPER_PAIRED_SUCCESS_TIMES.json',paired)
    write_report(summaries,case_records,paper,candidate_rows,cache)

def write_report(summaries,cases,papers,candidates,cache):
    # Kept separate so the final report is rebuilt only from completed evidence.
    report=['# Correction Oracle@64 与论文基线\n\n无 Generator、无训练。按用户要求，将原16点未成功案例扩至最多64点；成功即停，64点无成功就结束为本轮搜索失败。前16点保持不变，B随状态按固定规则更新，η整局固定，原nominal/安全层/动力学/任务判据未改。原留出结果已看过，本轮是事后预算扩展。旧目录只读，旧报告和维度结论不覆盖。\n\n']
    verdict=read(HERE/'ORACLE64_DIMENSION_VERDICT.json');chosen=verdict['recommended']
    report.append(f"按原最小维度规则事后应用于64点结果：推荐{'/'.join(chosen['columns'])}，维度{chosen['dimension']}，达到6/8覆盖的场景类型{chosen['enough_types64']}个。这里只比较已冻结候选族，不能称为全连续空间最小维度。\n\n")
    report.append('## 内部修正族：是否找到成功η\n\n每项120个controller×初态案例，15个有效场景类型×4留出初态×2个冻结nominal。另两类几何缺口沿用旧报告，不计为控制失败。三/四/五列是内部消融。\n\n|B|维度|@16|@32|@64|成功首次序号均值|全队列查询均值|成功任务均时(s)|全队列截断任务均时(s)|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for s in summaries:report.append(f"|{'+'.join(s['columns'])}{'，G≥0' if s['family']=='B3_wide' else ''}|{s['dimension']}|{s['success16']}/120|{s['success32']}/120|{s['success64']}/120|{fmt(s['first_success_index_mean'])}|{fmt(s['all_cohort_restricted_query_mean'])}|{fmt(s['success_mean_completion_time'])}|{fmt(s['all_cohort_truncated_task_time'])}|\n")
    report.append('\n首次成功序号按冻结池顺序从1开始；成功均值只含成功案例，全队列查询均值对失败记64。旧16点当时全部跑过，首次序号是本顺序下找到解需要的查询数，不是当时实际rollout消耗。新候选逐个执行并早停；`oracle64_visited_candidates.csv`区分缓存与新rollout，保留每个η和成本。`oracle64_per_type.csv`与`oracle64_per_controller.csv`给出分解；`ORACLE64_RESULTS.json`含按共同物理初态分块的区间。\n\n')
    report.append('## 指定论文基线\n\n正式外部baseline是五篇论文的六项方法。参数在开发集选取后冻结；已有540条含历史/开发选参profile的正式记录直接复用，本表选用开发集冻结profile，全部profile仍保留在旧PAPER_RESULTS.json。\n\n|方法端口|成功/60|成功均时(s)|碰撞|超时|unknown|全队列截断均时(s)|\n|---|---:|---:|---:|---:|---:|---:|\n')
    for p in papers:
        s=p['statistics'];report.append(f"|{p['method']}|{s['success']}/60|{fmt(s['success_mean_time'])}|{s['collision']}|{s['timeout']}|{s['numerical_unknown']}|{fmt(s['all_cohort_truncated_mean'])}|\n")
    report.append('\n论文端口使用自身controller和安全方法，同一SI物理环境承载；不接收C1的外部路线引导，也没有同等完整单机能力门。Wang双积分器和MGR独轮车的动力学映射、DAC空锥异常、风险QP非最优等限制见[旧异常审计](../scene_types_basis_dimension_v1/PAPER_FAILURE_ANALYSIS.md)。这些数字是本适配协议的完成情况，不能据此宣称原论文方法能力排名。C1的Oracle逐例事后选参数，论文参数固定；部署比较必须同时看C1固定参数。\n\n两个C1 nominal各60个共同物理初态；固定参数与Oracle的共同成功配对耗时见`PAPER_PAIRED_SUCCESS_TIMES.json`，零共同成功时均值为空，不记0。内部固定结果见[原分控制器结果](../scene_types_basis_dimension_v1/PER_CONTROLLER_RESULTS.json)。\n\n')
    original=read(HERE/'HISTORY_AUDIT.json');remaining={(x['type_id'],x['controller'],x['state_index'])for x in original['cases']}
    b4=[r for r in cases if r['family']=='4'and(r['type_id'],r['controller'],r['state_index'])in remaining]
    rescue=[r for r in b4 if r['success64']]
    report.append(f"## 原四列失败43例\n\n同一四列64点新覆盖其中{len(rescue)}例；其余{43-len(rescue)}例在该四列64点池中未找到成功η，不再扩搜。另有旧其他线性池的11例补救和56点跨开发初态筛选的1例成功；这些属于其他候选配置，不能累计冒充原四列的Oracle@64。逐例参数和来源见`ORACLE64_NEW_RESCUES.json`及`TRANSFER56_SUMMARY.json`。\n\n")
    new=[read(p)for p in(HERE/'raw').glob('*.json')];count=collections.Counter(r['outcome']for r in new)
    report.append(f"## 计算与可恢复性\n\n本目录当前完整新rollout {len(new)}，结局{dict(count)}；新rollout循环墙钟合计{sum(r['wall_seconds']for r in new):.3f}秒（并行前求和，非整轮实际等待时间）。Slurm每worker1CPU、BLAS1、GPU0，实际占用≤18。`ORACLE64_EXECUTION.json`记录整阶段等待时间，`ORACLE64_JOBS.json`记录作业。所有worker写journal，唯一协调者官方合并并读回。旧36832次尝试独立保留，不并入本轮新耗费。\n\n")
    report.append('64点未找到解按要求停止为有限搜索失败，不追加预算。timeout单独保留，不自动称为deadlock；求解/接口unknown另列。候选参数、raw、trace和正式数据库UID均保留。复现命令、参数盒和缓存规则见`ORACLE64_PROTOCOL.md`；旧首步回放64/64逐字节一致、旧200局完整加速认证复用，数值执行模块逐字节不变。\n')
    (HERE/'report.md').write_text(''.join(report))
    dump('ORACLE64_COMPLETE.json',dict(complete=True,groups=len(cases),new_complete_records=len(new),new_outcomes=dict(count),new_loop_wall_seconds=sum(r['wall_seconds']for r in new),old_primary_claim_unchanged=True,report_sha256=sha(HERE/'report.md')))
    print('Report complete:',HERE/'report.md')

if __name__=='__main__':main('--partial'in sys.argv)
