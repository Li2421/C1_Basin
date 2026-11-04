from native_io import *
import csv,collections,statistics,html
OLD=ROOT/'diagnostics/c1_v3_new_baseline_reproduction_20261103'
PAIRED=ROOT/'diagnostics/c1_v3_mcbf_circulation_paired_20261104'
LABELS={'Pipeline':'C1 V3 冻结部署','dac':'DAC / Zhang','mgr':'MGR / Lee','mcbf':'MCBF','circulation':'环流 CBF'}
def csv_read(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csv_write(name,rows):
    if not rows:return
    with (D/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def table(rows,keys):
    return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'

def main():
    links=read(D/'COMPARISON_LINKS.json');lookup={};detail=[];summary=[]
    for ref in links:
        old=read(ref['old_raw']);p=Path(ref['native_raw']);r=read(p) if p.exists() else None
        key=(ref['method'],ref['physical_key']);assert key not in lookup,key;lookup[key]=(r,ref)
        detail.append(dict(physical_key=ref['physical_key'],cohort=ref['cohort'],method=ref['method'],control_period_s=ref['control_period_s'],score_period_s=ref['score_period_s'],
            old_period_s=ref['old_period_s'],reused=ref['reuse'],old_outcome=old['outcome'],native_outcome=r['outcome'] if r else 'pending',
            native_success=r['success'] if r else None,old_completion_seconds=old.get('completion_time'),native_completion_seconds=r.get('completion_time') if r else None,
            old_uid=ref['old_uid'],native_uid=ref['native_uid'],native_trace=r.get('trace') if r else None,
            minimum_swept_clearance=r.get('minimum_swept_clearance') if r else None,old_solver_error_events=len(old.get('paper_error_events',[])),native_solver_error_events=len(r.get('paper_error_events',[])) if r else None))
    csv_write('PHYSICAL_NATIVE_VS_PRIOR.csv',detail)
    for cohort,method in sorted({(r['cohort'],r['method']) for r in detail}):
        rs=[r for r in detail if (r['cohort'],r['method'])==(cohort,method)];counts=collections.Counter(r['native_outcome'] for r in rs)
        summary.append(dict(cohort=cohort,method=method,control_period_s=rs[0]['control_period_s'],planned=len(rs),success=counts['success'],collision=counts['collision'],timeout=counts['timeout'],unknown=counts['unknown'],pending=counts['pending'],
            old_success=sum(r['old_outcome']=='success' for r in rs),reused=sum(r['reused'] for r in rs)))
    csv_write('SUMMARY.csv',summary)
    oldmatched=[r for r in csv_read(OLD/'MATCHED_EXISTING_BASELINES.csv') if r['method']=='circulation']
    matched=[]
    for row in oldmatched:
        for method in ['Pipeline','dac','mgr','mcbf','circulation']:
            if method=='Pipeline':success=int(row['pipeline_success']) if row['pipeline_success'] else None;outcome='success' if success==1 else 'failure' if success==0 else 'missing';uid=row['pipeline_uid'];seconds=row['pipeline_seconds']
            else:
                r,ref=lookup[method,row['physical_key']];success=r['success'] if r else None;outcome=r['outcome'] if r else 'pending';uid=ref['native_uid'];seconds=r.get('completion_time') if r else None
            matched.append(dict(scope=row['scope'],cohort=row['cohort'],controller=row['controller'],physical_key=row['physical_key'],method=method,outcome=outcome,success=success,completion_seconds=seconds,uid=uid,
                prior_cohort=row['prior_cohort'],prior_group=row['prior_group'],prior_failure_types=json.loads(row['prior_failure_types']),prior_d2=row['prior_d2']=='True'))
    csv_write('MATCHED_WITH_C1_NATIVE.csv',matched)
    mainrows=[r for r in matched if r['prior_cohort']=='test' and r['prior_group'] in ['甲组地图','乙组地图']]
    tags={'同通道对向互阻候选','路口或相邻通道互阻候选','被已到达的静止机器人阻挡'}
    predicates={'D1-proxy':lambda r:'开阔区域互阻候选' in r['prior_failure_types'],'D2':lambda r:r['prior_d2'],
        'D3-proxy-union':lambda r:bool(tags.intersection(r['prior_failure_types']))}
    strata=[]
    for group,predicate in predicates.items():
        for method in LABELS:
            rs=[r for r in mainrows if r['method']==method and predicate(r)];known=[r for r in rs if r['success'] is not None and r['outcome'] not in ['pending','unknown']]
            strata.append(dict(group=group,method=method,success=sum(r['success'] for r in known),known=len(known),logical_pairs=len(rs),physical_instances=len({r['physical_key'] for r in rs}),pending=sum(r['outcome']=='pending' for r in rs),unknown=sum(r['outcome']=='unknown' for r in rs)))
    csv_write('D123_NATIVE_COMPARISON.csv',strata)
    authors=[]
    for old in csv_read(PAIRED/'AUTHOR_SCENE_COMPARISON.csv'):
        if old['method'] in ['circulation','mcbf']:
            r,ref=lookup[old['method'],old['scene']];row=dict(old);row.update(outcome=r['outcome'] if r else 'pending',success=r['success'] if r else None,
                completion_seconds=r.get('completion_time') if r else None,uid=ref['native_uid'],trace=r.get('trace') if r else None,
                minimum_swept_clearance=r.get('minimum_swept_clearance') if r else None,control_period_s=ref['control_period_s'],score_period_s=ref['score_period_s'])
        else:
            row=dict(old);row.update(control_period_s=.02 if old['scene']=='circulation_author_default' else .05,score_period_s=.02 if old['scene']=='circulation_author_default' else .05)
        authors.append(row)
    csv_write('AUTHOR_SCENE_NATIVE_COMPARISON.csv',authors)
    mechanisms=[]
    # Preserve the old appendix's whole-initial-problem interpretation.
    for item in read(PAIRED/'MECHANISM_INITIAL_PROBLEMS.json'):
        row=dict(mechanism=item['mechanism'],state_id=item['state_id'],physical_identity=item['physical_identity'],C1_historical_uid=item['c1_uid'],C1_historical_success=item['c1_success'])
        for method,prior in item['new_methods'].items():
            if prior['reused']:old=read(prior['raw'])
            else:old=read(PAIRED/(prior['request_name']+'_RESULT.json'))
            ref=next(x for x in links if x['old_uid']==old['rollout_uid']);p=Path(ref['native_raw']);r=read(p) if p.exists() else None
            row[method]=dict(uid=ref['native_uid'],outcome=r['outcome'] if r else 'pending',success=r['success'] if r else None)
        row['interpretation']='whole initial problem; historical C1 fixed/diagnostic trajectory; no same-mid-state takeover certification';mechanisms.append(row)
    csv_write('MECHANISM_INITIAL_NATIVE_COMPARISON.csv',mechanisms)
    completed=sum(r['native_outcome']!='pending' for r in detail);audit=read(D/'FINAL_CHECK.json') if (D/'FINAL_CHECK.json').exists() else {}
    status='完成并通过复核' if audit.get('passed') else f'补跑/复核中：{completed}/{len(detail)} 条记录有结果'
    caveats=['外部方法按作者原生控制周期运行；仍是共同SI端口，不将其等同于作者原始单车/二阶动力学。',
        'DAC保留作者外层.10秒、内部形状积分.05秒与原计数；环流.02秒；MGR/MCBF .05秒。参数、求解器和nominal保持此前固定版本。',
        '控制间隔内零阶保持；沿每条SI线段做扫掠碰撞检查。成功仍在原场景评分时刻判定，碰撞优先，真实任务时限保持不变；最后不足一个控制周期只推进剩余时间。',
        'C1沿用原场景控制周期与冻结40步窗口，未用新结果调参；主D123为.05秒。Chen当前有效作者场景h=执行=.15秒，直接复用既有证据。',
        'C1在环流作者场景的描述子仍标记为训练分布外，沿用上一轮冻结部署；此次只修正外部方法频率，不重新训练或调参。',
        'D1/D3为旧运行级代理，D2沿用旧活跃约束标签；类别可重叠。逻辑配对跨nominal引用同一物理轨迹，不增加独立样本数。超时不直接当作死锁。',
        'MCBF单机作者场景中的C1单机安全层仍为未验证动态接口诊断；其障碍速度项缺口不由此次频率修正解决。',
        '任务成功时间按仿真物理秒数统计。共享资源调度可能暂停进程，保存的墙钟耗时包含等待，不用于计算效率排名。',
        '此前共同步长结果保留为历史端口结果；本目录原生频率结果是新的主要对比入口。']
    lines=['外部方法原生控制频率比较','状态：'+status,'',*caveats,'','总体记录：']
    lines.extend(str(r) for r in summary);lines+=['','主D123逻辑配对：'];lines.extend(str(r) for r in strata)
    dump(D/'RESULTS.json',dict(status=status,completed=completed,total=len(detail),summary=summary,strata=strata,caveats=caveats,new_training=0))
    (D/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    body='<h1>外部方法：原生控制频率比较</h1><p>'+html.escape(status)+'</p>'+''.join('<p>'+html.escape(s)+'</p>' for s in caveats)
    body+='<h2>作者场景</h2>'+table(authors,['scene','label','control_period_s','score_period_s','outcome','completion_seconds'])
    body+='<h2>D1/D2/D3旧分层</h2>'+table(strata,['group','method','success','known','logical_pairs','physical_instances','pending','unknown'])
    body+='<h2>全部场景</h2>'+table(summary,['cohort','method','control_period_s','planned','success','collision','timeout','unknown','pending','old_success'])
    body+='<h2>Chen：保留既有原生周期结果</h2><p>有效作者场景的模型与执行周期均为0.15秒，直接复用。旧广泛几何端口已废弃；此处不将其成绩并入总体排名。<a href="../method_comparison_success_time_20261102/chen_author_mpc.csv">既有逐场景比较</a> · <a href="../chen_b_coverage_and_d2_closed_loop_v1/REPORT.md">复现和适用范围</a></p>'
    body+='<p><a href="REPORT.txt">完整说明</a> · <a href="PHYSICAL_NATIVE_VS_PRIOR.csv">逐例新旧结果</a> · <a href="D123_NATIVE_COMPARISON.csv">D123表</a> · <a href="AUTHOR_SCENE_NATIVE_COMPARISON.csv">作者场景表</a> · <a href="MECHANISM_INITIAL_NATIVE_COMPARISON.csv">机制附表</a></p>'
    (D/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>原生频率对比</title><style>body{font:16px sans-serif;max-width:1350px;margin:30px auto;padding:20px}p{line-height:1.7}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'+body)
    print(status)

if __name__=='__main__':main()
