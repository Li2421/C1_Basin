"""Complete database/content/trace audit and the scoped Chinese delivery."""
from common import *
import subprocess,numpy as np

LABEL={'knn_dedup':'近邻选择器·去重训练池','ridge_dedup':'岭回归选择器·去重训练池',
    'knn_full':'近邻选择器·全18场景池','ridge_full':'岭回归选择器·全18场景池',
    'global_fixed':'全局最佳单一修正','eta13':'历史候选η13','safety':'仅安全过滤'}

def database():
    from controller import result
    seen={};counts={};hashes={};metadata={};checkpoints=0
    def digest(p):
        p=str(p)
        if p not in hashes:hashes[p]=sha(p)
        return hashes[p]
    with dbm.connect(True)as db:
        for stage in ('capability','pilot','main'):
            cases=read(HERE/f'{stage}_CASES.json');counts[stage]=len(cases)
            for c in cases:
                rid=c['rollout_uid']
                # Check each occurrence's metadata, including reused pilot cells.
                for table,key,want in [('scenario','scenario_uid',canonical(c['scenario_payload'])),
                    ('controller_config','controller_uid',canonical(c['controller_payload']))]:
                    identity=(table,c[key]);col='metadata_json'if table=='scenario'else 'config_json'
                    if identity not in metadata:
                        row=db.execute(f'SELECT * FROM {table} WHERE {key}=?',(c[key],)).fetchone();assert row is not None
                        metadata[identity]=row[col]
                    assert metadata[identity]==want,('METADATA_CONTENT',identity)
                physical=canonical({k:c['state'][k]for k in ('positions','velocities','goals')})
                identity=('state',c['state_uid'])
                if identity not in metadata:
                    row=db.execute('SELECT * FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone();assert row is not None
                    metadata[identity]=(row['scenario_uid'],row['physical_state_json'])
                assert metadata[identity]==(c['scenario_uid'],physical)
                if rid in seen:continue
                r=result(stage,c);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();assert row is not None,('DB_MISSING',rid)
                for k in ('scenario_uid','state_uid','eta_uid','controller_uid','seed_key'):
                    if k in row.keys():assert row[k]==c[k]==r[k],(rid,k)
                for k in ('success','deadlock','timeout','collision','numerical_failure'):assert row[k]==r.get(k,0),(rid,k)
                assert not row['conflict_quarantined'];assert digest(r['trace_path'])==r['trace_sha256'],rid
                e=db.execute('SELECT * FROM eta WHERE eta_uid=?',(c['eta_uid'],)).fetchone()
                assert dbm.eta_identity(c['eta_a'])[2]==e['canonical_hex']
                if c.get('cross_scene_new'):
                    for p,h in c['controller_payload']['execution_sources'].items():assert digest(p)==h,p
                op=r.get('operational_resume')
                if op:
                    for p,h in op['sources'].items():assert digest(p)==h,p
                    receipts=set(op['saved_checkpoint_receipts'])
                    receipts.update(str(p)for p in(HERE/'operational_checkpoints'/rid).glob('*.receipt.json'))
                    if op.get('resumed_from'):
                        prior=op['resumed_from'];assert digest(prior['receipt'])==prior['receipt_sha256'];receipts.add(prior['receipt'])
                    for p in receipts:
                        snap=read(p);assert snap['scientific_label']is None and snap['case_hash']==uid('checkpoint_case',c)
                        assert snap['sources']==op['sources'] and digest(snap['path'])==snap['sha256'];checkpoints+=1
                seen[rid]=dict(stage=stage,trace_path=r['trace_path'],trace_sha256=r['trace_sha256'],unknown=int(r['numerical_failure']))
    journals=cache_writer.ROOT/'journals'/EXP
    pending=list(journals.glob('*.jsonl'))+list(journals.glob('*.tmp'));assert not pending,pending
    for stage in counts:
        reg=read(HERE/f'{stage}_REGISTRATION.receipt.json')
        assert reg['original_rows_before']==reg['original_rows_after'] and reg['only_insert_new_metadata']
    write(HERE/'DB_POSTFLIGHT.json',dict(passed=True,utc=utc(),stage_requests=counts,unique_requests=len(seen),
        all_requests_in_database=True,all_identities_and_results_equal=True,metadata_per_occurrence_equal=True,
        exact_eta_float64_bytes_equal=True,all_trace_hashes_equal=True,no_unmerged_experiment_journals=True,
        existing_metadata_rows_unchanged_at_registration=True,unknowns_retained_without_retry=True,operational_checkpoint_receipts=checkpoints))
    write(HERE/'VERIFIED_RESULT_PROVENANCE.json',seen)
    write(HERE/'VERIFIED_OUTPUT_SHA256.json',hashes)

def integrity():
    start=read(HERE/'INTEGRITY_START.json');repos={};exceptions={}
    def normalize(status):
        kept=[];excluded=[]
        for line in status.splitlines():
            p=line[3:]
            skip=p.startswith('diagnostics/cross_scene_prospective_v1/')or any(p.startswith(d.rstrip('/')+'/')for d in start['parallel_directories'])or (p.startswith('docs/prompt_')and p.endswith('.md')and line.startswith('??'))
            (excluded if skip else kept).append(line)
        return kept,excluded
    for repo,old in start['repositories'].items():
        status=subprocess.check_output(['git','-C',repo,'status','--porcelain']).decode()
        diff=subprocess.check_output(['git','-C',repo,'diff']);cached=subprocess.check_output(['git','-C',repo,'diff','--cached'])
        before,be=normalize(old['status']);after,ae=normalize(status)
        repos[repo]=dict(status=status,status_sha256=hashlib.sha256(status.encode()).hexdigest(),
            diff_sha256=hashlib.sha256(diff).hexdigest(),cached_diff_sha256=hashlib.sha256(cached).hexdigest(),
            normalized_status_equal=before==after,diff_equal=hashlib.sha256(diff).hexdigest()==old['diff_sha256'],
            cached_diff_equal=hashlib.sha256(cached).hexdigest()==old['cached_diff_sha256'])
        exceptions[repo]=dict(before=be,after=ae)
        (HERE/('END_'+Path(repo).name+'.status')).write_text(status)
    expected=read(HERE/'REUSED_EVIDENCE_SHA256_START.json')['files'];during={}
    for line in(HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():
        r=json.loads(line);prior=during.setdefault(r['path'],r['sha256']);assert prior==r['sha256'],('INPUT_CHANGED_DURING_RUN',r['path'])
    for p,h in during.items():
        if p in expected:assert expected[p]==h,p
        expected[p]=h
    # Inventory every loaded project module; copies are private, external modules
    # are read-only. Their baseline is captured before rollout in IMPORTED_SOURCES.
    imported=HERE/'IMPORTED_SOURCES_START.json'
    if imported.exists():expected.update(read(imported)['files'])
    changed={};end={}
    for p,h in expected.items():
        actual=sha(p);end[p]=actual
        if actual!=h:changed[p]=dict(before=h,after=actual)
    frozen_changed={}
    for receipt in ('MODEL_FROZEN.receipt.json','EXECUTION_FROZEN.receipt.json'):
        if not(HERE/receipt).exists():continue
        for p,h in read(HERE/receipt)['files'].items():
            actual=sha(p)
            if actual!=h:frozen_changed[p]=dict(before=h,after=actual)
    write(HERE/'REUSED_EVIDENCE_SHA256_END.json',dict(files=end))
    ok=not changed and not frozen_changed and all(r['normalized_status_equal']and r['diff_equal']and r['cached_diff_equal']for r in repos.values())
    write(HERE/'INTEGRITY_END.json',dict(utc=utc(),repositories=repos,exceptions=exceptions,passed=ok))
    write(HERE/'INTEGRITY_COMPARE.json',dict(passed=ok,repositories=repos,exceptions=exceptions,
        input_file_count=len(expected),changed_inputs=changed,frozen_changed=frozen_changed,
        scope='Both repositories; own experiment/registered parallel activity/new docs prompt files separately listed. Actual reused inputs all rehashed; broad initial inventory retained.'))
    assert ok,('INTEGRITY_FAILED',changed,frozen_changed)

def report():
    stats=read(HERE/'STATISTICS.json');scenes=read(HERE/'SCENES.json');choices=read(HERE/'FROZEN_CHOICES.json')
    names={s['name']:s['full_name']for s in scenes};selected=read(HERE/'SELECTED_CONTROLLERS.json');tables=read(HERE/'TEST_LABEL_TABLES.json')
    primary=stats['primary'];pp=lambda x:f'{100*x:+.2f}';pct=lambda x:f'{100*x:.2f}%'
    def comparison(row):return f"{pp(row['estimate'])}个百分点，95%区间[{pp(row['ci'][0])}, {pp(row['ci'][1])}]"
    verdict='通过'if primary['verdict']=='PASSED'else '未通过'
    first=f"主判定{verdict}：近邻选择器（knn，去重训练池）减全局最佳单一修正（global_fixed，扩展范围候选sE29：朝目标0.34、侧移0.55、互斥1.20）为{comparison(primary)}。"
    first+=f"次要结果：岭回归选择器（ridge，去重训练池）为{comparison(stats['comparisons']['ridge_dedup'])}；"
    first+=f"全18场景池近邻为{comparison(stats['comparisons']['knn_full'])}，全18场景池岭回归为{comparison(stats['comparisons']['ridge_full'])}，均相对同一全局最佳。\n\n"
    lines=[first,'结论仅限在本修正族、本控制器集与本安全层下，对这12个新场景。瓶颈上的近邻对11个布局选取同一组修正，结果解释为“瓶颈族修正对全局修正”；不称为逐场景自适应，不外推未测拓扑。未通过不等于证明无效。\n\n',
        '近邻瓶颈族修正为扩展范围候选（sE05，朝目标0.11、侧移0.32、互斥0.53）；其他候选的精确float64参数见ACTIONS来源与FROZEN_CHOICES.json。训练标签全部复用上一轮18场景×49候选，不新增训练标签、不重训控制器。去重剩11场景，近邻k=2，全池k=3；两版岭回归λ=100；两池全局最佳均为sE29。全部模型选择、场景特征和统计代码在能力门之前冻结。\n\n',
        '场景之间等权，场景内合格控制器类型等权；正式评估仅种子0。unknown计失败，同时报告其计成功时的成功率上界。区间来自5000次场景→控制器类型→状态分层成对自助法；选择固定，不重新拟合训练池。\n\n',
        '| 新测试场景 | 去重近邻 | 去重岭回归 | 全池近邻 | 全池岭回归 | 全局固定 | 历史候选η13（eta13） | 仅安全过滤（safety） |\n|---|---:|---:|---:|---:|---:|---:|---:|\n']
    from analysis_stats import METHODS
    for n,row in stats['per_scene'].items():lines.append(f"| {names[n]}（{n}） | "+' | '.join(pct(row['rates'][m])for m in METHODS)+' |\n')
    lines.extend(['\n| 分层 | 去重近邻−全局固定 | 去重岭回归−全局固定 | 全池近邻−全局固定 | 全池岭回归−全局固定 |\n|---|---|---|---|---|\n'])
    for key,label in [('bottleneck','11个瓶颈布局'),('crowded14','拥挤随机14机')]:
        lines.append(f'| {label} | '+' | '.join(comparison(stats['strata'][key]['comparisons'][m])for m in METHODS[:4])+' |\n')
    lines.append('\n| 预登记稳健比较 | 差值中位数（百分点） | 两端各去1场景截尾均值 | Wilcoxon双侧p | 符号检验双侧p | 胜/负/平 |\n|---|---:|---:|---:|---:|---:|\n')
    for m,r in stats['robust'].items():lines.append(f"| {LABEL[m]} | {pp(r['median'])} | {pp(r['trimmed_mean_one_per_tail'])} | {r['wilcoxon_two_sided_p']:.6g} | {r['sign_test_two_sided_p']:.6g} | {r['wins']}/{r['losses']}/{r['ties']} |\n")
    lines.append('\n| 方法 | 主值成功率 | unknown计成功上界 |\n|---|---:|---:|\n')
    for m,r in stats['rates'].items():lines.append(f"| {LABEL[m]}（{m}） | {pct(r['rate'])} | {pct(r['unknown_success_upper'])} |\n")
    lines.append(f"\n近邻与全局最佳选择不同的场景数为{stats['different_knn_global_scenes']}/12。下表为已评估候选中的事后最佳，只作描述，不用于判定。\n\n| 场景 | 已评估候选中的事后最佳 | 成功率 |\n|---|---|---:|\n")
    for n,r in stats['per_scene'].items():lines.append(f"| {names[n]}（{n}） | {r['evaluated_posthoc_best']} | {pct(r['evaluated_posthoc_best_rate'])} |\n")
    lines.append('\n四局完整历史逐位重放通过（双瓶颈与拥挤随机8机各两局，排除墙钟计时）。11个瓶颈布局各16个状态均几何有效并有单机器人路径见证；联合可解未认证。能力门失败类型逐场景排除，详见CAPABILITY_TABLE.md。timeout未转换为deadlock，正式死锁解决证据未建立；碰撞按机器人、静态和移动分类保留，数值失败保留unknown、不重跑。\n\n')
    lines.append('各阶段官方预检、复用与执行数：\n\n| 阶段 | 请求数 | 已有结果保留/复用 | 预检后执行缺失请求 |\n|---|---:|---:|---:|\n')
    for stage in ('capability','pilot','main'):
        r=read(HERE/f'{stage}_PREFLIGHT.receipt.json');lines.append(f"| {stage} | {r['summary']['total_requested']} | {r['reused_count']} | {r['execution_count']} |\n")
    lines.append('\n能力门确定性实例种子1对经验证种子0结果作零步引用，保留来源。完整请求身份、元数据、标签、轨迹哈希与数据库一致，所有本轮journal已由单一官方merger合库；两个仓库与所有复用输入/冻结文件结束核对通过。实测成本见PILOT_COST.md，证据适用范围见GAPS.md。没有修改旧源码、控制器参数、安全层、物理或成功判据；没有视频交付要求。\n')
    (HERE/'REPORT.md').write_text(''.join(lines))
    (HERE/'AXIS_TABLE.md').write_text('| 泛化框架 | 本轮证据 | 结果与范围 |\n|---|---|---|\n'+
        f'| 跨场景·前瞻 | 18场景训练标签去重→12个新测试场景，选择在能力门前冻结 | 主判定{verdict}，{comparison(primary)} |\n'+
        '| 跨因素·几何新布局 | 11种新瓶颈布局，固定半径与动力学 | 瓶颈族修正对全局修正；联合可解未认证 |\n'+
        '| 跨因素·密度 | 拥挤随机14机，固定半径、运动参数与工作空间描述 | 单场景分层描述，不外推其他密度/拓扑 |\n')
    gap=['# 适用范围与缺口\n\n','要求的能力门、计时、正式评估、统计、数据库与结束完整性核对全部完成。\n\n',
        '- 不认证联合可解或正式死锁解决；timeout不等于deadlock。\n',
        '- 仅解释本修正族、本控制器集与本安全层下的12个新场景；瓶颈为族级固定修正比较。\n',
        '- 能力门造成各场景可评估类型不同；不强行补齐失败类型。\n',
        '- unknown主值计失败，上界计成功，不重跑。\n',
        '- 区间条件于冻结训练池与模型选择，不代表重采样完整训练过程。\n',
        '- 2状态×2候选试跑是成本估计，不能保证规划尾部的最长时长。\n',
        '- 未评估全部49候选，事后最佳仅在各场景已评估候选中定义。\n']
    (HERE/'GAPS.md').write_text(''.join(gap))
    layer=['# 五层证据\n\n','| 场景 | 几何有效 | 单机器人有路 | 联合可解 | 合格类型数 | 控制器成功（去重近邻） | 死锁解决 |\n|---|---|---|---|---:|---:|---|\n']
    for s in scenes:
        n=s['name'];count=sum(c is not None for c in selected[n].values())
        if s['family']=='bottleneck':
            r=read(HERE/'geometry'/f'{n}.json');valid=f"{sum(x['geometric_valid']for x in r['states'])}/16";paths=f"{sum(x['single_robot_paths']for x in r['states'])}/16"
        else:valid='来源状态与无界几何有效';paths='无墙直线路径'
        rate=stats['per_scene'][n]['rates']['knn_dedup'];layer.append(f'| {s["full_name"]}（{n}） | {valid} | {paths} | 未认证 | {count} | {pct(rate)} | 未建立 |\n')
        p=HERE/'five_layer'/f'{n}.md';p.parent.mkdir(exist_ok=True)
        p.write_text(f'# {s["full_name"]}（{n}）五层证据\n\n1. 几何有效：{valid}。\n\n2. 单机器人有路：{paths}；仅路径见证。\n\n3. 联合可解：未认证。\n\n4. 控制器成功：{count}类通过单机器人能力门；去重近邻成功率{pct(rate)}，其他方法见REPORT.md。\n\n5. 死锁解决：未建立。未把timeout或几何不可达标为deadlock。\n')
    (HERE/'FIVE_LAYER.md').write_text(''.join(layer))
    reports=('REPORT.md','AXIS_TABLE.md','HISTORY_CHECK.md','MODEL_FROZEN.receipt.json','FIVE_LAYER.md','CAPABILITY_TABLE.md',
        'PILOT_COST.md','GAPS.md','DB_POSTFLIGHT.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','STATISTICS.json')
    write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',primary=primary,reports={n:sha(HERE/n)for n in reports}))
    print(first,flush=True)

def main():
    database();integrity();report()

if __name__=='__main__':main()
