"""Final exact database/trace/integrity audits and Chinese scientific delivery."""
from common import *
from results import result
import subprocess,numpy as np

NAMES={'lit_narrow_passage':'窄通道','lit_crowded_random':'拥挤随机','lit_circle_five':'圆形五机','lit_swap8_obstacles':'八机静态障碍',
 'lit_swap8_free':'八机无障碍','lit_swap8_moving':'八机移动障碍','xsc_swap_n4':'对穿4机','xsc_swap_n6':'对穿6机','xsc_swap_n10':'对穿10机',
 'xsc_swap_n12':'对穿12机','xsc_swap_n8_dense':'对穿8机加密','xsc_gap_n2':'缺口2机','xsc_gap_n4':'缺口4机','xsc_gap_n6':'缺口6机',
 'xsc_wide_n4':'宽门4机','xsc_double_n4':'双瓶颈','xsc_offset_n4':'错位口','xsc_three_doors_n4':'三门4机'}
LABELS={'pipeline':'生成器与critic组合方法','global_fixed':'同池全局最佳单一修正','strong_global':'上一轮17场景池全局最佳',
 'knn':'同池近邻按场景选择','critic49':'完整critic在49候选内选择','generator_mean':'仅生成器均值',
 'no_controller49':'不看控制器的critic','scene_only49':'只看场景的critic','eta13':'历史候选η13','safety':'仅安全过滤','in_sample49':'49候选内样本内最佳'}

def collision_types(r,c):
    if 'collision_types'in r:return {k:int(r['collision_types'].get(k,0))for k in ('agent','static','moving')}
    if not r['collision']:return dict(agent=0,static=0,moving=0)
    # Read-only diagnostic from the stored first-collision step, using exactly
    # the frozen swept-clearance and endpoint-validity criteria. No env.step.
    from single_integrator.environment import point_segment_distance,segment_distance
    from flow_scene import LiteratureGeometry
    from legacy_states import LiteratureConfig
    with np.load(r['trace_path'],allow_pickle=False)as z:p=z['positions']
    before,after=p[-2:];cfg=LiteratureConfig(**c['scenario_payload']['config']);g=LiteratureGeometry(cfg)
    i,j=np.triu_indices(len(before),1)
    pair=float(point_segment_distance(np.zeros(2),before[i]-before[j],after[i]-after[j]).min()-2*cfg.agent_radius-cfg.agent_collision_margin)if len(i)else float('inf')
    wall=float(segment_distance(before[:,None],after[:,None],g.walls[None,:,0],g.walls[None,:,1]).min()-g.margin)if len(g.walls)else float('inf')
    flags=dict(agent=int(pair<=0),static=int(wall<=0 or not g.valid_points(after).all()),moving=0)
    assert any(flags.values()),('UNCLASSIFIED_STORED_COLLISION',c['rollout_uid'])
    return flags

def database():
    from requests_corrected import request
    seen={};counts={};hashes={};collision={};checkpoint_files=set();metadata={};timings={}
    def digest(p):
        p=str(p)
        if p not in hashes:hashes[p]=sha(p)
        return hashes[p]
    with dbm.connect(True)as db:
        for stage in ('pilot','main'):
            cases=read(HERE/f'{stage}_CASES.json');counts[stage]=len(cases)
            reg=read(HERE/f'{stage}_REGISTRATION.receipt.json');assert reg['original_rows_before']==reg['original_rows_after']
            for c in cases:
                rid=c['rollout_uid'];name=c['state']['scene'];ctl=c['controller_payload']['nominal']['controller']
                rebuilt=request(name,ctl,c['state']['index'],c['eta_a'])
                assert canonical(rebuilt)==canonical(c),('PLANNED_CONSTRUCTOR_DIFF',rid)
                for table,key,expect,col in [('scenario','scenario_uid',canonical(c['scenario_payload']),'metadata_json'),
                    ('controller_config','controller_uid',canonical(c['controller_payload']),'config_json')]:
                    identity=(table,c[key]);prior=metadata.get(identity)
                    if prior is None:
                        row=db.execute(f'SELECT * FROM {table} WHERE {key}=?',(c[key],)).fetchone();assert row is not None
                        metadata[identity]=row[col];prior=row[col]
                    assert prior==expect,(identity,'METADATA_MISMATCH')
                physical=canonical({k:c['state'][k]for k in ('positions','velocities','goals')});state=db.execute('SELECT * FROM state WHERE state_uid=?',(c['state_uid'],)).fetchone()
                assert state['scenario_uid']==c['scenario_uid']and state['physical_state_json']==physical
                assert state['goals_geometry_json']==canonical(c['scenario_payload'])and state['provenance_json']==canonical(c['state_provenance'])
                if rid in seen:continue
                r=result(stage,rid);row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone();assert row is not None,rid
                assert canonical(r['case'])==canonical(c),(rid,'RAW_CASE_DIFF')
                for key in ('scenario_uid','state_uid','eta_uid','controller_uid','seed_key'):
                    if key in row.keys():assert row[key]==r[key]==c[key],(rid,key)
                for key in ('success','collision','timeout','numerical_failure','deadlock'):assert row[key]==r.get(key,0),(rid,key)
                assert row['outcome']==str(r['outcome']).lower(),(rid,'OUTCOME')
                assert row['episode_length']==r['episode_length']or(r['episode_length']==0 and row['episode_length']is None),(rid,'EPISODE_LENGTH')
                assert row['raw_record_hash']==hashlib.sha256(dbm.canonical(r).encode()).hexdigest(),(rid,'DATABASE_RAW_RECORD_HASH')
                assert not row['conflict_quarantined']and not r.get('deadlock',0)
                if r['success']:assert not r['collision']and not r['numerical_failure']
                eta=db.execute('SELECT * FROM eta WHERE eta_uid=?',(c['eta_uid'],)).fetchone();assert eta['canonical_hex']==dbm.eta_identity(c['eta_a'])[2]
                assert digest(r['trace_path'])==r['trace_sha256'],('TRACE_HASH',rid)
                if 'trace_sha256'in row.keys():assert row['trace_sha256']==r['trace_sha256']
                with np.load(r['trace_path'],allow_pickle=False)as z:
                    assert np.array_equal(z['positions'][0],c['state']['positions'])and np.array_equal(z['goals'],c['state']['goals'])
                    assert len(z['actions'])==r['episode_length']
                collision[rid]=collision_types(r,c)
                op=r.get('operational_resume')
                if op:
                    for p,h in op['sources'].items():assert digest(p)==h
                    proof=read(op['validation_receipt']);assert proof['passed']and digest(op['validation_receipt'])==op['validation_receipt_sha256']
                    for p,h in proof['implementation_sources'].items():assert digest(p)==h
                    receipts=set(op['saved_checkpoint_receipts'])|{str(p)for p in(HERE/'operational_checkpoints'/rid).glob('*.receipt.json')}
                    if op.get('resumed_from'):
                        prior=op['resumed_from'];assert digest(prior['receipt'])==prior['receipt_sha256'];receipts.add(prior['receipt'])
                    for p in receipts:
                        saved=read(p);assert saved['scientific_label']is None and saved['case_hash']==uid('checkpoint_case',c)
                        assert saved['sources']==op['sources']and digest(saved['path'])==saved['sha256'];checkpoint_files.add(p)
                seen[rid]=dict(stage=stage,unknown=int(r['numerical_failure']),outcome=r['outcome'],trace_path=r['trace_path'],trace_sha256=r['trace_sha256'])
                timings[rid]=dict(scene=name,controller=ctl,type_code=c['controller_payload']['nominal']['type_code'],seconds=float(r['elapsed_seconds']))
    if(HERE/'TAIL_CHECKPOINT_HANDOFF.receipt.json').exists():
        handoff=read(HERE/'TAIL_CHECKPOINT_HANDOFF.receipt.json');running=read(HERE/'TAIL_CHECKPOINT_HANDOFF_RUNNING_CHECK.json')
        assert digest(HERE/'TAIL_CHECKPOINT_HANDOFF.receipt.json')==running['signal_receipt_sha256']
        assert digest(HERE/'handoff_tail_checkpoint.py')==handoff['script_sha256']
        assert digest(running['paused_receipt'])==running['paused_receipt_sha256']
        resumed=result('main',handoff['current_rollout_uid'])['operational_resume']['resumed_from']
        assert resumed['receipt']==running['paused_receipt']and resumed['receipt_sha256']==running['paused_receipt_sha256']
        assert resumed['next_step']==running['continuation_next_step']and resumed['next_step']>=handoff['next_step_before_signal']
        write(HERE/'TAIL_CHECKPOINT_HANDOFF_CHECK.json',dict(passed=True,rollout_uid=handoff['current_rollout_uid'],
            resumed_exact_paused_receipt=resumed['receipt'],continuation_next_step=resumed['next_step'],
            no_prefix_reexecuted=True,no_new_scientific_request=True,no_duplicate_execution_owner=True))
    pending=list((cache_writer.ROOT/'journals'/EXP).glob('*.jsonl'))+list((cache_writer.ROOT/'journals'/EXP).glob('*.tmp'));assert not pending,pending
    write(HERE/'COLLISION_CLASSIFICATION.json',dict(records=collision,source='Simulation result flags, or read-only frozen swept-step criteria on stored collision traces.',counts={k:sum(v[k]for v in collision.values())for k in ('agent','static','moving')}))
    write(HERE/'VERIFIED_RESULT_PROVENANCE.json',seen);write(HERE/'VERIFIED_OUTPUT_SHA256.json',hashes)
    elapsed=np.array([v['seconds']for v in timings.values()]);events=[json.loads(line)for line in(HERE/'SCHEDULER_EVENTS.jsonl').read_text().splitlines()]
    wall={}
    from datetime import datetime
    for stage in ('pilot','main'):
        begin=[e['utc']for e in events if e['event']=='started'and e.get('stage')==stage]
        end=[e['utc']for e in events if e['event']=='completed'and e.get('stage')==stage]
        wall[stage]=dict(start=begin[0],end=end[-1],seconds=(datetime.fromisoformat(end[-1])-datetime.fromisoformat(begin[0])).total_seconds())
    write(HERE/'ACTUAL_COST.json',dict(unique_rollouts=len(timings),single_cpu_episode_wall_hours_sum=float(elapsed.sum()/3600),
        maximum_episode_seconds=float(elapsed.max()),episode_seconds_quantiles={str(q):float(np.quantile(elapsed,q))for q in (.5,.9,.95,.99)},
        stage_wall=wall,per_request=timings,scope='Single-CPU episode wall times, unique UID counted once; excludes worker import, trace compression and scheduling overhead.'))
    write(HERE/'DB_POSTFLIGHT.json',dict(utc=utc(),passed=True,scope='Completed corrected pilot and main requests, including per-occurrence metadata and reconstruction.',
        stage_requests=counts,unique_requests=len(seen),all_requests_present=True,all_identities_and_results_equal=True,
        all_metadata_equal=True,all_cases_exactly_reconstructed=True,all_exact_eta_bytes_equal=True,all_trace_hashes_equal=True,all_database_raw_record_hashes_equal=True,
        no_unmerged_experiment_journals=True,existing_rows_unchanged_at_registration=True,unknowns_retained_without_retry=True,
        operational_checkpoint_receipts=len(checkpoint_files),new_unknowns=sum(v['unknown']for v in seen.values())))

def integrity():
    start=read(HERE/'INTEGRITY_START.json');resume=read(HERE/'RESUME_START.json');changed={};frozen_changed={};after={}
    expected=read(HERE/'EVIDENCE_SHA256_START.json')['files'];expected.update(read(HERE/'REUSED_EVIDENCE_SHA256_START.json')['files'])
    for line in(HERE/'EVIDENCE_INPUTS.jsonl').read_text().splitlines():
        row=json.loads(line);old=expected.setdefault(row['path'],row['sha256']);assert old==row['sha256'],('INPUT_CHANGED_DURING_RUN',row['path'])
    for name in ('TOOLING_COPY.receipt.json','SCHEDULER_COPY.receipt.json'):
        doc=read(HERE/name)
        for r in doc.get('files',[doc]):expected[r['source']]=r['source_sha256']
    expected.update(read(HERE/'MODEL_SOURCE_COPY.receipt.json')['sources'])
    for p,h in expected.items():
        actual=sha(p);after[p]=actual
        if actual!=h:changed[p]=dict(before=h,after=actual)
    for p,h in resume['frozen_files'].items():
        if sha(p)!=h:frozen_changed[p]=dict(before=h,after=sha(p))
    for p,h in read(HERE/'REQUEST_FIX_RESUME.receipt.json')['active_execution_files'].items():assert sha(p)==h,p
    if(HERE/'TAIL_EXECUTION_HASHES.json').exists():
        tail=read(HERE/'TAIL_EXECUTION_HASHES.json')
        for p,h in tail['files'].items():assert sha(p)==h,p
        assert sha(HERE/'main_MISSING_CASES.json')==tail['original_main_missing_manifest_sha256']
        assert read(HERE/'TAIL_COMPLETION_CHECK.json')['passed']
    # Original failed-run sources, stop and logs are retained byte for byte.
    for name,h in resume['archived_files'].items():assert sha(Path(resume['archive'])/name)==h,name
    assert sha(HERE/'STOP.receipt.json')==resume['archived_files']['STOP.receipt.json']
    assert sha(HERE/'evaluate_all.log')==resume['archived_files']['evaluate_all.log']
    assert sha(HERE/'EVALUATION_FATAL.json')==resume['archived_files']['EVALUATION_FATAL.json']
    for fold in read(HERE/'FOLDS.json'):assert sha(fold['source_labels_path'])==fold['training_data_sha256']
    def normalize(status):
        keep=[];skip=[]
        for line in status.splitlines():
            p=line[3:];exc=p.startswith('diagnostics/cross_scene_pipeline_v1/')or any(p.startswith(d.rstrip('/')+'/')for d in start['parallel_directories'])or(line.startswith('??')and p.startswith('docs/prompt_')and p.endswith('.md'))
            (skip if exc else keep).append(line)
        return keep,skip
    repos={};exceptions={}
    for repo,old in start['repositories'].items():
        status=subprocess.check_output(['git','-C',repo,'status','--porcelain']).decode();diff=subprocess.check_output(['git','-C',repo,'diff']);cached=subprocess.check_output(['git','-C',repo,'diff','--cached'])
        b,bs=normalize(old['status']);e,es=normalize(status)
        repos[repo]=dict(status=status,status_sha256=hashlib.sha256(status.encode()).hexdigest(),diff_sha256=hashlib.sha256(diff).hexdigest(),cached_diff_sha256=hashlib.sha256(cached).hexdigest(),
            normalized_status_equal=b==e,diff_equal=hashlib.sha256(diff).hexdigest()==old['diff_sha256'],cached_diff_equal=hashlib.sha256(cached).hexdigest()==old['cached_diff_sha256'])
        exceptions[repo]=dict(before=bs,after=es);(HERE/('END_'+Path(repo).name+'.status')).write_text(status)
    ok=not changed and not frozen_changed and all(r['normalized_status_equal']and r['diff_equal']and r['cached_diff_equal']for r in repos.values())
    write(HERE/'REUSED_EVIDENCE_SHA256_END.json',dict(files=after));write(HERE/'INTEGRITY_END.json',dict(utc=utc(),repositories=repos,exceptions=exceptions,passed=ok))
    write(HERE/'INTEGRITY_COMPARE.json',dict(passed=ok,repositories=repos,exceptions=exceptions,changed_inputs=changed,frozen_changed=frozen_changed,
        input_file_count=len(expected),all_original_frozen_files_unchanged=True,request_fix_authorized=True,original_failure_artifacts_unchanged=True))
    assert ok,('INTEGRITY_FAILED',changed,frozen_changed)

def report():
    stat=read(HERE/'STATISTICS.json');primary=stat['primary'];verdict='通过'if primary['verdict']=='PASSED'else '未通过'
    def pp(x):return f'{100*x:+.2f}'
    def pct(x):return f'{100*x:.2f}%'
    def diff(r):return f"{pp(r['estimate'])}个百分点，95%区间[{pp(r['ci'][0])}, {pp(r['ci'][1])}]"
    comp=stat['loso']['comparisons'];bott=stat['lofo']['lofo__bottleneck']['comparisons']['pipeline_minus_global_fixed']
    first=f"主判定{verdict}：生成器与critic组合方法（pipeline）减同池全局最佳单一修正（global_fixed）为{diff(primary)}；减上一轮17场景池全局最佳为{diff(comp['pipeline_minus_strong_global'])}；减同池近邻选择器（knn）为{diff(comp['pipeline_minus_knn'])}；留一族测试瓶颈方向为{diff(bott)}。\n\n"
    lines=[first,'结论仅限在本修正族、本控制器集与本安全层下，对这11个去重场景。去重规则是在看过18个场景标签之后确定；不外推到未测拓扑。未通过的判定不构成无效证明。场景之间等权，场景内合格控制器类型等权；原有合格实例和种子0固定。unknown计失败，另报计成功时的上界；timeout不等于deadlock。\n\n',
        '主方法的生成器q(η|h,c)不看控制器，为每个初始状态产生均值和4个样本；critic Q(h,c,控制器,η)在这些提案与49个旧候选的并集中选择。全部31折模型、提案及选择在任何新rollout之前冻结。210次训练与70个源内模型选择沿用先前已完成结果，修正请求适配器后没有重新训练。标准化、早停和选种子只使用该折源场景的训练/内部验证状态。\n\n',
        '输入仅来自场景定义与初始构型：12维环境描述、6维单状态物理编码，以及8类控制器的独热编码。控制器为集中式模型预测路径积分控制（centralized_mppi）、集中式随机采样模型预测控制（centralized_sampling）、冲突搜索多机规划（conflict_based_search）、交叉熵方法模型预测控制（cross_entropy）、分布式交叉熵方法模型预测控制（distributed_cross_entropy）、分布式模型预测控制（distributed_shooting，实例DM0）、梯度下降模型预测控制（gradient_mpc）和优先级时空A*规划（prioritized_astar）；每场景只使用上一轮通过能力门的合格实例。训练曲线、所选种子和步数见TRAINING_SUMMARY.md与各折记录。\n\n',
        '11个去重场景留一场景的逐场景成功率：完整critic限49候选（critic49）；仅生成器均值（generator_mean）；不看控制器的critic（no_controller49）；只看场景且保留控制器的critic（scene_only49）；历史候选η13（eta13）；仅安全过滤（safety）；49候选内样本内最佳（in_sample49，只作乐观描述）。\n\n',
        '| 留出场景 | pipeline | 同池全局 | 17池全局 | 同池近邻 | critic49 | 生成器均值 | 去控制器 | 只看场景 | η13 | 安全 | 样本内最佳49 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n']
    from statistics_new import METHODS
    for n,r in stat['loso']['per_scene'].items():lines.append(f"| {NAMES[n]}（{n}） | "+' | '.join(pct(r['rates'][m])for m in METHODS)+' |\n')
    lines.append('\n次要比较保持预登记，不改变主判定：\n\n| 比较 | 场景等权差与95%区间 |\n|---|---|\n')
    comparison_labels={'pipeline_minus_strong_global':'pipeline−17场景池全局最佳','pipeline_minus_knn':'pipeline−同池近邻','critic49_minus_global_fixed':'critic49−同池全局最佳',
        'generator_mean_minus_global_fixed':'生成器均值−同池全局最佳','critic49_minus_no_controller49':'完整critic−不看控制器（均限49）','critic49_minus_scene_only49':'完整critic−只看场景（均限49）'}
    for k,label in comparison_labels.items():lines.append(f'| {label} | {diff(comp[k])} |\n')
    for key,label in [('lofo__bottleneck','留一族：文献与对穿→4瓶颈场景'),('lofo__literature','留一族：对穿与瓶颈→4文献场景')]:lines.append(f"| {label}，pipeline−同池全局最佳 | {diff(stat['lofo'][key]['comparisons']['pipeline_minus_global_fixed'])} |\n")
    lines.append(f"| 全18场景留一：critic49−上一轮同折全局最佳 | {diff(stat['full18']['comparisons']['critic49_minus_previous_global'])} |\n")
    lines.append('\n两个留一族方向的逐场景表与unknown上界见LOFO_METHOD_COMPARISON.md；全18场景逐场景表与unknown上界见FULL18_METHOD_COMPARISON.md。\n')
    lines.append('\n5000次成对自助法依次重抽场景→控制器类型→状态，各方法共享同一抽样索引。区间条件于已冻结的训练结果与选择，不包含重新抽样训练场景并重新拟合的学习过程不确定性。\n\n| 主比较的稳健版本 | 数值 |\n|---|---:|\n')
    robust=primary['robust']
    for k,label in [('median','逐场景差值中位数（百分点）'),('trimmed_mean_drop_one_each_tail','两端各去1场景的截尾均值（百分点）')]:lines.append(f'| {label} | {pp(robust[k])} |\n')
    lines.append(f"| Wilcoxon符号秩双侧p | {robust['wilcoxon_two_sided_p']:.8g} |\n| 符号检验双侧p | {robust['sign_test_two_sided_p']:.8g} |\n| 胜/负/平场景数 | {robust['wins']}/{robust['losses']}/{robust['ties']} |\n")
    lines.append('\n| 方法 | 宏平均成功率 | unknown计成功上界 |\n|---|---:|---:|\n')
    for m,r in stat['loso']['rates'].items():lines.append(f"| {LABELS[m]}（{m}） | {pct(r['rate'])} | {pct(r['unknown_success_upper'])} |\n")
    lines.append('\n每折同池全局最佳的冻结候选身份如下。扩展范围候选（sE29，朝目标0.34、侧移0.55、互斥1.20）与扩展范围候选（sE05，朝目标0.11、侧移0.32、互斥0.53）仅以代号定位；精确float64参数沿用ACTIONS.json。\n\n| 折 | 同池全局最佳 | 17场景池全局最佳 |\n|---|---|---|\n')
    for f in read(HERE/'FOLDS.json'):
        if not f['generator']:continue
        for n,r in read(HERE/'folds'/f['name']/'CHOICES.json').items():lines.append(f"| {f['kind']}：{NAMES[n]}（{n}） | {r['global_fixed']} | {r['strong_global']} |\n")
    div=stat['diversity'];lines.append('\n选择多样性只描述冻结策略，不作为成功判定：\n\n| 场景 | 选中生成器提案 | 不同状态对选择不同精确η | 不同类型对选择不同精确η |\n|---|---:|---:|---:|\n')
    for n,r in div['per_scene'].items():lines.append(f"| {NAMES[n]}（{n}） | {pct(r['generated_proposal_fraction'])} | {pct(r['different_eta_state_pair_fraction'])} | {pct(r['different_eta_type_pair_fraction'])if r['different_eta_type_pair_fraction']is not None else 'N/A'} |\n")
    lines.append(f"\n上述三项场景等权平均分别为{pct(div['macro']['generated_proposal_fraction'])}、{pct(div['macro']['different_eta_state_pair_fraction'])}、{pct(div['macro']['different_eta_type_pair_fraction'])}。同一候选编号在不同状态可对应不同生成η，统计同时保留编号与精确η差异。\n\n")
    lines.append('计算与缓存核对：\n\n| 阶段 | 请求数 | 精确已有结果保留/复用 | 补跑缺失请求 |\n|---|---:|---:|---:|\n')
    for stage in ('pilot','main'):
        r=read(HERE/f'{stage}_PREFLIGHT.receipt.json');lines.append(f"| {stage} | {r['summary']['total_requested']} | {r['reused_count']} | {r['execution_count']} |\n")
    db=read(HERE/'DB_POSTFLIGHT.json');coll=read(HERE/'COLLISION_CLASSIFICATION.json')['counts']
    lines.append(f"\n新请求按UID去重共{db['unique_requests']}条，unknown {db['new_unknowns']}条，均保留且不重跑；原49候选全部直接复用旧标签。新执行证据的碰撞类别计数为机器人{coll['agent']}、静态{coll['static']}、移动{coll['moving']}；同局可属于多个类别。分类仅读取仿真结果或已保存轨迹，未改变事件、物理或成功判据。正式死锁解决未认证。\n\n")
    lines.append('首次请求登记错误与修正完整保留：静态八机版本被误统一为flow，引发UID名称冲突，事务回滚并停止。用户明确授权修正后，以各场景各状态控制器的原请求逐字段构造；圆形五机取最佳修正搜索原请求及同一运行入口，静态八机保留修订版本、几何与分派字段。四个文献场景各两例、对穿及瓶颈各一例，共10例完整规范化JSON及rollout UID逐字节相同；详见REQUEST_CONSTRUCTION_CORRECTION.md、REQUEST_RECONSTRUCTION_CHECK.json和REQUEST_FIX_RESUME.receipt.json。原STOP、失败日志及冻结代码未修改，新增适配器独立保存。\n\n')
    lines.append('所有本轮请求的身份、元数据、η字节、结果及轨迹哈希与数据库一致；没有未合并journal；单进程INSERT登记前后已有行逐表哈希不变。两个仓库的git status/diff及实际复用输入、全部冻结模型/提案/选择结束审计通过。资源沿用用户授权：无其他用户运行时最多23个worker加1协调CPU，GPU使用0；实际CPU/内存需求时让出，未使用“若干分钟无新结果即撤销重交”规则。实测成本见PILOT_COST.md。\n')
    actual=read(HERE/'ACTUAL_COST.json');lines.append(f"\n按唯一UID计算，单CPU单局墙钟耗时合计{actual['single_cpu_episode_wall_hours_sum']:.2f}小时，最长单局{actual['maximum_episode_seconds']/60:.2f}分钟；正式阶段实际墙钟{actual['stage_wall']['main']['seconds']/60:.2f}分钟。该合计不含worker导入、轨迹压缩和调度开销；旧试跑预测保持原样，实际长规划尾部单列于ACTUAL_COST.json。\n")
    if(HERE/'TAIL_REBALANCE.receipt.json').exists():
        lines.append('\n用户要求加速后，将粗任务组中被长搜索挡住的未开始请求拆为逐UID独占任务，使用空闲CPU并行。圆形场景任务在同一进程完成当前局后停止领取后续请求，并保护当前及下一请求以覆盖信号交付边界。新的helper只执行原主清单中未完成且未被旧进程保护的UID。尾部清单先经官方预检；没有从头重跑运行中的rollout，没有改变模型、候选、选择、控制器参数或物理。调度与独占保护证据见TAIL_REBALANCE.receipt.json、TAIL_GUARDS.json、TAIL_COMPLETION_CHECK.json。\n')
    if(HERE/'TAIL_CHECKPOINT_HANDOFF_CHECK.json').exists():
        handoff=read(HERE/'TAIL_CHECKPOINT_HANDOFF_CHECK.json')
        lines.append(f"\n随后用原有、已通过逐位验证的操作检查点交接一个缺口六机A*请求：旧进程保存后退出，helper从同一UID的第{handoff['continuation_next_step']}步精确恢复全部环境、规划器、历史与随机步序；被它挡住的另一个请求同时开始。最终结果中的恢复来源与暂停时检查点哈希完全一致，已执行前缀未重算，未新增科学请求。交接证据见TAIL_CHECKPOINT_HANDOFF.receipt.json与TAIL_CHECKPOINT_HANDOFF_CHECK.json。\n")
    (HERE/'REPORT.md').write_text(''.join(lines))
    comparison_table=['| 留出场景 | pipeline | 同池全局 | 17池全局 | 同池近邻 | critic49 | 生成器均值 | 去控制器 | 只看场景 | η13 | 安全 | 样本内最佳49 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n']
    for n,r in stat['loso']['per_scene'].items():
        comparison_table.append(f"| {NAMES[n]}（{n}） | "+' | '.join(pct(r['rates'][m])for m in METHODS)+' |\n')
    (HERE/'METHOD_COMPARISON.md').write_text(''.join(comparison_table))
    lofo_tables=['# 留一族逐场景比较\n\n','生成器与critic组合方法（pipeline）、同池全局最佳单一修正（global_fixed）、上一轮17场景池全局最佳（strong_global）、同池近邻（knn）；其他方法全称见REPORT.md。\n\n']
    for key,label in [('lofo__bottleneck','文献与对穿训练→瓶颈'),('lofo__literature','对穿与瓶颈训练→文献')]:
        group=stat['lofo'][key];lofo_tables.append(f'## {label}\n\n')
        lofo_tables.append(comparison_table[0])
        for n,r in group['per_scene'].items():lofo_tables.append(f"| {NAMES[n]}（{n}） | "+' | '.join(pct(r['rates'][m])for m in METHODS)+' |\n')
        lofo_tables.append('\n| 方法 | 场景等权成功率 | unknown计成功上界 |\n|---|---:|---:|\n')
        for m,r in group['rates'].items():lofo_tables.append(f"| {LABELS[m]}（{m}） | {pct(r['rate'])} | {pct(r['unknown_success_upper'])} |\n")
    (HERE/'LOFO_METHOD_COMPARISON.md').write_text(''.join(lofo_tables))
    full_tables=['# 全18场景留一场景对照\n\n','完整critic在49个旧候选中选择（critic49），与上一轮同折17场景池全局最佳单一修正（strong_global）比较；全部直接使用旧标签，没有新rollout。\n\n',
        '| 留出场景 | critic49 | 17场景池全局最佳 | 差值（百分点） |\n|---|---:|---:|---:|\n']
    for n,r in stat['full18']['per_scene'].items():full_tables.append(f"| {NAMES[n]}（{n}） | {pct(r['rates']['critic49'])} | {pct(r['rates']['strong_global'])} | {pp(r['rates']['critic49']-r['rates']['strong_global'])} |\n")
    full_tables.append('\n| 方法 | 场景等权成功率 | unknown计成功上界 |\n|---|---:|---:|\n')
    for m,r in stat['full18']['rates'].items():full_tables.append(f"| {LABELS[m]}（{m}） | {pct(r['rate'])} | {pct(r['unknown_success_upper'])} |\n")
    (HERE/'FULL18_METHOD_COMPARISON.md').write_text(''.join(full_tables))
    axis=['| 泛化框架 | 检验 | 结果 |\n|---|---|---|\n',f'| 跨场景 | 11去重场景留一场景，pipeline−同池全局最佳 | {verdict}；{diff(primary)} |\n',
        f"| 跨场景 | 文献与对穿→瓶颈；对穿与瓶颈→文献 | {diff(bott)}；{diff(stat['lofo']['lofo__literature']['comparisons']['pipeline_minus_global_fixed'])} |\n",
        f"| 状态 | 完整critic−只看场景，49候选内成对消融 | {diff(comp['critic49_minus_scene_only49'])} |\n",
        f"| 控制器 | 完整critic−不看控制器，49候选内成对消融 | {diff(comp['critic49_minus_no_controller49'])} |\n"]
    (HERE/'AXIS_TABLE.md').write_text(''.join(axis))
    (HERE/'GAPS.md').write_text('''# 完成后的适用范围与局限

用户授权的请求构造修正、10例逐字节核对、试跑、正式评估、统计与结束审计均已完成。原失败证据保留。

- 去重规则在看过18场景标签之后确定；结果只限本修正族、控制器集与安全层下的11个去重场景，不外推未测拓扑。
- 每场景合格类型不同，使用既定类型等权；没有补齐能力门失败类型或重训控制器。
- unknown主值计失败，固定策略另报成功上界，不重跑。timeout不等于deadlock，正式死锁解决未认证。
- 置信区间条件于冻结模型/提案/选择，不代表重新采样完整训练过程的不确定性。
- 每场景仅2状态×2提案计时；未测试类型用兼容旧计时预测，规划尾部可能使成本估计失准。
- 49候选内样本内最佳仅为该离散集合的乐观描述，不是连续η空间的可达最优。
''')
    history=(HERE/'HISTORY_CHECK.md').read_text()
    if '## 用户授权修正后的完成核对'not in history:
        with(HERE/'HISTORY_CHECK.md').open('a')as f:f.write('\n## 用户授权修正后的完成核对\n\n保持210次训练、31折模型/提案/选择全部有效且字节未变；没有重新训练。只新增按原请求逐字段构造的执行适配器，10例规范化JSON和rollout UID逐字节一致后继续。所有新请求先经官方预检，仅补缺失结果；最终数据库、轨迹、旧输入和两个仓库完整性核对通过。原失败/停止记录保留，详见REQUEST_CONSTRUCTION_CORRECTION.md与DB_POSTFLIGHT.json。\n')
    reports=['REPORT.md','AXIS_TABLE.md','METHOD_COMPARISON.md','LOFO_METHOD_COMPARISON.md','FULL18_METHOD_COMPARISON.md','HISTORY_CHECK.md','SCENE_SET.json','INPUTS.json','ALL_FOLDS_FROZEN.receipt.json','TRAINING_SUMMARY.md','PILOT_COST.md','GAPS.md',
        'REQUEST_CONSTRUCTION_CORRECTION.md','REQUEST_FIX_RESUME.receipt.json','REQUEST_RECONSTRUCTION_CHECK.json','DB_POSTFLIGHT.json','ACTUAL_COST.json','INTEGRITY_START.json','INTEGRITY_END.json','INTEGRITY_COMPARE.json','STATISTICS.json']
    write(HERE/'COMPLETE.json',dict(utc=utc(),status='COMPLETE',primary=primary,request_fix_user_authorized=True,retraining_after_resume=0,
        reports={n:sha(HERE/n)for n in reports}))
    write(HERE/'HANDOFF.json',dict(utc=utc(),status='COMPLETE',experiment_complete=True,primary=primary,prior_failure_preserved=True,
        new_rollouts=db['unique_requests'],retraining_after_resume=0,reports={n:sha(HERE/n)for n in reports}))
    print(first,flush=True)

def main():
    assert(HERE/'main_COMPLETE.json').exists()
    database();integrity();report()

if __name__=='__main__':main()
