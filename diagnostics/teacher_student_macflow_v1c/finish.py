"""Finalize existing evidence, provenance, reports and protected-tree audit."""
from core import *
from audit_utils import source_inventory
import ast,difflib,csv,subprocess
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
PLOTS='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python'
ENDPOINTS=['TS1','TS2','TS3a','TS3b','TS6','TS6b']
def call(script,python=PYTHON):subprocess.run([python,'-B',str(HERE/script)],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','JAX_PLATFORMS':'cpu'},check=True)
def audit():
    checks={};checks['pipeline_and_episode_replay_reuse']=all(sha(HERE/f)==sha(D/'teacher_student_macflow_v1b'/f) for f in ['core.py','pipeline.py','episode_frozen.py','episode_extended.py','actions.py']);checks['component_tests']=read(HERE/'COMPONENT_TESTS.json')['status']=='PASS';checks['runtime_reuse_equivalence']=read(HERE/'REUSE_RUNTIME_EQUIVALENCE.json')['status']=='PASS'
    for name in ['PREREGISTRATION.receipt.json','REGATE_FROZEN.receipt.json','IMPLEMENTATION_FROZEN.receipt.json','CHOICES_FROZEN.receipt.json','RHR_FROZEN.receipt.json','MAIN_IMPLEMENTATION_FROZEN.receipt.json','ANALYSIS_READY.receipt.json']:
        checks[name]=all(sha(path)==h for path,h in read(HERE/name)['files'].items())
    for origin in ['teacher_student_macflow_v1','teacher_student_macflow_v1b']:
        checks[origin+'_old_choices_frozen']=all(sha(path)==h for path,h in read(D/origin/'CHOICES_FROZEN.receipt.json')['files'].items())
    checks['all_checkpoints_frozen']=all(sha(r['checkpoint'])==r['checkpoint_sha256'] for r in read(HERE/'STUDENTS.json').values())
    checks['gate_before_pilot']=read(HERE/'REGATE_FROZEN.receipt.json')['utc']<read(HERE/'pilot_SUBMISSION.json')['utc'];checks['choices_before_main']=read(HERE/'CHOICES_FROZEN.receipt.json')['utc']<read(HERE/'main_SUBMISSION.json')['utc'];checks['rhr_selection_before_main']=read(HERE/'RHR_FROZEN.receipt.json')['utc']<read(HERE/'main_SUBMISSION.json')['utc']
    grid=read(HERE/'RHR_GRID_RESULTS.json')['grid'];expected=min(grid,key=lambda r:(-r['success'],r['speed'],r['trigger_steps']));selected=read(HERE/'RHR_SELECTED.json');checks['grid_selection_exact']=(selected['speed'],selected['trigger_steps'])==(expected['speed'],expected['trigger_steps']);checks['all12_grid_points']=len(grid)==12 and all(len(g['controllers'])==len(read(HERE/'REGATE.json')['valid_names']) for g in grid)
    from shared_rollout_db.src import cache_writer
    from shared_rollout_db.src.rollout_db import connect
    checks['official_db_package']=Path(shared_rollout_db.__file__).resolve()==ROOT/'shared_rollout_db/__init__.py';checks['default_journal_path']=(cache_writer.ROOT/'journals').resolve()==ROOT/'shared_rollout_db/journals'
    with connect(True) as db:
        sources=[dict(r) for r in db.execute('SELECT * FROM source_file WHERE experiment_uid LIKE ?',('exp_teacher_student_macflow_v1c_%',))]
        checks['probes_no_fake_task_outcomes']=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',('exp_teacher_student_macflow_v1c_probe',)).fetchone()[0]==0
    checks['all_journals_official_path']=all(Path(r['path']).resolve().is_relative_to(ROOT/'shared_rollout_db/journals') for r in sources);write(HERE/'DB_PATH_AUDIT.json',dict(passed=checks['official_db_package'] and checks['default_journal_path'] and checks['all_journals_official_path'],package=str(shared_rollout_db.__file__),default_journals=str(cache_writer.ROOT/'journals'),sources=sources,guard='core.py before file writes, worker.py/run.py before work and returned journal path assertion'))
    imported={};changed=[];inventories=[]
    for f in HERE.glob('imports_*.json'):
        src=read(f)['imported_sources'];pkg=str(ROOT/'shared_rollout_db/__init__.py');writer=str(ROOT/'shared_rollout_db/src/cache_writer.py');assert pkg in src and writer in src;inventories.append(f.name)
        for path,h in src.items():
            if Path(path).is_relative_to(HERE):continue
            if path in imported and imported[path]!=h:changed.append(path)
            imported[path]=h
    changed.extend(p for p,h in imported.items() if sha(p)!=h);checks['existing_import_sources_unchanged']=not changed;write(HERE/'IMPORTED_SOURCE_AUDIT.json',dict(existing_source_hashes=imported,changed=changed,inventories=inventories))
    for phase in ['pilot','tune','main']:checks[phase+'_db_exact']=read(HERE/f'{phase}_MERGE_RECEIPT.json')['db_exact']
    r=read(HERE/'results.json');checks['primary_population_exact']=set(r['CONFIRMATORY']['controllers'])==set(read(HERE/'CAPABILITY_GATE.json')['eligible']+[read(HERE/'TEACHERS.json')[i]['name'] for i in [1,7]])
    checks['resource_cpu_cap']=max(x['active_cpu'] for x in read(HERE/'RESOURCE_MONITOR.json'))<=20
    write(HERE/'RESULTS_VALIDATION.json',dict(utc=utc(),passed=all(checks.values()),checks=checks));assert all(checks.values()),checks
    prov=read(HERE/'COPY_PROVENANCE.json');patch=[]
    for row in prov:
        src=Path(row['source']);dst=HERE/row['destination'];assert sha(src)==row['source_sha256'];row['destination_sha256']=sha(dst);patch.extend(difflib.unified_diff(src.read_text().splitlines(True),dst.read_text().splitlines(True),fromfile=str(src),tofile=str(dst)))
    write(HERE/'COPY_PROVENANCE.json',prov);(HERE/'COPY_DIFF.patch').write_text(''.join(patch));source_inventory('imports_finish.json')

def report():
    r=read(HERE/'results.json');integ=read(HERE/'INTEGRITY_COMPARISON.json');reg=r['regate'];selected=r['rhr_selected'];names=r['CONFIRMATORY']['controllers'];n=len(names);valid=reg['n_new_valid'];f=read(HERE/'FAILURE_SOURCES.json');audit_pass=integ['protected_content_pass'] and not integ['removed_status_entries'] and read(HERE/'DB_PATH_AUDIT.json')['passed'];says=lambda k:'，'.join(t+'='+r[k].get(t,{}).get('status','INCONCLUSIVE') for t in ENDPOINTS)
    text=f"完整性审计{'通过' if audit_pass else '未完全通过（见差异）'}，Basin_C1 shared_rollout_db与默认journals路径断言通过。8个学生中{valid}个通过修正门，主分析共{n}个学生。主分析：{says('CONFIRMATORY')}；仅新增敏感性：{says('SENSITIVITY')}；合并DEVELOPMENT：{says('DEVELOPMENT')}。调参后RHR选择速度{selected['speed']}m/s、触发{selected['trigger_steps']*.05:g}s；主分析S500−B-RHR-tuned={r['CONFIRMATORY']['TS6b']['mean']:.2%}，95%CI={r['CONFIRMATORY']['TS6b']['CI95']}。\n\n"
    text+='''# Round30c — regate and tuned right-hand baseline

本轮没有训练或更换任何checkpoint。8名学生的门在计算新指标之前预注册，目标、噪声、2048固定dev state/noise与round30b相同，唯一改动是仅在teacher动作norm≥.10m/s的样本算余弦。step非0与全样本误差≤.05m/s占比≥80%不变。已有6名学生维持原有效性，不用描述性新指标改变其判定。REGATE.md/JSON及dev_predictions/包含全部14人的.05/.10/.20敏感性、样本数、覆盖率、RMSE、原平均余弦与完整预测。近零动作的方向不能由全样本平均余弦稳健反映，本轮改变的是门定义，不是网络。

主分析由round30b已评估的2名加本轮修正门及能力门通过者组成；如实披露旧2人标签已打开，并在修正门后补入学生，所有人仍用同一组原预注册终点。敏感性只用本轮新增者；DEVELOPMENT追加旧4名local，其标签此前已打开。主分析少于8则全部终点IC；TS3b各分析集独立检查至少3名各偏好F10/F13≥10pp的前提。FROZEN_CHOICES.json只保存本轮新学生；旧6人的冻结选择按原收据复用，在analysis合并而不重选。

## RHR tuning and fairness

调参为16个新的state，固定seed区间2026101503000–2026101503999按升序取前16个，与DB uid/content/physical、round28 eval和round30 pilot去重。所有Ring DB state具备positions/velocities/goals，RING_PHYSICAL_SCHEMA_AUDIT.json说明非Ring异形state导致通用signature缺失的范围；不存在未解析Ring物理state的去重遗漏。12组speed(.10,.15,.25,.35)×trigger(20,40,80步)，低速阈值.01、顺时针、未到目标、D-local安全层保持原定义。不添加OrthoFlow3。每个有效学生每点16个state、seed0，按学生成功率等权平均最高选一个全局参数；平局速度更小、时长更短。unknown以下界参与选择，另报上界，不称为已知失败。参数冻结先于任何tuned主评估。全部网格结果及逐学生计数见RHR_GRID_RESULTS.json、调参raw和DB_ROWS，图见rhr_tuning_grid.png。该RHR用了学生调参标签，而η在联合教师上确定，未针对学生调整，因此比较偏向RHR。

'''
    grid=read(HERE/'RHR_GRID_RESULTS.json')['grid'];text+='| Speed m/s | Trigger s | Tuning success lower–upper | Unknown |\n|---|---:|---:|---:|\n'
    for g in grid:text+=f"| {g['speed']} | {g['trigger_steps']*.05:g} | {g['student_mean_lower']:.2%}–{g['student_mean_upper']:.2%} | {g['unknown']} |\n"
    text+='\n## Primary, sensitivity and development tables\n\n成功率下界/上界保留unknown；controller后state成簇配对bootstrap5000次，seed2026103032，两个seed成簇保留，每次重抽中重选最佳fixed/schedule/oracle。四臂oracle仅F10/F13/S300/S500。所有碰撞与超时率另列；timeout不等于deadlock。\n\n'
    for key in ['CONFIRMATORY','SENSITIVITY','DEVELOPMENT']:
        ss=r[key];text+='### '+key+'\n\n';path=HERE/f'paper_table_{key.lower()}.md';text+=path.read_text() if path.exists() else '无合格样本。\n';text+='\n'
        for t in ENDPOINTS:text+=f"- {t}: {json.dumps(ss.get(t,{'status':'INCONCLUSIVE'}),ensure_ascii=False)}\n"
        if ss.get('n_controllers'):text+=f"- always S500−A3: {json.dumps(ss['always_S500_minus_A3'])}\n- 安全层臂碰撞数（包含tuned）：{ss['safety_layer_collisions']}；unknown：{ss['unknown_total']}。\n"
        text+='\n'
    text+='''## Failure sources and rescue counts

自身停滞=nominal timeout且无碰撞；安全层导致停滞=nominal碰撞→safety timeout。两者为预注册描述性拆分，其他转移和unknown单列，不强行归类。TRANSFERS.json逐条保留nominal→safety，FAILURE_SOURCES.json逐学生完整结果构成；failure_sources.csv包含各组被F13、S500、原RHR和tuned救回的数量。

| Student | Nominal success/collision/timeout/unknown | Own stagnation | F13/S500/RHR/tuned rescued | Safety induced stagnation | F13/S500/RHR/tuned rescued |
|---|---|---:|---|---:|---|
'''
    for name,d in f['controllers'].items():
        o=d['nominal_outcomes'];a=d['failure_sources']['own_stagnation'];b=d['failure_sources']['safety_induced_stagnation'];counts='/'.join(str(o.get(k,0)) for k in ['success','collision','timeout','unknown']);resc=lambda g:'/'.join(str(g[k]) for k in ['rescued_F13','rescued_S500','rescued_BRHR','rescued_BRHR_tuned']);text+=f"| {name} | {counts} | {a['count']} | {resc(a)} | {b['count']} | {resc(b)} |\n"
    own=f['totals'].get('own_stagnation',0);safety=f['totals'].get('safety_induced_stagnation',0);text+=f"\n合并分析两类计数：自身停滞{own}、安全层导致停滞{safety}。"+('自身停滞占两类多数，问题陈述应以分散协调困难为主，安全层是另一部分原因。' if own>safety else '本分析不支持自身停滞占两类多数的表述。')+'不能仅从timeout推断真实deadlock。\n\n'
    primary=r['CONFIRMATORY'];text+='## Conclusion mapping\n\n'
    if primary['TS1']['status']=='PASS' and primary['TS6b']['status']=='PASS':text+='TS1、TS6b通过：在本协议专家不可部署、不通信的分散学生上，修正有必要且优于调参后的经典RHR。\n\n'
    elif primary['TS6']['status']=='PASS' and primary['TS6b']['status']!='PASS':text+='TS6通过但TS6b未通过：原优势部分依赖RHR参数，论文应以tuned版本为基线，并强调η没有在学生上调参；CI覆盖0不构成等效性证明。\n\n'
    if primary['TS2']['status']=='NOT_PASS':text+='TS2未通过：学生的阶段时间表优势未获支持；按任务映射，分散学生使用固定η13已足够，一律S500仍可作默认选择。该结论限于本固定菜单与样本，不把不显著当作普遍等效。\n\n'
    if primary['TS3a']['status']=='NOT_PASS':text+=f"TS3a未通过，A3平均oracle遗憾{primary['TS3a']['regret']['mean']:.2%}；需结合冻结选择逐学生错误判断系统性迁移失败，零样本优先比较稳健默认S500。\n\n"
    text+='''## Inheritance, cost, cache and integrity

TS4在results.json报告学生/教师最佳臂、固定偏好、四臂Spearman。TS5报告原教师D-local/central/shared四臂配对差；TS5_TEACHER_SUPPLEMENT.md/JSON另报教师nominal/safety原有标签与学生的配对差，全部复用。已有准备输入、单核JIT预热后的学生网络/教师10步/四个QP耗时从round30b的10个global及round30的4个local直接复用，出处和hash见REUSED_TIMING.json；它们不等同新tuned完整闭环延迟，未重新采样耗时。round30b等变/act_scale诊断只作原证据引用，不把修正门当作严格旋转等变证明。

copy/core/pipeline/episode/actions与round30b相同，16条逐位回放门直接引用；tuned的原R28参数点动作与85步计数序列逐位通过，12组边界/方向/幅度测试通过。Student.nominal主体AST与round30b相同，唯checkpoint路径定位改变；4名local的旧to_world公式与当前部署逐位一致。没有monkeypatch或修改已有源码、物理、安全层、事件、成功标准、checkpoint，没有git写命令或环境安装。DB_PATH_AUDIT.json检查每个入口/worker模块路径及所有source_file journal路径均在Basin_C1共享库；PoC未写入journal。

每阶段design→preflight封存→missing→default journal→官方单进程merger→postflight，原6名全部七臂标签精确复用。若tuned选择原R28参数，则同样复用已有B-RHR，绝不按改名重复rollout。registration/每次记录所有旧行hash不变与新增行/sha256，未UPDATE/DELETE。各阶段planned/reuse/missing、DB_ROWS、CACHE_POSTFLIGHT、MERGE_RECEIPT保留完整链。unknown不重跑，不算已知失败。

INTEGRITY_START/END对Basin git状态/diff哈希、官方仓库HEAD/status/diff哈希、PoC/round30/round30b全部文件路径/hash核对；INTEGRITY_COMPARISON单列并行任务新增git条目（非本轮所写）。IMPORTED_SOURCE_AUDIT记录每个已有import源，COPY_PROVENANCE/COPY_DIFF.patch记录复制/派生源及最终diff。RESULTS_VALIDATION核门、选择、时序、缓存、冻结文件和资源。RESOURCE_MONITOR保存所有提交时的squeue与运行占用，CPU总上限20，本轮GPU推理申请2shards。没有要求视频。
'''
    (HERE/'REPORT.md').write_text(text)
    handoff=f'''# Round30c handoff

任务已完成：8名候选中{valid}名通过修正门；CONFIRMATORY={n}名，SENSITIVITY={r['SENSITIVITY']['n_controllers']}名，DEVELOPMENT={r['DEVELOPMENT']['n_controllers']}名。{says('CONFIRMATORY')}。tuned全局参数speed={selected['speed']}m/s、trigger={selected['trigger_steps']}steps。报告第一段及三张paper_table_*包含完整结论和CI。

工作根目录/home/zhihan/research/Basin_C1，唯一新增目录diagnostics/teacher_student_macflow_v1c。主环境/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；绘图同目录.venv-c1-plots。输入任务docs/prompt_round30c_regate_and_tuned_rhr_20261015.md；PREREGISTRATION.md及收据在任何新指标前封存。没有训练或换checkpoint，STUDENTS.json指向round30b10个global与round30旧4个local，均核hash。

链：integrity.py start→prepare.py→Slurm regate.py→tests.py及planning.py initial/pilot→orchestrate.py：pilot官方合并/能力门→R5/freeze_choices→12点tune官方合并→freeze_rhr→main preflight和Slurm缺失任务→main官方合并→analysis.py→finish.py（教师补表、网格图、验证、完整性结束、报告）。阶段和作业ID见RUN_STATUS.json/*_SUBMISSION.json/orchestration.log。已完成，不要重复启动协调器/训练/rollout，unknown不得重跑。要重算统计，使用analysis_arrays.npz和results.json，输出新文件名保留收据。

新8人在修正门前没有闭环标签，主分析的旧2人及DEVELOPMENT旧4人的标签此前打开过，研究顺序如实披露。三套主表与6个终点、alwaysS500−A3、失败/救回表都在results.json及REPORT.md。RHR网格是全体有效学生等权平均选一个全局参数，不是按学生分别选；η没有在学生上调参。REGATE的.05/.20阈值只作敏感性，旧6人判定不变。数据与教师标签均复用，缓存manifest/pre/postflight和登记行审计全留存。所有源码和环境保护规则继续适用，只写本目录；DB官方流程是唯一外部写例外。资源全用户20CPU/2GPU shard，提交前查squeue。

审计：protected_content_pass={integ['protected_content_pass']}，DB路径断言通过；复现门引用及tuned原参数点逐位测试通过。COPY_DIFF.patch、IMPORTED_SOURCE_AUDIT、DB_PATH_AUDIT、INTEGRITY_COMPARISON与RESULTS_VALIDATION为交付证据。旧30/30b文件从未修改。timeout不是deadlock，unknown不是已知失败。没有视频交付。
'''
    (HERE/'HANDOFF.md').write_text(handoff);seal('DELIVERY.receipt.json',[HERE/'REPORT.md',HERE/'HANDOFF.md',HERE/'results.json',HERE/'RESULTS_VALIDATION.json',HERE/'INTEGRITY_COMPARISON.json',HERE/'TS5_TEACHER_SUPPLEMENT.json'],complete=True)

def main():
    call('teacher_supplement.py');call('plot_grid.py',python=PLOTS)
    timings={}
    for origin,indices in [('teacher_student_macflow_v1b',[0,1,2,3,4,5,6,7,9,11]),('teacher_student_macflow_v1',[8,10,12,13])]:
        p=D/origin/'INFERENCE_TIMING.json';records=read(p);names={read(HERE/'TEACHERS.json')[i]['name'] for i in indices};timings[origin]=dict(path=str(p),sha256=sha(p),records=[row for row in records['controllers'] if row['controller'] in names],scope=records['scope'])
    write(HERE/'REUSED_TIMING.json',timings);audit();subprocess.run([PYTHON,'-B',str(HERE/'integrity.py'),'end'],cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},check=True);report()
if __name__=='__main__':main()
