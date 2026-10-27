"""Render conclusions only from sealed numerical, database and source evidence."""
from experiment import *
from registry import verify_receipt
import csv
import collections

METHODS={'safety':'只加安全投影','fixed_eta10':'全程固定 η10 修正','fixed_eta13':'全程固定 η13 修正','human_progress':'人定双 η 广播进度切换','learned_scene':'场景内重新学习修正'}
LAYERS={'central':'集中式联合投影','distributed':'分布式逐机器人投影','paper_dynamics':'论文动力学下分布式投影'}

def pct(value):return 'N/A' if value is None else f'{100*value:.2f}%'
def interval(values):return '['+', '.join(f'{100*x:.2f}' for x in values)+']'
def cell(row):
    if row is None:return 'N/A'
    return f"{pct(row['success_rate'])} {interval(row['ci95'])}"
def delta(row):
    if not row:return 'N/A'
    return f"{100*row['delta']:+.2f} pp {interval(row['ci95'])}；unknown 识别界 {interval(row['identified_bounds'])}"
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x).replace('|',' / ').replace('\n',' ') for x in row)+' |' for row in rows]])+'\n'
def save(name,text):
    path=HERE/name;assert path.resolve().is_relative_to(HERE);path.write_text(text.rstrip()+'\n')
def pick(summary,layer,method,type_code='type_equal'):
    return next((r for r in summary.get('statistics',[]) if r['layer']==layer and r['method']==method and r['type']==type_code),None)
def comparison(summary,method):return next((r for r in summary.get('layer_comparisons',[]) if r['type']=='type_equal' and r['method']==method),{})

def axis_tables(results,additional,types):
    tables=[];flat=[]
    def append(label,summary,kind):
        if not summary.get('statistics'):return
        rows=[]
        for typ in [*sorted({r['type'] for r in summary['statistics']} - {'type_equal'}),'type_equal']:
            values=[]
            for layer in ('central','distributed'):
                for method in METHODS:
                    row=pick(summary,layer,method,typ);values.append(cell(row))
                    flat.append(dict(scope=label,state_scope=kind,type=typ,type_name='类型等权合并' if typ=='type_equal' else types[typ],layer=layer,method=method,success_rate=None if row is None else row['success_rate'],ci95=None if row is None else row['ci95'],success_upper=None if row is None else row['success_upper'],upper_ci95=None if row is None else row['upper_ci95']))
            rows.append([('类型等权合并' if typ=='type_equal' else types[typ]),*values])
        tables.append(f'### {label}（{kind}）\n\n'+table(['控制器类型',*[LAYERS[l]+'：'+METHODS[m] for l in ('central','distributed') for m in METHODS]],rows))
    for scene in NAMES:
        append(NAMES[scene],results[scene]['fixed_full'],'全部主 state；重新学习另表')
        learned=results[scene]['learned_evaluation']
        refs=additional['evaluation_fixed_reference_tables'].get(scene,{})
        if learned.get('statistics'):
            # Add central fixed references on exactly the learned evaluation states.
            existing={(r['type'],r['layer'],r['method']) for r in learned['statistics']}
            matched=dict(learned,statistics=[*learned['statistics'],*[r for r in refs.get('statistics',[]) if (r['type'],r['layer'],r['method']) not in existing]])
            append(NAMES[scene],matched,'重新学习的共同独立评估 state')
    append('四场景合并',results['four_scene_type_equal'],'全部主 state；类型→实例→可用场景/state 等权')
    if 'matched_evaluation_scene_pool' in additional['evaluation_fixed_reference_tables']:
        pooled=additional['evaluation_fixed_reference_tables']['matched_evaluation_scene_pool']
        append('重新学习共同评估范围合并：'+'、'.join(NAMES[s] for s in pooled.get('scope',[])),pooled,'仅共同独立评估 state；集中式重新学习仅已有两场景另表')
    save('AXIS_TABLE.md','开发证据。单元格为成功率 % [95% CI]。unknown 按失败计下界；上界与其区间保存在 AXIS_TABLE.csv 和 MIGRATION_RESULTS.json。5000 次配对 bootstrap，类型→实例→state 成簇，两 seed 同簇。N/A 表示本表范围没有该做法，不把重新学习的评估子集混入全部主 state。\n\n'+'\n\n'.join(tables))
    with (HERE/'AXIS_TABLE.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in flat)

def failures(results,additional,census,controllers):
    rates=additional['weighted_outcome_and_diagnosis_rates'];rows=[]
    for scope,entries in rates.items():
        for key,entry in entries.items():
            layer,method=key.split('|');r=entry['rates']
            rows.append([NAMES.get(scope,'合格场景合并'),LAYERS[layer],METHODS[method],pct(r['success']),pct(r['upper']),pct(r['collision']),pct(r['timeout']),pct(r['unknown']),pct(r['deadlock_episode']),pct(r['wall_episode']),pct(r['canonical_joint_episode']),pct(r['actual_joint_episode']),pct(r['deadlock_agent_fraction'])])
    save('FAILURE_DECOMPOSITION.md','开发证据。以下为类型等权率，保留全部合格实例，包括 controller 级失败。4a′/4a″ 是有限窗口操作判据，timeout 本身不是死锁。episode 率表示至少一台机器人满足；agent fraction 先在每局内除以 N，再按类型→实例→state→seed 聚合。原始计数另见 MIGRATION_RESULTS.json 的 pooled_counts，不能与类型等权率混用。\n\n'+table(['范围','安全层','做法','成功下界','成功上界','碰撞','超时','unknown','逐机器人 4a′','墙前 4a″','原集中式联合锥参照','实际投影乘积联合锥','4a′ agent fraction'],rows)+'\n集中式采用联合起作用行在各机器人块上的分量；分布式采用各自实际二维约束。两层都计算逐机器人判据、原联合锥参照和实际投影乘积联合锥。W=2.5/5/10 秒、τ=0.02/0.05/0.1 的九格结果在 DIAGNOSTICS/<rollout_uid>.json；基线的类型等权敏感性在 FAILURE_SENSITIVITY_SUMMARY.json。\n\n去掉各层明确 controller 级失败后的率在 MAIN_ADDITIONAL_METRICS.json 的 per_layer_excluding_definite_controller_failure_rates；严格配对去除分析在 MIGRATION_RESULTS.json 的 excluding_controller_failures。两者支持的类型和实例范围必须随结果一起读取。圆形围场速度小，vmax×W 可能小于机器人半径，因此有限窗口静止条件较弱，不能由该标志单独证明永久死锁。')
    detail=[];discrimination=[]
    for scene in NAMES:
        for layer in ('central','distributed'):
            rr=[r for r in census if r['scene']==scene and r['layer']==layer and r['qualified']]
            discrimination.append([NAMES[scene],LAYERS[layer],len(rr),sum(r['controller_failure_definite'] for r in rr),sum(r['controller_failure_possible'] for r in rr),sum(r['state_related_failure_definite'] for r in rr),sum(r['state_related_failure_possible'] for r in rr),sum(r['at_least_one_4a_prime'] for r in rr),sum(r['at_least_one_4a_prime'] and not r['controller_failure_definite'] for r in rr)])
    for r in census:
        ctl=controllers[r['controller']]
        detail.append([NAMES[r['scene']],LAYERS[r['layer']],ctl['type']+'（'+r['controller']+'）',r['qualified'],r['episodes'],pct(r['success_lower'])+'–'+pct(r['success_upper']),r['controller_failure_definite'],r['controller_failure_possible'],r['state_related_failure_definite'],r['state_related_failure_possible'],r['at_least_one_4a_prime']])
    save('CONTROLLER_LEVEL_FAILURES.md','成功率≤5% 的合格实例计为 controller 级失败，保留在总体内，不用作死锁证据。unknown 形成成功率识别区间；上界≤5% 为明确失败，下界≤5%<上界为可能失败。(5%,95%) 内的成功率表示随 state 变化的失败，明确与可能分开。4a′ 标志先作为诊断候选；明确 controller 级失败的候选不作为死锁证据，可能失败也需保留不确定性。\n\n'+table(['场景','安全层','合格实例','明确实例失败','可能实例失败','明确状态相关失败','可能状态相关失败','有 4a′ 候选的实例','排除明确实例失败后有候选'],discrimination)+'\n'+table(['场景','安全层','控制器全称（定位代号）','合格','局数','成功率识别区间','明确实例失败','可能实例失败','明确状态相关失败','可能状态相关失败','有 4a′ 候选'],detail)+'\n10 pp 的伤害/灾难性下降分别相对只加安全投影与全程固定 η10，明确/可能界及逐实例记录见 MAIN_ADDITIONAL_METRICS.json.controller_drops。伤害局数（基线成功而修正失败）与包含 unknown 的上界见 MIGRATION_RESULTS.json.harms_vs_safety。')
    return discrimination

def geometry():
    rows=read(HERE/'GEOMETRY_CENSUS.json');groups=collections.defaultdict(list)
    for r in rows:groups[(r['scene'],r['layer'])].append(r)
    output=[]
    def bound(records,key,fn):
        values=[r[key] for r in records if r.get(key) is not None];return 'N/A' if not values else f'{fn(values):.6g}'
    for (scene,layer),rr in sorted(groups.items()):
        reasons=collections.Counter()
        for r in rr:reasons.update(r['escape_reasons'])
        output.append([NAMES[scene],LAYERS[layer],len(rr),sum(r['candidate_agent_steps'] for r in rr),bound(rr,'maximum_safe_speed',max),bound(rr,'minimum_conditional_bound',min),bound(rr,'maximum_conditional_bound',max),bound(rr,'minimum_normal_angle_degrees',min),bound(rr,'maximum_normal_angle_degrees',max),bound(rr,'minimum_local_cone_escape_infimum',min),bound(rr,'maximum_local_cone_escape_infimum',max),canonical(dict(reasons))])
    save('GEOMETRY_CENSUS.md','仅从保存的仿真轨迹重算。以下计数为描述性候选计数，不是类型等权发生率。约束法向、条件下界、名义速度与安全速度均取实际机器人二维块；非零安全速度、非零条件下界会使理想齐次锥解释不精确。逃逸距离为给定局部锥几何的描述量，不证明物理可解、控制器可达或长期死锁。\n\n'+table(['场景','安全层','候选轨迹','候选 agent-step','最大安全速率','最小条件下界','最大条件下界','最小法向角°','最大法向角°','最小逃逸下确界','最大逃逸下确界','逃逸计算分类'],output)+'\n完整逐轨迹条目见 GEOMETRY_CENSUS.json，原始局部 A、条件下界与名义输入见 DIAGNOSTICS。数值缺口保持显式记录，不生成新的控制轨迹。')

def dynamics(optional):
    paragraphs=['第二阶段为非主线开发证据；只跑合格实例、只加安全投影与三种固定修正。控制器参数不变，改变跟踪层、动力学与相应的论文安全层。与第一阶段配对差异解释为动力学执行栈的变化；与 v1 比较则安全层与动力学同时不同。']
    counts=[]
    for stage,entry in optional['results'].items():
        if not stage.startswith('optional_dynamics_'):continue
        scene=stage.removeprefix('optional_dynamics_');c=entry['counts']
        counts.append([NAMES[scene],entry['status'],c['completed'],c['planned'],c['unknown'],c['constraint_violation_steps'],c['sampled_speed_violation_steps'],c['acceleration_bound_violation_steps'],c['diagnosis_unavailable']])
        rows=[[METHODS[r['method']],cell(r),pct(r['success_upper'])] for r in entry.get('statistics',{}).get('statistics',[]) if r['type']=='type_equal']
        paragraphs.append('### '+NAMES[scene]+'\n\n'+(table(['做法','成功率 [95% CI]','unknown 成功上界'],rows) if rows else 'N/A：'+str(entry.get('reason'))))
        comparisons=entry.get('versus_first_stage',{})
        if comparisons.get('comparisons'):paragraphs.append(table(['做法','第二阶段−第一阶段','修正增益之差'],[[METHODS[r['method']],delta(r['second_minus_first']),delta(r.get('correction_gain_difference'))] for r in comparisons['comparisons']]))
        rates=entry.get('weighted_rates',{})
        if rates:paragraphs.append(table(['做法','逐机器人 4a′','墙前 4a″','耦合联合参照','实际乘积联合锥','直接静止','直接静止 agent fraction'],[[METHODS[k.split('|')[1]],*[pct(v['rates'][m]) for m in ('deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode','direct_stationary_episode','direct_stationary_agent_fraction')]] for k,v in rates.items()]))
    paragraphs.insert(1,table(['场景','状态','完成','计划','unknown','约束违反步','采样速度违反步','加速度上限违反步','诊断缺失局'],counts))
    paragraphs.append('双积分器在加速度层、独轮车在前视点速度层计算局部与联合判据。联合项是该动力学下另建的耦合参照，不代表实际运行了集中式动力学控制器，也不复用单积分器速度锥。直接静止判据为最后 5 秒位置不再变化且至少 90% 步速度≤1e−3×vmax。\n\n双积分器使用二阶控制障碍约束；无解时按预注册最小最大违反再跟踪的规则，违反事件不是 unknown。并未获得离散采样下的全局安全证明；约束违反与速度违反必须与成功率一起读取。门口场景的一步非线性求解器有可行性检查，但无全局最优证书。前视点半径膨胀、真实机器人碰撞、分段墙与区间碰撞检查见 DYNAMICS_IMPLEMENTATION.md；公式、联合参照及测试收据见 DYNAMICS_DIAGNOSTICS.md。')
    save('DYNAMICS_STAGE.md','\n\n'.join(paragraphs))

def handoff():
    save('HANDOFF.md',f'''本轮是四文献场景分布式安全层迁移 v2，全部为开发证据。工作目录为 {HERE}。v1 在 {V1}，已经按用户要求完成三场景收尾并冻结；禁止写入、恢复圆形围场被丢弃的局或调用旧协调器。圆形围场 v1 流程版的缺口保持为缺口。

当前入口为 coordinator_complete.py；其 detached PID/日志分别见 COORDINATOR_PID.json、coordinator.log。重启前先读 PID 与 ps，确认没有现存协调器，再使用同一流程 Python 运行此入口；禁止并发启动第二个协调器。下游入口为 downstream_complete.py，学习 rollout 使用 fine_dispatch.py、worker_fine.py；第二阶段和补充 rollout 使用 optional_fine_dispatch.py、optional_worker_fine.py。deploy_current.py 是旧部署快照生成器，不要重新运行覆盖 current 文件。当前进度与每场景完整/部分成功率见 STATUS.md、STATUS.json、LIVE_SAFETY_SUCCESS.json、LIVE_CONTROLLER_SAFETY_SUCCESS.csv；LIVE 为已完成局池化，不能替代最终类型等权率。

流程 Python：/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B；图仅最后用 .venv-c1-plots/bin/python -B。禁止安装包。experiment.py 把 TMP/MPL/XDG/CUDA 缓存限定在本轮并禁用 bytecode。新代码与输出只在本轮；共享 journal 只能通过官方 cache_writer 写默认实验目录，数据库合并只调用 db_merge.py 包装的官方单进程 merger。实验 ID 为 exp_baseline_literature_scenes_distributed_v2。

分布式安全层由 distributed_safety.py 构造，每台机器人独立解二维速度投影，使用全部邻居当前位置，不交换名义/安全速度。配对行 2(p_i−p_j)·u_i≥−cγh：窄通道、拥挤随机和圆形主线 c=1/2，门口 c=1；墙全责，原速度球及同问题数值重试。runtime.py 是规范执行，runtime_reservation_v2.py 是等价后端：必要条件证书只在原时空搜索必无解时省略穷举；其余运行同一原搜索。RESERVATION_V2_EQUIVALENCE.receipt.json 包含有限原搜索对照与完整逐位轨迹回放，case UID/种子未改变。内存/时限故障仅恢复未完成部分，不作为 unknown。

执行顺序：名义控制→原速度界与到达停止→逐机器人安全→local_basis 的目标/正交/相对速度三分量修正→同一逐机器人安全→原物理。人定双 η 主线使用平均进度广播；个人进度只在第二优先级另报。二维拥挤随机去掉墙，也去掉名义墙代价，其余参数不变。集中式非拥挤场景使用 central_reservation_reference_v2.py：保留原联合投影、锥诊断及unknown语义，仅加入原搜索必无解的必要条件证书。CENTRAL_FIXED_CERTIFICATE_REPLAY.receipt.json封存一个原耗时3592秒的2000步固定修正案例回放，全部物理数组及联合锥/起作用行/结果逐位一致，耗时9.85秒。只按精确身份复用。study_cases.py、registry.py、work_queue.py 管理身份、preflight 和不可变分组。任何运行前先读该部分 PREFLIGHT 收据；已存在 unknown 也不可默默重跑。

控制器为 8 类×6 实例=48 个，参数冻结；12 指学习动作菜单，不是训练 12 个 controller。合格实例窄通道/拥挤随机各 44、门口/圆形各 45。主 state 窄通道/门口各64，拥挤随机各 N=2/4/6/8/10/12/14 的16组共112，圆形前16。LEARNING_STATE_SPLIT_<scene>.json 密封各32训练 state；窄通道/门口评估32，拥挤随机评估主线 N8/14 共32，圆形评估16。训练标签每类前2实例×32×12×seed0；按能力以外的预注册 informative 规则决定是否学习。learn_critic.py、freeze_scene_choices.py 保持旧模型与嵌套留一类型程序，三初始化17/23/41，36组 source fits/场景。选择在首条 learned 评估提交前密封；基线结果按流程先前已知，这不是全局盲测。

first-stage 主数值在 MIGRATION_NUMBERS.receipt.json 封存后才进入 optional_delivery_fine.py。固定修正主 state 与 learned 独立评估子集必须分表；类型→实例→可用场景/state→两个配对 seed 聚合，5000次bootstrap，unknown 下界0/上界1。统计程序 statistics.py；补充指标 derived_metrics.py；原轨迹后验诊断 analysis_jobs_cached.py/optional_analysis_worker_cached.py 用 atomic_write.py 的唯一临时文件保护共享缓存，不能改已密封规范 experiment.write。scene_learning_scheduler.py每10秒按场景推进标签、冻结模型与密封选动作后的评估；fine_dispatch.py只将已密封案例切成单局/双种子小批次；worker_fine.py执行体与worker_active.py相同。scheduler_watchdog_complete.py独立检查协调器存活及空闲原因。diagnosis_alias_cache.py仅复用trace SHA相同、case除seed身份外一致、完整raw除seed/计时元数据外一致的确定性副本；23组独立既有诊断核对通过，语义不匹配就正常重算。每个seed仍有独立UID/诊断入口并保留统计权重。DIAGNOSTICS 只读仿真轨迹，不改控制器或物理。

非主线第二阶段使用 dynamics_runtime_reservation_v2.py 与 dynamics_safety.py；另有圆形全责、个人进度、拥挤随机无安全层背景。各部分独立 preflight，新身份且使用对应第一阶段物理 state UID 作随机数配对。无解违反、速度事件、非线性求解限制必须披露。optional_delivery_fine.py 负责可恢复队列、可选缺口、最终官方合并、DB postflight、最后画图、结束源审计和 final_reports_complete.py。任何 optional 缺口不得改变已封存主数值。

用户最新资源覆盖 prompt 的旧限额：已纠正释放日期：2026-10-21起已释放6 CPU和3 GPU份额，总计18 CPU份额/2 GPU份额；RESOURCE_SCHEDULE_CORRECTED.json覆盖旧日程。GPU份额为单卡 shard；所有用户作业（含pending）计入，CPU预留1给本地协调/分析。resources_released.py 每次提交前检查；resource_guard_released.py 独立守护，只取消本轮 blv2_ 作业至限额，完整raw/journal保留，未完成基础设施重试。

历史来源、完整性与缓存：HISTORY_CHECK.md/收据、SOURCE_COPIES*.receipt、INTEGRITY_START.json 和 PROTECTED_INPUTS_START.jsonl。旧文件/受保护目录规则见该审计与 prompt；并行受限目录只 stat，唯独指定 round28 只读例外。不要导入 v1 experiment（会创建旧目录缓存）。所有已密封 receipt 所列文件不可修改，current operational 部署快照与代码演进另记，不变的是科学语义和预注册身份。最终完整性结论只认 INTEGRITY_END.json，数据库只认 DB_POSTFLIGHT.json，尚未产生这些文件时不能声称最终审计通过。v1 旧两个非Python配置缺开工全文件哈希，不能由本轮审计倒补。

最终交付 REPORT.md、AXIS_TABLE.md/.csv、MIGRATION_RESULTS.json/.csv、FAILURE_DECOMPOSITION.md、CONTROLLER_LEVEL_FAILURES.md、GEOMETRY_CENSUS.md、DYNAMICS_STAGE.md、GAPS.md；最终完成标志 V2_COMPLETE.json。若标志不存在，继续按 STATUS/日志处理，不把后台仍在跑写成完成。''')

def main():
    for proof in ('MIGRATION_NUMBERS.receipt.json','MAIN_ADDITIONAL_METRICS.receipt.json','OPTIONAL_NUMBERS.receipt.json','DB_POSTFLIGHT.receipt.json','INTEGRITY_END.receipt.json'):verify_receipt(proof)
    audit=read(HERE/'INTEGRITY_END.json');db=read(HERE/'DB_POSTFLIGHT.json');assert audit['passed'] and db['passed']
    results=read(HERE/'MIGRATION_RESULTS.json')['results'];additional=read(HERE/'MAIN_ADDITIONAL_METRICS.json');census=read(HERE/'CONTROLLER_LEVEL_FAILURES.json');optional=read(HERE/'OPTIONAL_RESULTS.json')
    controllers={r['controller']:r for r in read(HERE/'CONTROLLER_MANIFEST.json')};types={r['type_code']:r['type'] for r in controllers.values()}
    axis_tables(results,additional,types);discrimination=failures(results,additional,census,controllers);geometry();dynamics(optional);handoff()
    paras=[f"完整性结束审计通过：检查 {audit['checked_files']} 个受保护文件，差异 {len(audit['differences'])}；两个仓库的受保护 git 状态/diff、PoC 与复用输入按开工清单核对。v1 保持只读。数据库结束审计通过，唯一完成身份 {db['unique_completed_identities']}，独立轨迹文件 {db['unique_trace_files']}，本轮冲突隔离 {db['own_quarantined_rows']}、未合并 journal {db['unmerged_complete_journals']}。证据见 INTEGRITY_END.json、DB_POSTFLIGHT.json；全部为开发证据。v1 原有两个非Python配置缺开工全文件哈希的限制仍然保留，本轮审计不倒补 v1 的严格覆盖。"]
    scene_text=[];gains=[]
    for scene in NAMES:
        full=results[scene]['fixed_full'];C=pick(full,'central','safety');D=pick(full,'distributed','safety');cmp=comparison(full,'safety');rate=additional['weighted_outcome_and_diagnosis_rates'][scene]
        r=rate['distributed|safety']['rates'];cr=rate['central|safety']['rates'];ds=next(x for x in discrimination if x[0]==NAMES[scene] and x[1]==LAYERS['distributed'])
        scene_text.append(f"{NAMES[scene]}：分布式成功 {cell(D)}，集中式 {cell(C)}，配对差 {delta(cmp.get('distributed_minus_central'))}；逐机器人 4a′ 分布式 {pct(r['deadlock_episode'])}、集中式 {pct(cr['deadlock_episode'])}。分布式合格 {ds[2]} 个实例，明确 controller 级失败 {ds[3]} 个，明确状态相关失败 {ds[5]} 个，排除明确实例失败后有 4a′ 候选 {ds[8]} 个。")
        items=[]
        for method in list(METHODS)[1:4]:
            row=pick(full,'distributed',method);cmp=comparison(full,method)
            items.append(METHODS[method]+'增益 '+delta(row.get('delta_vs_safety') if row else None)+'，相对集中式的增益之差 '+delta(cmp.get('correction_gain_difference')))
        learned=pick(results[scene]['learned_evaluation'],'distributed','learned_scene')
        if learned:
            best=learned['delta_vs_best_aggregate_fixed'];items.append(f"场景内重新学习在共同独立评估 state 成功 {cell(learned)}，减最佳聚合固定修正 {100*best['delta']:+.2f} pp {interval(best['ci95'])}（每次 bootstrap 重选最佳固定）")
        else:items.append('场景内重新学习 N/A，原因见 MIGRATION_RESULTS.json 对应 learned_evaluation')
        gains.append(NAMES[scene]+'：'+'；'.join(items)+'。')
    paras.append(' '.join(scene_text)+' 这些是有限窗口操作性候选，timeout 不能直接当作死锁；明确 controller 级失败不作死锁证据，可能失败保留 unknown 不确定性。圆形围场静止窗口较弱，不能据此单独证明长期死锁。详见 FAILURE_DECOMPOSITION.md、CONTROLLER_LEVEL_FAILURES.md。')
    paras.append(' '.join(gains)+' 全部主 state 的固定修正与重新学习评估子集分开；共同范围的比较及四场景类型等权合并见 AXIS_TABLE.md，unknown 上界、配对伤害、10 pp 实例下降与训练菜单描述性 oracle 另见 JSON。')
    stage2=[s for s,e in optional['results'].items() if s.startswith('optional_dynamics_') and e['status']=='complete']
    paras.append(f"第二阶段已有 {len(stage2)}/4 个场景完成计划请求。逐场景成功率、动力学−单积分器的配对差、修正增益之差、约束违反、速度违反和直接静止判据见 DYNAMICS_STAGE.md；未完成项按缺口报告，不当作 unknown。联合锥仅为同动力学下的耦合参照，未运行额外集中式动力学控制器；二阶约束没有全局采样安全证明，门口非线性问题没有全局最优证书。")
    from learning_label_support import main as label_support
    support=label_support(seal_result=not (HERE/'LEARNING_LABEL_SUPPORT.receipt.json').exists())
    gaps=read(HERE/'OPTIONAL_GAPS.json') if (HERE/'OPTIONAL_GAPS.json').exists() else []
    notes=['四篇论文版完全沿用 v1 的 PAPER_REPRODUCTION.*，本轮不重跑。v1 流程版只有窄通道、二维拥挤随机、门口三场景收尾，圆形流程版为用户决定停止、转入 v2 的缺口；不能把 v1 三场景当四场景。v2 圆形仅前16主 state，旧已落盘部分精确复用，其余为 v2 独立请求，不恢复 v1 已丢弃的8918局。',
        '冻结旧流程零样本、冻结旧表示 A3、旧物理接口 A0、无兼容冻结选择接口 G_E、未要求的专家参照均 N/A。集中式场景内重新学习只复用窄通道与门口，其他两场景本轮仅要求固定修正。第二阶段不重新学习。',
        '独立评估没有穷举12动作，因此完整菜单独立评估 regret 为 N/A。32训练 state 的12动作 oracle、跨 state 固定选择与已保存的留出类型预测是描述性补充，不能充当独立 state 性能；带符号 regret 可以为负。',
        '基线结果在学习前按规定已经可见；选择密封文件中的 outcomes_opened=False 仅表示选动作阶段不打开所选评估输出，不声称基线全局盲测。模型与超参数固定，仅使用训练标签与 source-type 嵌套验证。',
        'v1 原严格审计覆盖限制、圆形有限静止窗口、数值 unknown 识别界、动力学约束违反与非线性最优性限制如前述；所有结论为开发证据。']
    for scene in NAMES:
        if results[scene]['learned_evaluation'].get('status')=='N/A':notes.append(NAMES[scene]+'学习缺口：'+canonical(results[scene]['learned_evaluation'].get('reason')))
    for row in support['scenes']:
        if not row['observed_positive_action_support']:
            notes.append(NAMES[row['scene']]+'：预注册12动作菜单的训练标签中成功 '+str(row['success'])+'、失败 '+str(row['failure'])+'、unknown '+str(row['unknown'])+'；该数据没有观测到正例，不能用来证明模型识别成功动作的能力。密封选择计数与逐动作标签见 LEARNING_LABEL_SUPPORT.json；不据此改菜单、超参数或重训练。')
    notes.extend('非主线运行缺口：'+canonical(r) for r in gaps)
    save('GAPS.md','\n\n'.join(notes));save('REPORT.md','\n\n'.join(paras)+'\n\n缺口摘要：\n\n'+'\n\n'.join(notes))
    outputs=['REPORT.md','AXIS_TABLE.md','AXIS_TABLE.csv','FAILURE_DECOMPOSITION.md','CONTROLLER_LEVEL_FAILURES.md','GEOMETRY_CENSUS.md','DYNAMICS_STAGE.md','GAPS.md','HANDOFF.md','LEARNING_LABEL_SUPPORT.json','LEARNING_LABEL_SUPPORT.receipt.json']
    seal('FINAL_REPORTS.receipt.json',[HERE/n for n in [*outputs,'final_reports_complete.py']],source_audit_passed=True,db_postflight_passed=True,main_numbers_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),optional_numbers_receipt_sha256=sha(HERE/'OPTIONAL_NUMBERS.receipt.json'))

if __name__=='__main__':
    if sys.argv[1:]==['handoff']:handoff()
    else:main()
