"""Capability-gated delivery: never invent correction outcomes or curve points."""
from common import *
import csv,shutil
from collections import Counter

NAMES={'A':'单门隔墙','B':'串联瓶颈走廊','C':'多门隔墙','D':'十字路口','E':'环岛','F':'房间与门','G':'随机障碍场'}
KINDS={'deterministic':'确定性行为克隆策略','mixture':'混合密度行为克隆策略'}

def table(path,fields,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def integrity():
    end=read(HERE/'INTEGRITY_DELIVERY_END.json');out={}
    for name in ['BEGIN','RESUME_BEGIN','RESUME2_BEGIN']:
        start=read(HERE/f'INTEGRITY_{name}.json')
        changes={k:[start['sources'].get(k),end['sources'].get(k)] for k in start['sources'].keys()|end['sources'].keys() if start['sources'].get(k)!=end['sources'].get(k)}
        out[name]=dict(Basin_equal=start['Basin']==end['Basin'],MAC_equal=start['MAC']==end['MAC'],source_changes=changes,
            status_added=sorted(set(end['Basin']['status'].splitlines())-set(start['Basin']['status'].splitlines())),
            status_removed=sorted(set(start['Basin']['status'].splitlines())-set(end['Basin']['status'].splitlines())))
    # Research source closure covered conservatively at every start, including indirect MACFlow imports.
    prefixes=[ROOT/'bottleneck_family',ROOT/'shared_control',ROOT/'shared_rollout_db',ROOT/'single_integrator',ROOT/'new_benchmark_common',
              D/'axis4a_geometry_horizon_v1',D/'axis4b_agent_count_v1',D/'double_bottleneck_eta3_basin',D/'blackbox_addendum_v1',FIELD,
              ROOT.parent/'01_MACFlow_Baseline_Reproduction/MACFlow_Official']
    beginning=read(HERE/'INTEGRITY_BEGIN.json')['sources']
    closure={p:v for p,v in beginning.items() if any(Path(p).is_relative_to(prefix) for prefix in prefixes)}
    closure_changes={p:[v,end['sources'].get(p)] for p,v in closure.items() if end['sources'].get(p)!=v}
    assert not closure_changes
    write(HERE/'IMPORTED_AND_COPIED_SOURCE_COVERAGE.json',dict(coverage='Conservative superset of all imported/copied research source dependencies, captured at original start',source_sha256=closure,changed=closure_changes))
    write(HERE/'INTEGRITY_FINAL_COMPARISON.json',out)
    return all(out['BEGIN'][k] for k in ['Basin_equal','MAC_equal']) and not out['BEGIN']['source_changes'],out

def render_geometry(cat):
    folder=HERE/'geometry_previews_final';folder.mkdir(exist_ok=True);manifest=[]
    for family,states in cat.items():
        grouped={}
        for s in states:grouped.setdefault(s['geometry_seed'],[]).append(s)
        for seed,group in grouped.items():
            s=group[0];roles=sorted({v['role'] for v in group});parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 430">','<rect width="800" height="400" fill="white" stroke="black"/>']
            for x0,y0,x1,y1 in s['rectangles']:parts.append(f'<rect x="{(x0+8)*50}" y="{(4-y1)*50}" width="{(x1-x0)*50}" height="{(y1-y0)*50}" fill="#555"/>')
            for i,(p,g,path) in enumerate(zip(s['positions'],s['goals'],s['paths'])):
                color=['#3465a4','#cc0000','#4e9a06','#75507b'][i];points=' '.join(f'{(x+8)*50},{(4-y)*50}' for x,y in path)
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" opacity=".5"/>')
                parts.append(f'<circle cx="{(p[0]+8)*50}" cy="{(4-p[1])*50}" r="8" fill="{color}"/>')
                parts.append(f'<circle cx="{(g[0]+8)*50}" cy="{(4-g[1])*50}" r="8" fill="none" stroke="{color}"/>')
            parts.append(f'<text x="10" y="420" font-size="12">{NAMES[family]} / {seed} / 初始状态与单机器人路径证据</text></svg>')
            path=folder/f'{family}_{seed}.svg';path.write_text('\n'.join(parts))
            manifest.append(dict(family=NAMES[family],geometry_seed=seed,roles=roles,preview=str(path),config=s['config'],rectangles=s['rectangles'],state_aliases=[v['alias'] for v in group]))
    write(HERE/'GEOMETRY_MANIFEST.json',manifest)

def capability_plot(nav):
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 660">','<rect width="1200" height="660" fill="white"/>',
           '<text x="55" y="35" font-size="22">合法留出几何上的单机器人导航能力门</text>',
           '<text x="55" y="61" font-size="14">每个实例16个回合；通过要求至少15次成功；所有策略均未通过</text>']
    for percent in [0,25,50,75,90,100]:
        y=470-percent*3.7;parts.append(f'<line x1="65" y1="{y}" x2="1160" y2="{y}" stroke="'+('#c33' if percent==90 else '#ddd')+'"/>')
        parts.append(f'<text x="22" y="{y+4}" font-size="12">{percent}%</text>')
    for fi,(family,v) in enumerate(nav.items()):
        for j,r in enumerate(v['instances']):
            x=85+fi*154+j*27;rate=100*r['success']/16;y=470-rate*3.7;color=['#3465a4','#729fcf','#75507b','#ad7fa8'][j]
            parts.append(f'<rect x="{x}" y="{y}" width="19" height="{470-y}" fill="{color}"/>')
            parts.append(f'<text x="{x+9}" y="{y-6}" text-anchor="middle" font-size="11">{r["success"]}/16</text>')
        parts.append(f'<text x="{125+fi*154}" y="500" text-anchor="middle" font-size="14">{NAMES[family]}</text>')
    for j,label in enumerate(['确定性行为克隆策略，种子17','确定性行为克隆策略，种子23','混合密度行为克隆策略，种子17','混合密度行为克隆策略，种子23']):
        y=540+j*24;parts.append(f'<rect x="70" y="{y-12}" width="18" height="14" fill="{["#3465a4","#729fcf","#75507b","#ad7fa8"][j]}"/><text x="100" y="{y}" font-size="13">{label}</text>')
    parts.append('</svg>');(HERE/'NAVIGATION_CAPABILITY.svg').write_text('\n'.join(parts))

def main():
    source_pass,comparison=integrity();nav=read(HERE/'ALL_VALID_NAVIGATION_GATES.json')['families'];caps=read(HERE/'ALL_CAPABILITY_FINAL.json')['families'];cat=read(HERE/'CATALOG_FINAL.json')['catalog'];verification=read(HERE/'EVIDENCE_VERIFICATION.json')
    assert verification['passed'] and all(not r['family_passed'] for r in nav.values()) and all(not r['family_passed'] for r in caps.values())
    archive=HERE/'previous_hard_stop_delivery';archive.mkdir(exist_ok=True)
    for name in ['REPORT.md','HANDOFF.md','results.json','GAPS.json','AXIS_TABLE.md','AXIS_TABLE.csv','FINAL_STATUS.receipt.json']:
        if (HERE/name).exists() and not (archive/name).exists():shutil.copyfile(HERE/name,archive/name)
    records=[];family_results={};total_success=0
    for family,v in nav.items():
        instances=[]
        for r in v['instances']:
            kind,seed=r['name'].split('_')[1:];row=dict(场景族=NAMES[family],策略=KINDS[kind],种子=int(seed),成功=r['success'],碰撞=r['collision'],超时=r['timeout'],未知=r['unknown'],回合数=16,成功率百分比=100*r['success']/16,通过=r['passed']);records.append(row)
            instances.append(dict(policy_name=KINDS[kind],seed=int(seed),successes=r['success'],collisions=r['collision'],timeouts=r['timeout'],unknown=r['unknown'],episodes=16,success_pct=100*r['success']/16,passed=r['passed'],checkpoint=r['checkpoint']))
            total_success+=r['success']
        family_results[family]=dict(name=NAMES[family],navigation_gate=dict(passed_instances=sum(r['passed'] for r in instances),total_instances=4,family_passed=False,instances=instances),multiagent_gate=dict(status='NOT_RUN',reason='Fewer than two valid navigation-qualified instances'),correction_preference=None,deadlock_repair_success_uplift_pp=None)
    table(HERE/'NAVIGATION_CAPABILITY.csv',list(records[0]),records);capability_plot(nav);render_geometry(cat)
    headings=['场景族','确定性行为克隆策略，种子17','确定性行为克隆策略，种子23','混合密度行为克隆策略，种子17','混合密度行为克隆策略，种子23','合格实例','多机器人能力门']
    md=['| '+' | '.join(headings)+' |','| '+' | '.join(['---']*len(headings))+' |']
    for family,v in nav.items():md.append('| '+' | '.join([NAMES[family]]+[f"{r['success']}/16" for r in v['instances']]+['0/4','未运行：合格实例不足两个'])+' |')
    (HERE/'CAPABILITY_GATE.md').write_text('\n'.join(md)+'\n\n门槛为成功率至少百分之九十，即十六个回合中至少十五个成功。这里只报告首次合法留出能力门；原几何重叠筛查不参与判定。\n')
    stages=[];dbtotals=Counter()
    for path in sorted(HERE.glob('*_POSTFLIGHT.json')):
        stage=path.name[:-len('_POSTFLIGHT.json')];casepath=HERE/f'{stage}_CASES.json'
        if not casepath.exists():continue
        pf=read(path);assert not pf.get('absent',[]);rows=read(casepath)
        unknown=len(pf.get('unknown_preserved',[]));stages.append(dict(stage=stage,cases=len(rows),unknown_preserved=unknown,postflight=str(path)))
        dbtotals['planned_executed_cases']+=len(rows);dbtotals['unknown_preserved']+=unknown
    write(HERE/'DB_STAGE_INDEX.json',dict(stages=stages,totals=dict(dbtotals),abandoned_unexecuted_plans=['reproduction','joint_0'],all_mergers_official_single_process=True))
    gaps=[dict(item='各族导航能力',reason='七族均无实例达到首次合法导航门的15/16要求；每族均低于至少两个合格实例的要求。',conditions_to_fill='在新预注册研究中获得各族至少两个合格导航实例；本轮不补训、不调参、不替换。'),
          dict(item='多机器人能力门、无标签行为探测与修正菜单',reason='所有族均在前置导航门退出；无合格实验对象。',conditions_to_fill='先满足导航能力门，再按固定三臂先导能力门、探测、选择冻结、评估的顺序执行。'),
          dict(item='选择模型与规模曲线',reason='没有合格目标族或新增训练族，无法估计零样本遗憾、回归斜率及配对置信区间。',conditions_to_fill='满足各族能力门且有兼容修正标签后，在任何对应目标标签打开前冻结选择与预处理。'),
          dict(item='随机障碍场锁定测试',reason='该族四个独立训练策略的导航成功次数为6、3、3、1，均未通过；最终选择模型未训练，修正菜单锁定评估零次。',conditions_to_fill='遵守新研究预注册、能力门与前六族分析冻结，最终模型冻结后只评估一次；本轮不复活被排除实例。'),
          dict(item='历史场景菜单兼容性',reason='旧标签有流内部修正、场景中心切向持续右手规则、不同切换或身份不完整等情况；按逐条审计保留兼容子集，不新跑旧场景。',conditions_to_fill='仅复用身份明确兼容的旧标签，无法恢复的项目保留缺口。'),
          dict(item='最初几何重叠及筛查',reason='原种子跨度有误，旧评估几何与训练几何部分重叠；旧筛查不能作为合法能力门。已修正全部评估几何，种子与矩形内容交集均为空，28个策略及其归一化未变。',conditions_to_fill='已完成修正与首次合法能力门；旧标签仅保留工程诊断，不删除、不重跑。'),
          dict(item='严格全仓库起终一致',reason='原始开工至交付期间其他目录有新增或修改；本轮实际依赖的受保护源码逐文件相同。',conditions_to_fill='保留全量差异与来源范围，不修改或恢复其他任务文件。')]
    write(HERE/'GAPS.json',gaps)
    curves=[dict(training_family_count=m,mean_regret_pp=None,CI_lower_pp=None,CI_upper_pp=None,status='GAP',reason='No navigation-qualified families') for m in [0,1,3,5]]
    table(HERE/'SCALING_CURVE.csv',list(curves[0]),curves)
    few=[dict(labels_per_arm=k,mean_regret_pp=None,status='GAP',reason='Pilot correction labels not queried: no qualified families') for k in [1,2,4,8,16]]
    table(HERE/'FEWSHOT_CURVE.csv',list(few[0]),few)
    for filename,title in [('SCALING_CURVE.svg','训练族数量与遗憾'),('FEWSHOT_CURVE.svg','少量标签数量与遗憾')]:
        (HERE/filename).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 180"><rect width="800" height="180" fill="white"/><text x="35" y="45" font-size="23">{title}</text><text x="35" y="100" font-size="19">缺口：没有能力门合格族，未产生可估计的数据点</text><text x="35" y="140" font-size="14">未估计不是零遗憾；不得由此推断规模趋势或少量标签效果。</text></svg>')
    methods=['只执行控制器动作','只加安全投影','全程修正参数十','全程修正参数十三','全程修正参数十五','修正参数十三至十按比例切换','修正参数十五至十按比例切换','修正参数十三至十按进度切换','通用持续右手规则','场景无关零样本选择模型','源场景平均最佳臂','少量标签经验选择','少量标签安全选择','四个先导状态简化选择','最佳固定参数交叉拟合参照','完整菜单交叉拟合参照','样本内完整菜单参照','原冻结控制器级选择模型','原冻结不看控制器的选择模型','探测估计规则选择','顺序通行专家']
    fields=['目标族']+methods+['碰撞率','超时率','平均遗憾','最大遗憾','灾难数','伤害数','证据等级','原因'];axis=[]
    for name in list(NAMES.values())[:6]+['前六族合并','随机障碍场（锁定测试）']:
        row={k:'N/A：导航门缺口' for k in fields};row.update(目标族=name,证据等级='未评估，无确认性结论',原因='每族合格导航实例不足两个；无多机器人修正标签')
        row['顺序通行专家']='N/A：较长预算联合可解参照见五层表，不作方法成功率比较';axis.append(row)
    table(HERE/'AXIS_TABLE.csv',fields,axis);(HERE/'AXIS_TABLE.md').write_text('| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+''.join('| '+' | '.join(r[k] for k in fields)+' |\n' for r in axis))
    layers=[]
    for family,v in verification['summary'].items():layers.append(dict(场景族=NAMES[family],几何有效状态=v['valid_states'],单机器人路径数=v['single_agent_paths'],顺序通行参考成功=v['joint_successes'],参考预算=10000,导航门合格实例='0/4',四机器人控制器成功='未评估',死锁修复成功率提升='未评估'))
    table(HERE/'FIVE_LAYER_EVIDENCE.csv',list(layers[0]),layers)
    write(HERE/'SELECTOR_MODEL_MANIFEST.json',dict(status='NOT_TRAINED_CAPABILITY_GAP',checkpoints=[],training_data_used=[],eligible_target_families=[],historical_compatible_labels_available=14996,G_correction_labels_used=0))
    write(HERE/'CHOICE_FREEZE_STATUS.json',dict(status='NOT_ENTERED_CAPABILITY_GAP',choices_made=[],evaluation_correction_labels_opened=False,locked_G_opened=False,no_fake_model_freeze_receipt=True))
    endpoints={str(i):dict(status='GAP_NOT_EVALUATED',reason='No navigation-qualified families',regret_pp=None) for i in [1,2,3,4]}
    results=dict(status='COMPLETED_WITH_CAPABILITY_GAPS',integrity_strict_original_start_to_end=source_pass,imported_and_copied_sources_unchanged=True,
        families=family_results,endpoints=endpoints,navigation=dict(passed_instances=0,instances=28,successes=total_success,episodes=448,unknown=0),
        scaling_curve=curves,fewshot_curve=few,locked_test=dict(status='NOT_OPENED_CAPABILITY_GAP',correction_evaluations=0,regret_pp=None,success_uplift_pp=None),
        source_label_audit=read(HERE/'HISTORICAL_ACTION_COMPATIBILITY.json'),gaps=gaps)
    write(HERE/'results.json',results)
    audit='严格开工至结束的完整性审计未完全通过：其他目录发生变化；本轮导入和复制的受保护源码逐文件一致。'
    each=''.join(f'{NAMES[f]}（族 {f}）：导航门0/4，修正作用与偏好未评估。' for f in 'ABCDEFG')
    paragraph=audit+'多场景终点一至四均为缺口，未评估：28个冻结策略在合法留出几何上均未达到导航门槛，最高13/16（81.25%），要求至少15/16。'+each+'训练族数为0、1、3、5时均无遗憾结果；随机障碍场修正菜单锁定评估0次，标签未打开。'
    detail='''

这轮完成了生成器、两类环境与单元测试、七族几何有效性与联合参考证据、导航示范、二十八个策略训练和首次合法导航能力门。每族均缺少至少两个合格实例，按预注册规则退出后续实验；没有补训、调参、替换或用其他策略冒充合格实例。能力门表见 CAPABILITY_GATE.md，逐实例成功、碰撞、超时与未知数见 NAVIGATION_CAPABILITY.csv，图见 NAVIGATION_CAPABILITY.svg。单机器人导航结果不能作为多机器人死锁修复证据；本轮没有成功率相对只加安全投影的提升估计，也没有臂偏好结论。

合法几何清单为 CATALOG_FINAL.json：每族十六个训练几何、八个评估几何，种子及矩形内容与训练集的交集均为空。每族八十个状态具备合法起终点、三百二十条单机器人路径和八十条全队无碰撞顺序通行证据。保存轨迹的运动、墙体与机器人之间扫掠碰撞已离线重算核对，见 EVIDENCE_VERIFICATION.json。顺序通行专家以原默认一万步作几何联合可解参照，控制器状态预算仍按最长路径规则记录；没有把较长参考预算的成功当作评估预算内的控制器成功。几何图与清单见 GEOMETRY_MANIFEST.json 和 geometry_previews_final/，静态图只展示初态、目标和几何路径证据。

随机障碍场的四个策略仅用自身十六个训练几何的导航示范训练；新的八个评估几何未用于任何训练。每族分别有确定性行为克隆策略和混合密度行为克隆策略，各两个种子；世界坐标观测为自身速度、相对目标、十六条测距射线、三个邻居的位置速度与掩码，共三十五维。训练设置、固定开发集损失、检查点与归一化均已冻结；合法能力门之后所有权重哈希仍与原冻结收据相同。导航能力门每实例十六回合，成功率至少百分之九十，即至少十五次成功；全部二十八个实例均未通过，最高为十字路口的确定性行为克隆策略、种子十七，十三次成功。

首次导航筛查存在训练几何与旧评估几何重叠的实现错误，不能作为正式能力门。原筛查全部保留为工程诊断；修正种子跨度后，以不变的二十八个冻结策略，在完全独立的几何上作首次合法能力门。不是通过更换检查点、补训或调整门槛复活旧筛查实例。修订、时间与哈希见 PREREGISTRATION_HOLDOUT_CORRECTION.receipt.json；交集审计见 ORIGINAL_GEOMETRY_OVERLAP_AUDIT.json、CATALOG_FINAL.json。原失败样例记录 GEOMETRY_EQUIVALENCE_FAILURE.json 也完整保留，未以它冒充轨迹不一致。

用户授权的单机器人环境仅用于示范与导航能力门，step 逐字复制原函数、只删除机器人对相关代码；stalled_groups 为空，不作为结果。原文件哈希与差异见 SINGLE_AGENT_COPY_PROVENANCE.json、SINGLE_AGENT_STEP.diff。修正非法测试目标并增加合法性断言后，单门隔墙、十字路口、环岛的成功与撞墙样例全通过；每一步均保持原双机器人间距大于两米，位置与逐机器人墙体扫掠间距逐位一致，单个机器人的终止判定也一致。四机器人 GeneratedEnv 仅重写初始化，step、reset、snapshot 仍继承原函数；单门逐位等价性通过。历史执行链十六条逐位复现已保留，没有重做；参照范围为机器人数量泛化轮的双机器人确定性潜势控制器、只加安全投影，不能扩大解释为所有历史流匹配策略均在本轮复现。

示范只来自真正单机器人回合，顺序通行专家不作为训练示范。四千四百八十条示范候选中，三千八百四十七条成功、六百三十三条碰撞；全部原始记录保留，训练仅使用成功示范，没有替换失败示范。每个族的十六个训练几何都提供了有效训练数据。导航门的碰撞、超时和未知分别保存，timeout 不称 deadlock。顺序专家未找到路线的记录作为 unknown 保留，取下一个合法种子产生不同场景或状态，没有重跑原未知身份。

共享数据库实验身份为 exp_multiscene_generalization_v1。全部执行阶段均为官方 design 追加登记、preflight、只跑真正缺失、默认 journal、单进程官方 merger、postflight；登记表旧行哈希前后一致。所有执行阶段的记录与未知数见 DB_STAGE_INDEX.json。初版复现计划的扩展 seed_key 在执行前改为官方 future_index=3900，确定性控制器忽略未来随机键；旧计划无执行。初版 joint_0 也仅登记预检，未执行，已被明确保留为弃用计划。未直接插入 rollout，未修改旧行，未知记录保留不重跑。

历史标签逐条兼容性审计保留了环形交换场景的一万四千九百九十六条记录，其中九十八条未知。旧双瓶颈场景和四向交叉口场景的流内部修正与本轮投影后修正菜单不兼容，旧的绕场景中心右手切向方向也不同于本轮目标方向顺时针旋转，不能混为同一个动作身份。其他身份不明确或切换不同的记录保留排除原因，没有因 critic 输入编码改变而重新生成旧 rollout。没有新跑任何已有场景。因为本轮没有合格目标族，未用任何源标签训练新选择模型；随机障碍场的修正标签进入选择模型的数量为零。

多机器人能力门、行为探测、少量标签选择、零样本选择模型、前六族修正评估及随机障碍场锁定评估均未进入。SCALING_CURVE.csv 与 FEWSHOT_CURVE.csv 保留全部预定横坐标，并明确写缺口；不存在可估计的遗憾、斜率或置信区间，不能画成零曲线。SELECTOR_MODEL_MANIFEST.json 明确没有新选择模型检查点，CHOICE_FREEZE_STATUS.json 明确没有伪造选择或最终模型冻结收据。AXIS_TABLE.md/csv 的缺口不是零成功率，也不是未知失败标签。多场景终点一至四没有被检验，不能据此宣称跨场景选择有效或无效，也不能宣称时间表或持续右手规则获胜。

资源记录见各提交前队列、各 SUBMISSION.json 与 RESOURCE_ACCOUNTING.json。保存的提交额度均不超过十八个总处理器请求与两个图形处理器分片；Slurm accounting 未启用，保留原始队列与提交参数，不伪造实际峰值计量。本轮作业均结束。没有 Git 写操作、包安装、修改任何已有源码或访问禁止的并行研究目录。严格全仓库起终一致性受其他目录新增或改动影响，精确差异见 INTEGRITY_FINAL_COMPARISON.json；实际导入和复制的研究依赖覆盖清单逐文件一致，见 IMPORTED_AND_COPIED_SOURCE_COVERAGE.json。

缺口与补齐条件见 GAPS.json。当前证据把瓶颈定位在进入多机器人研究之前的导航能力门；在新的预注册研究中获得各族至少两个合格导航实例后，才有条件检验修正选择与规模趋势。本轮冻结实例不再补训、调参或替换。
'''
    (HERE/'REPORT.md').write_text(paragraph+detail)
    (HERE/'HANDOFF.md').write_text('任务为执行 docs/prompt_round39_multiscene_generator_cross_scene_20261017.md 及其第二节全部用户补充决定。工作根目录 /home/zhihan/research/Basin_C1；Python 为 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python；新代码和输出只能在 diagnostics/multiscene_generalization_v1/，现有源码只读，共享数据库只走官方流程，禁止目录不得读写。\n\n'+paragraph+detail+'\n独立阅读顺序：先 REPORT.md、CAPABILITY_GATE.md、CATALOG_FINAL.json、EVIDENCE_VERIFICATION.json；再 PREREGISTRATION.md 与各修订收据、DB_STAGE_INDEX.json、INTEGRITY_FINAL_COMPARISON.json。权威能力门为 NAVIGATION_VALID_CAPABILITY_*.json；无 VALID 的旧导航文件只作几何重叠工程诊断。权威几何为 CATALOG_FINAL.json；旧清单保留追溯。不要重运行 complete_joint.py、navigation_all.py 或训练脚本；它们描述早期阶段。menu_and_probe.py 等文件仅是未执行的后续准备代码，不代表对应菜单、探测、特征或选择模型已验证。后续阶段因无合格对象跳过，选择模型、完整几何描述器与统计估计不应被误认为已经完成或可以直接发表。\n')
    write(HERE/'AF_ANALYSIS_FROZEN.json',dict(families={f:family_results[f] for f in 'ABCDEF'},endpoints={str(i):endpoints[str(i)] for i in [1,2,3]},scaling_curve=curves,correction_labels_opened=False))
    seal('AF_ANALYSIS_FROZEN.receipt.json',[HERE/'AF_ANALYSIS_FROZEN.json'],status='Capability gaps; no AF correction evaluation',before_any_G_correction_label=True)
    write(HERE/'LOCKED_TEST_STATUS.json',dict(family='随机障碍场',status='NOT_OPENED_CAPABILITY_GAP',correction_menu_evaluations=0,selector_model_trained=False,G_own_training_geometries=16,G_own_policy_instances=4,G_eval_geometry_used_for_training=False,G_correction_labels_used_for_selection_training=0))
    write(HERE/'HARD_STOP_SUPERSEDED.json',dict(utc=utc(),original_hard_stop_retained=True,user_authorized_resume=True,equivalence_tests_passed=True,current_status='Completed with capability gaps'))
    names=['REPORT.md','HANDOFF.md','results.json','GAPS.json','AXIS_TABLE.md','AXIS_TABLE.csv','CAPABILITY_GATE.md','NAVIGATION_CAPABILITY.csv','NAVIGATION_CAPABILITY.svg','CATALOG_FINAL.json','EVIDENCE_VERIFICATION.json','DB_STAGE_INDEX.json','SCALING_CURVE.csv','FEWSHOT_CURVE.csv','LOCKED_TEST_STATUS.json']
    seal('FINAL_DELIVERY.receipt.json',[HERE/n for n in names],completed_with_capability_gaps=True,no_correction_labels_generated=True)
    print(paragraph,flush=True)

if __name__=='__main__':main()
