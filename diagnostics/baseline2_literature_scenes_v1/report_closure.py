"""Final v1 report under the explicit three-scene user closure decision."""
from experiment import *
from closure_scope import FLOW_NAMES,STOP_REASON,circle_gap
from finish_reports import equal,delta,rate,MN,FIXED
import csv
import subprocess


def signed(value):return f'{100*value:+.2f}个百分点'
def ci_comparison(r):return signed(r['delta'])+f" [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}]"


def main():
    jobs=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%j']).decode().splitlines()
    assert not any('baseline2-' in j for j in jobs),jobs
    required=['MIGRATION_RESULTS','PAPER_REPRODUCTION','SAFETY_DIAGNOSTICS','BACKGROUND_RESULTS',
              'PILOT_RESULTS','REFERENCE_ENDPOINTS','SWITCHING_DIAGNOSTICS','TRAINING_MENU_REFERENCE',
              'ARRIVAL_RELEASE_DIAGNOSTICS','DB_POSTFLIGHT','INTEGRITY_END','USER_DIRECTED_CLOSURE','FINAL_NUMERIC_RESULTS']
    for name in required:assert (HERE/(name+'.receipt.json')).exists(),name
    result=read(HERE/'MIGRATION_RESULTS.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');gap=circle_gap()
    assert set(result['per_scene'])==set(FLOW_NAMES)
    assert result['closure_scope']['four_scene_combination_computed'] is False
    safety=list(csv.DictReader((HERE/'SAFETY_BASELINE.csv').open()))
    release=next(r for r in read(HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.json')['summaries'] if r['population']=='qualified_blackbox')
    p1='完整性审计：'+audit['summary']+'。本轮新增代码与输出均在v1目录，外部写入仅为官方共享DB追加登记、journal与单进程merger，未做Git commit。用户决定收尾v1、转入v2：三个已完成流程版场景纳入结论，圆形流程版中止；原四场景任务并未全部完成。所有结果均为开发证据。'
    p2='论文版：窄通道场景按400dpi图8首帧建立几何，无限时域模型预测控制与死锁解除能通过，但修正后的缓冲Voronoi单元与右手绕行点规则也能通过，未复现所要求的活锁；真正t0未公布。二维拥挤随机场景原样使用官方GUI采样器，逐机器人数的成功率、数值unknown上下界及与论文表II/III的比较见PAPER_REPRODUCTION，官方参数和离散动力学差异使其不能称为严格方程复刻。门口场景和第二优先级文献交叉口的单个关闭实例静止未到达、开启实例通过，复现单实例定性行为，不能据此估计总体100%/0%概率；官方模拟器实际推进30秒，早期15秒元数据已勘误。已完成的圆形围场论文版记录保留在论文附表，不用来补齐中止的流程版结论。'
    parts=[]
    for scene,name in FLOW_NAMES.items():
        u=read(HERE/f'BASELINE_USABILITY_{scene}.json');rr=[r for r in safety if r['scene_code']==scene and r['qualified']=='True'];n=sum(r['category']=='filter_equilibrium_deadlock' for r in rr)
        parts.append(f"{name}{u['qualified']}/48个实例通过单机能力门，合格实例的4a操作候选{n}个机器人局")
    p3='流程版：'+'；'.join(parts)+'。静止未到达、阻挡、零名义输入、振荡与缓慢前进互斥拆分，timeout不直接当死锁。窄通道原始实例中，'+str(release['initially_arrived_agent_episodes'])+'个初始已到达的合格黑盒机器人局里，'+str(release['moved_after_projection_first_step'])+'个第一步即被联合投影分配非零执行速度：到达停止在投影前，中央安全投影本身已允许退让。因此目前没有证据证明这些黑盒失败具有跨controller共享的严格滤波器平衡点结构。圆形流程版未完成，不作死锁判断。'
    combined=result['three_scene_fixed_combined']
    fixed='；'.join(MN[m]+'相对只加安全投影'+delta(equal(combined,m)) for m in FIXED)
    learn=[]
    for scene,d in result['per_scene'].items():
        r=equal(d['learned_evaluation'],'learned_scene')
        if not r:continue
        comparisons='、'.join(MN[m]+signed(r['delta_vs_fixed'][m]['delta']) for m in FIXED)
        learn.append(FLOW_NAMES[scene]+'的场景内学习相对安全基线'+delta(r)+'，学习减去三种固定修正分别为'+comparisons)
    p4='迁移：三个已完成场景的有效状态池按类型、controller、状态逐层等权，'+fixed+'。两种冻结流程零样本均因旧四机圆环编码无定义而不适用。'+'；'.join(learn)+'；相对固定修正的完整配对区间见下表。二维拥挤随机场景没有预定8机/14机有效学习状态，完整三场景学习合并无定义，只另报窄通道与门口的独立评估子集合并。圆形流程版和学习型试点按用户决定停止，不计作失败或unknown，也不纳入合并。'
    gaps=[
        f"圆形围场五机随机场景流程版：{STOP_REASON}。9600条安全基线请求中保留{gap['completed_raw_retained']}条raw及journal，{gap['unwritten_discarded']}条未落盘请求丢弃、不记unknown、不恢复。未完成基线、固定修正与学习评估；部分样本不估计效果，不作四场景合并。单机筛选48000条完成证据保留。",
        '两类行为克隆controller及其十类型扩展按用户要求均未启动；没有试点成功率，也不能把未训练解释为能力门未通过。后续是否开展以指定v2文档为准，v1不再提交。',
        '窄通道真实t0未公布；图8首帧为1.8秒。本轮声明重置假设，右手规则活锁未复现，不能通过调参贴近论文。',
        '二维拥挤随机的700组原样本在流程版硬墙下剔除672组，仅二机21组、四机7组有效；8机/14机无有效样本，不能补抽、缩半径或以其他数量代替学习状态。',
        '两种冻结零样本流程依赖固定四机圆环观测、33维上下文和RingEnv恢复，在新场景无定义；不得改模型后称零样本。',
        '官方IMPC的r_min、ρ、η更新与先覆盖A再计算B的离散动力学不同于论文，原代码和所有轨迹保持；表II每步优化无解率缺少完整保存的QP状态。',
        '论文中的增量序列凸规划与分布式模型预测控制未取得实现，也未在v1重建，表II/III相应列未复现。',
        '圆形论文版的软墙权重、恢复规则和采样分布有声明假设，正式TCST全文未取得；已有DF/DR等数值unknown保留不重试。它们不补齐流程版缺口。',
        '独立评估仅执行冻结选择的动作，没有完整十二动作矩阵，平均/最大完整菜单遗憾不识别；训练人群完整菜单参照和已有四臂参照另列，不冒充独立评估oracle。',
        '部分论文版unknown仅保留起点，异常前未落盘局部轨迹不可恢复；除圆形论文版的段内配对检查外，其余论文驱动主要检查保存位置，不能统一宣称连续轨迹安全认证。',
        '没有全状态联合可解、共享死锁几何或统一任务进展裕量的形式证明，不支持固定数量checkpoint直接用于N=100。']
    if not audit['strict_full_coverage_pass']:gaps.append('完整性开工覆盖限制：'+audit['coverage_limitation'])
    if not (HERE/'FIGURES.receipt.json').exists():gaps.append('最后图未全部生成；数字结果已封存，图不作为主线前置条件。')
    (HERE/'GAPS.md').write_text('# v1缺口与停止边界\n\n'+'\n\n'.join(f'{i}. {g}' for i,g in enumerate(gaps,1))+'\n')
    text=['# v1三个已完成文献场景的迁移报告','',p1,'',p2,'',p3,'',p4,'',
          '## 独立评估与全部有效状态的区别','',
          '下表在有学习方法时统一使用32个独立评估state；拥挤随机使用全部28个有效state。仅纳入相应场景单机成功率≥95%的实例。完整上界、unknown敏感区间、逐类型结果、碰撞/超时/伤害与controller下降≥10个百分点计数见[MIGRATION_RESULTS.json](MIGRATION_RESULTS.json)和[AXIS_TABLE.md](AXIS_TABLE.md)。', '',
          '|场景|只加安全投影|全程固定η10|全程固定η13|人定双η进度切换|场景内学习|',
          '|---|---|---|---|---|---|']
    for scene,d in result['per_scene'].items():
        pop=d['learned_evaluation'] or d['fixed_full_inventory']
        text.append('|'+ '|'.join([FLOW_NAMES[scene],*[rate(r) if (r:=equal(pop,m)) else '缺口' for m in ['safety',*FIXED,'learned_scene']]])+'|')
    text+=['','|场景|学习−全程固定η10|学习−全程固定η13|学习−人定双η进度切换|','|---|---|---|---|']
    for scene,d in result['per_scene'].items():
        r=equal(d['learned_evaluation'],'learned_scene')
        if r:text.append('|'+ '|'.join([FLOW_NAMES[scene],*[ci_comparison(r['delta_vs_fixed'][m]) for m in FIXED]])+'|')
    text+=['','按类型→controller→state依次配对重抽5000次，同一state两个seed保持同簇。合并时各controller先在自身有效状态池取均值，再在类型内等权、最后类型等权；场景不强制等权。未知结果在成功率下界计0、上界计1，身份保留且不重试；这不是把unknown认定为真实失败。合并点估计与bootstrap保持相同逐层权重，见POOLED_POINT_ESTIMATE_CHECK收据。', '',
        '全部有效state的固定修正结果在JSON的fixed_full_inventory中，不能与32个独立评估state混作同一分母。所有提升和下降均为开发证据，点估计为正而区间包含零时不宣称严格正增益。', '',
        '## 构造、失败与执行机制','',
        '[PARAMETERS.md](PARAMETERS.md)逐行列出论文版与流程版动力学、周期、速度、安全层和终止差异。[SCENES.md](SCENES.md)、[CONTROLLER_PORTING.md](CONTROLLER_PORTING.md)保留四个场景的构造与完成的单机筛选；只有三个完成场景进入本报告迁移统计。几何有效、单机有路、联合可解、控制器成功和死锁解除五层证据分开。', '',
        '[FAILURE_DECOMPOSITION.md](FAILURE_DECOMPOSITION.md)给互斥失败拆分及九组敏感性；[GEOMETRY_CENSUS.md](GEOMETRY_CENSUS.md)仅描述4a操作候选，不下共享结论。[ARRIVAL_RELEASE_DIAGNOSTICS.md](ARRIVAL_RELEASE_DIAGNOSTICS.md)给已到达机器人被联合投影移动的逐条证据。[THEORY_CONDITIONS.md](THEORY_CONDITIONS.md)给条件式安全与任务进展关系，学习成功率本身不构成进展定理。', '',
        '[ORIGINAL_INSTANCE_RESULTS.md](ORIGINAL_INSTANCE_RESULTS.md)单列窄通道、门口state0；REFERENCE_ENDPOINTS.csv按明确物理判据给论文参照，未冒充IMPC预测cost_index。[SWITCHING_DIAGNOSTICS.md](SWITCHING_DIAGNOSTICS.md)区分真正进度触发、时限回退和未发生切换，不把方法名称当作发生过两段动作的证据。', '',
        '## 训练、复用与保留证据','',
        '场景内修正critic使用32个训练state、每类型两个实例与十二动作标签，按类型留一、源类型内选checkpoint，三个训练seed集成；独立评估选择在首条评估preflight前密封。完整训练菜单参照见[TRAINING_MENU_REFERENCE.md](TRAINING_MENU_REFERENCE.md)，不补齐独立评估oracle缺口。[BACKGROUND_TABLE.md](BACKGROUND_TABLE.md)单列名义控制和白盒背景，NOMINAL_TO_SAFETY_TRANSITIONS给逐条转移；不混入迁移主结论。[PILOT_RESULTS.md](PILOT_RESULTS.md)明确试点按用户决定未启动。', '',
        '全部完成raw、trace、checkpoint、选择收据和journal保留。官方DB只追加身份，经默认journal与单进程merger索引；DB_POSTFLIGHT区分已完成证据、数值unknown、旧0步错误分支及用户主动丢弃的8918条未落盘请求，未伪造结束结果。paper_bvc_v2是v1中的论文版修正分支名称，和接下来的分布式流程v2不是同一轮。', '',
        '图最后读取封存数字与已有几何，不调用控制器或改变物理、事件及成功标准。圆形场景的计算优化、严格字节回放及单worker尝试均保留日志；它们不构成已完成的圆形迁移结果。', '',
        '## 缺口与后续入口','',
        '[GAPS.md](GAPS.md)列原因与补齐条件。[HANDOFF.md](HANDOFF.md)包含构造调用、环境、缓存身份与原始路径，不依赖本对话。v1全部作业结束并完成收尾后，执行docs/prompt_baseline_literature_scenes_distributed_v2_20261020.md；在v2中v1目录只读。若无其他任务占用，按最新用户授权使用全用户24个CPU、5个GPU份额，每次提交前检查squeue。']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
    write(HERE/'DELIVERY_SUMMARY.json',dict(
        utc=utc(),evidence='DEVELOPMENT',integrity=audit,gaps=gaps,
        included_flow_scenes=list(FLOW_NAMES),excluded_circle=gap,
        all_authorized_computable_work_processed=True,original_four_scene_objective_completed=False,
        closed_under_explicit_user_scope=True,pending_research_jobs=0,
        no_four_scene_combination=True,report_sha256=sha(HERE/'REPORT.md')))


if __name__=='__main__':main()
