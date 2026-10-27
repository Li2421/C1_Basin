from experiment import *
import csv
import collections
import subprocess
from paper_evidence import LABELS

FIXED=['fixed_eta10','fixed_eta13','human_progress']
MN={'fixed_eta10':'全程固定修正η10','fixed_eta13':'全程固定修正η13','human_progress':'人定双η·按进度切换','learned_scene':'场景内重新学习双η与进度阈值','safety':'只加安全投影'}

def delta(r):return f"{100*r['delta_vs_safety']:+.2f}个百分点 [{100*r['delta_ci95'][0]:+.2f}, {100*r['delta_ci95'][1]:+.2f}]"
def rate(r):return f"{100*r['success_rate']:.2f}% [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}]"
def equal(data,method):return next((r for r in data['statistics'] if r['type']=='type_equal' and r['method']==method),None) if data else None

def main():
    jobs=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-r','-o','%j']).decode().splitlines()
    assert not any('baseline2-' in j for j in jobs),jobs
    required=['MIGRATION_RESULTS.receipt.json','PAPER_REPRODUCTION.receipt.json','SAFETY_DIAGNOSTICS.receipt.json','BACKGROUND_RESULTS.receipt.json','PILOT_RESULTS.receipt.json','REFERENCE_ENDPOINTS.receipt.json','SWITCHING_DIAGNOSTICS.receipt.json','TRAINING_MENU_REFERENCE.receipt.json','DB_POSTFLIGHT.receipt.json','INTEGRITY_END.json','INTEGRITY_COMPARISON.json']
    assert all((HERE/n).exists() for n in required)
    result=read(HERE/'MIGRATION_RESULTS.json');paper=read(HERE/'PAPER_REPRODUCTION.json');audit=read(HERE/'INTEGRITY_COMPARISON.json');pilot=read(HERE/'PILOT_RESULTS.json')
    release=next(r for r in read(HERE/'ARRIVAL_RELEASE_DIAGNOSTICS.json')['summaries'] if r['population']=='qualified_blackbox')
    paper_by={(r['scene_code'],r['method_code'],r['N']):r for r in paper};safety=list(csv.DictReader((HERE/'SAFETY_BASELINE.csv').open()));gaps=[]
    gaps+=['窄通道场景真正t0未公布，图8首帧是1.8秒。重置假设与BVC右手启停细节不足以精确复刻论文；修正BVC实现能通过，没有复现要求的活锁。补齐需要原始初态与作者基线细节，不能调参追逐论文结果。',
           '二维拥挤随机场景的700个官方样本中672个未通过流程版硬墙几何检查；仅28个有效，预定8机/14机学习状态为0。场景内学习与行为克隆试点在该N范围不能开展，完整四场景学习合并无定义。补齐需另立不同采样/几何协议，不能补抽后混入本轮。',
           '冻结流程的四机圆环观测/上下文和RingEnv恢复在新场景无定义，两种零样本方法均不适用。改编码或训练新模型属于新方法，不能称冻结零样本。',
           '官方IMPC的r_min、ρ、η更新及覆盖A后的B计算与论文参数/方程有差异。保留原代码；不能把本轮官方实现结果称作严格论文方程复刻。其成功轨迹未保存完整每步QP状态，表II无解率不能完整重建。',
           '圆形围场五机随机场景的外圆软约束权重、恢复策略与分布细节未完整披露，正式TCST版未取得；DF/DR等求解器unknown均保留。未知结果不重试；这些条件限制完整复现判断。',
           '增量序列凸规划与原论文分布式模型预测控制没有取得对应实现，也未在本轮重建，论文表II/III的这些列是未复现项。',
           '独立评估状态仅执行已冻结选择的十二动作之一，没有完整十二动作矩阵，因此完整菜单oracle及其独立评估平均/最大遗憾不识别。四臂已有标签的交叉拟合/样本内参照另给，不冒充完整菜单oracle。',
           '原协议中的更早冻结A3/A0/探测选择器依赖旧环形任务及冻结表示，未定义为本轮新几何方法；专家轨迹只作训练示范，白盒与名义背景不进入主结论。',
           '圆形围场五机随机场景的5秒最大可行行程1.868米小于半径2米，静止窗口条件对正常行进亦成立；4a仍按原定义，但只能作操作候选，不能当精确平衡证明。',
           '学习成功率critic没有强制统一任务进展裕量，本轮不能给出普遍任务进展或全局死锁解除保证；不支持固定数量checkpoint直接用于N=100。']
    for scene,d in pilot.items():
        if 'gap' in d:gaps.append(f"{NAMES[scene]}行为克隆试点：{d['gap']['reason']}")
        if 'learning_gap' in d:gaps.append(f"{NAMES[scene]}行为克隆迁移学习：{d['learning_gap']['reason']}")
    if not audit.get('strict_full_coverage_pass',False):gaps.append('完整性审计覆盖限制：'+audit.get('coverage_limitation','见INTEGRITY_COMPARISON.json；不能补造开工前收据。'))
    if not (HERE/'FIGURES.receipt.json').exists():gaps.append('最后图未全部生成或核验；数字与主线结果已封存，几何及迁移数值可直接复用。')
    (HERE/'GAPS.md').write_text('# 缺口与适用边界\n\n'+ '\n\n'.join(f'{i}. {g}' for i,g in enumerate(gaps,1))+'\n')
    atext=audit.get('summary','完整性差异详见INTEGRITY_COMPARISON.json')
    p1='完整性审计：'+atext+'。全部新增代码与输出位于本轮目录，外部写入仅为官方DB追加登记/journal/单进程merger；未做Git commit。原始数据、checkpoint和数值unknown均保留。全部结果为开发证据。'
    p2='论文版：窄通道场景按400dpi图8首帧建立几何，无限时域模型预测控制与死锁解除能通过，但右手绕行点活锁未复现；二维拥挤随机场景按官方GUI采样，逐N成功率与unknown上下界的判定见PAPER_REPRODUCTION.csv，官方参数/离散动力学与论文存在明确差异；圆形围场五机随机场景的集中式与单步延迟伪协作避碰均100次满足论文收敛判据、0个静止未到达候选，但严格物理重叠与数值unknown分开报告，去中心化策略未能完整复现；门口场景和第二优先级文献交叉口的单个关闭实例静止未到达、开启实例通过，与要求的定性行为一致，不能据各一个实例估计总体100%/0%概率。门口代码实际推进30秒；15秒原元数据已勘误，不改轨迹。窄通道以及其余不完整项均标注“论文证据未复现或仅部分复现”。'
    pieces=[]
    for scene,name in NAMES.items():
        u=read(HERE/f'BASELINE_USABILITY_{scene}.json');qq=[r for r in safety if r['scene_code']==scene and r['qualified']=='True'];count=sum(r['category']=='filter_equilibrium_deadlock' for r in qq);pieces.append(f"{name}（{LABELS[scene]}）{u['qualified']}/48个实例通过单机门，合格实例中4a操作候选{count}个机器人局")
    p3='流程版：'+'；'.join(pieces)+'。静止未到达、被已到达者阻挡、零名义输入、振荡与缓慢前进按互斥规则报告，timeout未直接当死锁。窄通道原始实例中，合格黑盒的'+str(release['initially_arrived_agent_episodes'])+'个初始已到达机器人局里，'+str(release['moved_after_projection_first_step'])+'个第一步即被联合投影分配非零执行速度：到达停止在投影前，不会把机器人钉在目标；中央投影自身已经允许退让。圆形围场五机随机场景的窗口/半径/速度尺度使5秒静止条件不能识别物理静止，低锥残差也需结合约束偏置；因此本轮不能从候选普查宣称跨controller共享严格滤波器平衡点结构。'
    combined=result['four_scene_fixed_combined'];fixed='；'.join(MN[m]+'相对只加安全投影'+delta(equal(combined,m)) for m in FIXED)
    learn=[]
    for scene,d in result['per_scene'].items():
        r=equal(d['learned_evaluation'],'learned_scene')
        if r:
            f=r['delta_vs_fixed']['human_progress'];robust=r['unknown_sensitive_delta_ci95'];judgment='未知敏感差值区间全正' if robust[0]>1e-12 else '差值区间全负' if robust[1]<-1e-12 else '区间包含0，不能确认严格正提升';learn.append(NAMES[scene]+'的场景内重新学习相对只加安全投影'+delta(r)+'（'+judgment+'）'+f"，相对人定双η进度切换{100*f['delta']:+.2f}个百分点 [{100*f['ci95'][0]:+.2f}, {100*f['ci95'][1]:+.2f}]")
    p4='迁移：四场景有效状态池、按场景分别执行能力门后，'+fixed+'。两种冻结流程零样本均不适用。'+'；'.join(learn)+'。相对另两种固定修正的配对差值、逐类型结果、碰撞/超时/unknown与伤害见MIGRATION_RESULTS.json。二维拥挤随机场景的学习状态几何缺口使完整四场景场景内学习合并无定义；仅另报有数据的场景子集，不冒充四场景。'
    positive=[MN[m] for m in FIXED if equal(combined,m)['unknown_sensitive_delta_ci95'][0]>1e-12]
    p4+=('合并平均差值区间全正的固定做法为'+ '、'.join(positive)+'，不代表每个controller均获益。') if positive else '三种固定做法的合并95%区间均未支持严格正的平均增益。'
    text=['# 四个文献场景流程迁移报告','',p1,'',p2,'',p3,'',p4,'','## 主要结果与读法','','论文版与流程版的动力学、墙约束、周期、速度界与终止差异逐行列于[PARAMETERS.md](PARAMETERS.md)。原始实例state0单列于[ORIGINAL_INSTANCE_RESULTS.md](ORIGINAL_INSTANCE_RESULTS.md)，论文判据参照在REFERENCE_ENDPOINTS.csv。主表见[AXIS_TABLE.md](AXIS_TABLE.md)：有场景内学习时统一使用32个独立评估state比较；所有有效state的固定修正另存JSON，二者不混作同一统计人群。','','|场景|只加安全投影|全程η10|全程η13|人定双η进度切换|场景内学习|','|---|---|---|---|---|---|']
    for scene,d in result['per_scene'].items():
        pop=d['learned_evaluation'] or d['fixed_full_inventory'];cells=[rate(r) if (r:=equal(pop,m)) else '缺口/不适用' for m in ['safety',*FIXED,'learned_scene']];text.append('|'+ '|'.join([NAMES[scene],*cells])+'|')
    text+=['','类型→controller→state依次重抽5000次，seed0/1保持同簇；配对差值从同一重抽计算。unknown在成功率下界计0、上界计1，保留身份且不重试；这不是把未知真实结局宣布为失败。完整上界和未知敏感差值区间在JSON中。四场景合并把(scene,N,index)作为状态簇，类型/控制器等权；不是强制每个场景等权。','','## 失败与几何','','[FAILURE_DECOMPOSITION.md](FAILURE_DECOMPOSITION.md)给全部及合格controller的互斥分类和九组敏感性。[GEOMETRY_CENSUS.md](GEOMETRY_CENSUS.md)只做4a候选描述，不作共享性结论。离开闭锥的修正报告距离下确界；秩亏时为0，不虚构唯一非零最小修正。几何有效、单机有路、联合可解、控制器成功、死锁解除五层证据分开；当前没有全状态联合可解形式证书。','','## 训练、冻结与学习型试点','','八类型场景内critic只使用32个训练state、每类清单前两个实例与十二动作标签；按类型留一，内层checkpoint只用源类型验证，三个训练seed集成。选择在独立评估preflight前密封。原状态输入换为physical12，其余架构和训练规则保持。学习型试点在原48实例和白盒全部场景背景/固定修正完成后开展，严格复用既有成功冲突搜索多机规划训练轨迹；见[PILOT_RESULTS.md](PILOT_RESULTS.md)。如有十类型补充，八类型结果、模型和密封选择保持独立，旧同语义标签复用。','','## 来源、复用与执行边界','','[HISTORY_CHECK.md](HISTORY_CHECK.md)、[CONTROLLER_PORTING.md](CONTROLLER_PORTING.md)、[SCENES.md](SCENES.md)记录历史兼容性、48个参数实例、场景与几何检查。历史P2执行链与八类新场景实例的严格字节回放收据、精确墙缓存及原网格端点无入边证书均在本目录。只改本轮子类/新文件；原物理、事件、成功判据与默认投影容差保持。数值完成后才生成图；图不参与控制器或成功定义。','','官方DB只追加登记，登记前后旧行内容哈希一致，rollout只经默认journal与官方merger。DB_POSTFLIGHT.json验证所有活动请求均已索引；旧8条0步实现错误分支单独保留、不继续计算。时间元数据与动力学解释勘误见TIME_METADATA_CORRECTION.json和OFFICIAL_DYNAMICS_AUDIT.json，全部轨迹原样保留。数值unknown不重跑。','','论文来源：[Chen等，IMPC-DR](https://arxiv.org/abs/2202.06071v4)、[Jankovic等，多机CBF](https://arxiv.org/abs/2207.04915)、[Chandra等，社交小游戏导航](https://arxiv.org/abs/2308.10966v6)、[Zhou等，缓冲Voronoi单元](https://msl.stanford.edu/papers/zhou_fast_2017.pdf)。门口论文发表于Autonomous Robots 49:12（2025）；按本任务写作约定，它是不错的期刊，但层级低于IJRR、IEEE T-RO。','','## 缺口与复用入口','','[GAPS.md](GAPS.md)列出证据限制及补齐条件。[THEORY_CONDITIONS.md](THEORY_CONDITIONS.md)给出条件式安全与进展关系；学习结果本身未建立统一任务进展保证。[HANDOFF.md](HANDOFF.md)包含构造接口、环境、缓存身份、恢复命令与可直接复用路径，不依赖本对话。']
    text+=['','完整十二动作菜单的训练人群交叉拟合与样本内参照见[TRAINING_MENU_REFERENCE.md](TRAINING_MENU_REFERENCE.md)，仅使用既有训练标签、没有新增rollout；它不能补齐独立评估完整菜单遗憾。实际按进度触发与时限回退的拆分见[SWITCHING_DIAGNOSTICS.md](SWITCHING_DIAGNOSTICS.md)，不能把双η名义名称当作实际发生过两段动作的证据。集中式投影移动初始已到达机器人的逐条证据见[ARRIVAL_RELEASE_DIAGNOSTICS.md](ARRIVAL_RELEASE_DIAGNOSTICS.md)。论文版圆形场景记录保持加速度段内的连续配对距离最小值；其余论文驱动仅能检查已保存位置，门口第二次MPC还存在观测重置，不能统一宣称连续轨迹安全认证。部分论文版unknown只保留起点，异常前未落盘的局部轨迹是保留的证据缺口。']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n')
    write(HERE/'DELIVERY_SUMMARY.json',dict(utc=utc(),evidence='DEVELOPMENT',integrity=audit,gaps=gaps,all_authorized_computable_work_processed=True,pending_research_jobs=0,report_sha256=sha(HERE/'REPORT.md')))

if __name__=='__main__':main()
