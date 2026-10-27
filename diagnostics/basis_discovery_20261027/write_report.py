"""Generate the discovery report from completed, versioned summaries."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent

def load(n):return json.load(open(OUT/n))
def f(x):return '—'if x is None else f'{x:.3f}'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])

def main():
    s=load('summary.json');names=list(s['primary_single']);verification=load('verification.json')
    screen=table(['方向','主预算候选','初态救回/14','对照保留/8','候选碰撞','失败状态救回/6'],[
        [n,s['primary_single'][n]['candidates'],s['primary_single'][n]['rescued'],s['primary_single'][n]['guards'],s['primary_single'][n]['collision'],s['restart'][n]['rescued']]for n in names])
    static=table(['方向','非零状态/22','增rank状态','改变原投影动作状态','平均非零列的B3外分量比例'],[
        [n,s['static']['columns'][n]['nonzero'],s['static']['columns'][n]['rank_gain_states'],s['static']['columns'][n]['projection_changes'],f(s['static']['columns'][n]['mean_outside_B3'])]for n in names])
    cost=table(['方向','构造kernel μs/call','其中SVD μs/call','整步计算 ms','全局零列步占比'],[
        [n,f(s['costs'][n]['kernel_us_per_call']),f(s['costs'][n]['svd_us_per_call']),f(s['costs'][n]['mean_total_ms_step']),f(s['costs'][n]['unavailable_fraction'])]for n in names])
    matched=table(['匹配替换','困难救回/14','对照保留/8','碰撞候选'],[
        [n,v['rescued'],v['guards'],v['collision']]for n,v in s['matched_controls'].items()])
    fixed_tables={}
    for split,metrics in s['fixed'].items():
        fixed_tables[split]=table(['固定策略','成功/16','timeout','成功条件时间 s','全队列截断时间 s','计算 ms/step'],[
            [n,v['success'],v['outcomes'].get('timeout',0),f(v['conditional_time']),f(v['truncated_time']),f(v['ms_per_step'])]for n,v in metrics.items()])
    paired=table(['确认集策略','配对成功数','方法/B3时间 s','救回/损失','成功率差95%区间'],[
        [n,v['paired_success'],f(v['paired_method_time'])+'/'+f(v['paired_B3_time']),str(v['rescued'])+'/'+str(v['broken']),str(v['paired_rate_ci'])]
        for n,v in s['fixed']['confirm'].items()if n!='B3_frozen'])
    scene=table(['确认集场景（各4次）','B3','G+C2固定','G+C2释放','T5固定','RHR B'],[
        [sc]+[x[n]['success']for n in ('B3_frozen','compressed_fixed','compressed_release','T5_fixed','RHR_B')]
        for sc,x in s['fixed_by_scene']['confirm'].items()])
    ctl=table(['确认集nominal（各8次）','B3','G+C2固定','G+C2释放','T5固定','RHR B'],[
        [sc]+[x[n]['success']for n in ('B3_frozen','compressed_fixed','compressed_release','T5_fixed','RHR_B')]
        for sc,x in s['fixed_by_controller']['confirm'].items()])
    recovery=table(['方向','成功恢复候选','救回/6','碰撞','出现诊断进展的候选','进展条件时间 s'],[
        [n,v['success_candidates'],v['rescued'],v['collision'],v['diagnostic_progress_episodes'],f(v['conditional_progress_time'])]for n,v in s['restart'].items()])
    final=load('CONFIRMATORY_FINAL.json');minimum=s['minimal'];oracles=s['oracle']
    text=f'''# C1：统一 Correction Dictionary Discovery 与 Oracle 筛选

2026-10-27；基线 commit `5da96e7`；独立分支 `research/basis-dictionary-20261027`。代码与完整证据位于 `diagnostics/basis_discovery_20261027/`；交付提交见该分支 `git log -1`。

**本轮最有价值的新方向是 C2（局部净空梯度）；当前找到的最小有效候选族是 [G,C2]。** 它在开发有限候选记录中覆盖22/22，在新确认集的冻结16候选池中覆盖16/16，在14个真实失败轨迹状态的恢复池中覆盖14/14。对应 B3 在确认集最多80个候选的合并池中为14/16，恢复 B3@48 为8/14。但 [G,C2] 的单一固定参数只有10/16，释放 C2 后12/16，均低于固定B3的14/16：**表达能力收益成立到有限实验层面，部署策略收益尚未成立。** 不应据此直接删除生产 P/R 或替换原 B3。

实际完成 **{s['sources']['new_rollout']:,} 次新闭环仿真**，精确复用旧 B3/T1/T2 结果 {s['sources']['exact_prior_behavior_reuse']} 条，共 {s['records']:,} 条候选记录；另外复用上轮 B3@168/296 搜索确定困难队列，不重复生成该队列。未训练任何模型，未使用GPU。用户解除18CPU限额后，检查发现单节点24CPU全部空闲、Slurm无运行作业，使用cpu分区24CPU、每worker单核、BLAS单线程。累计新仿真worker墙钟时间 {s['new_simulator_seconds']:.1f}s（约{s['new_simulator_seconds']/3600:.2f} worker-hours），不是实际总CPU时间；Slurm accounting storage 未启用。

## 1. 先审计、冻结与证据复用

已读取原 `AGENTS.md`、`shared_rollout_db/README.md`、上轮报告、实际 `runtime_new.py`、`swap_safety.py`、`swap_safety_user_fix_20261023.py`、`correction_revision_20261021.py`、`basis_families.py` 及上轮逐候选结果。证据路径、旧结果SHA256、初态/nominal冻结配置见 [PROTOCOL.json](PROTOCOL.json)。

参考包仍缺失：本轮再次检索研究目录、`/tmp`、`/mnt`，未找到 `C1_B5_validation_v0_20261027.zip` 或 `IMPLEMENTATION_SPEC_CN.md`。上轮也记录缺失，故没有运行该包的 `run_checks.py`；自有回归不冒充该包验收。上轮的旧四场景checkpoint回测与29,952次B5法锥顶点数值检查作为历史证据保留于 [上轮报告](../b5_oracle_20261027/report.md)，不能把这些结果当作新15列的完整认证。

真实流水线依然是：原nominal及原限速/到达掩码 → 第一次原安全投影 a → correction → 第二次原安全投影 → 原物理环境。原 P 使用目标方向的 Tikhonov 正交残差，RMS尺度3.303687238760696；R为全体平均分离，但生产代码还有接触门控。没有把讨论中的“无门控全体平均R”换进生产。

“raw nominal”在本实验指**第一次安全投影之前、原限速和到达掩码之后**的原流水线输入 `ref`；直接从该行取值，没有第二次调用nominal，也没有改其随机key。原 nominal 为 `centralized_sampling`、`gradient_mpc` 的冻结原生优化配置，当前主流水线不使用神经网络checkpoint。旧四场景checkpoint、控制器、安全层及既有结果均未改动；本轮没有重跑旧四checkpoint全套测试，不能将新结论外推到其旧联合solver。

当前真实联合速度域是原各机器人二维域的笛卡尔积。保留原CBF行、neighbor radius=.5、responsibility=.5、径向速度锥、Clarabel、重试及已有restoration。所有成功/碰撞/timeout仍来自原环境，不新增hard HOLD，不把timeout当deadlock。上轮发现的墙碰撞几何与CBF间距不一致**保持原样**，坏候选的碰撞照常计失败；本轮没有借修复安全模型取得收益。

新增 [basis_dictionary.py](../../shared_control/basis_dictionary.py) 和 [runner.py](runner.py)；原文件不改。runner从原生产函数源码只插入一个拿到 `safe/ref/filters` 的target hook。66个已保存真实状态的零扩展逐位检查通过，414次原投影动作对照误差0；T1/T2与上轮一致的配置通过逐位检查后才复用旧轨迹。[checks.json](checks.json)。

每批先登记完整物理状态、controller UID及包含所有系数/recipe/源码hash的配置，再运行官方preflight；只提交真正缺少的记录。worker写journal，父进程串行官方merger，阶段结束立即读回验证。没有重现上轮缺失state provenance的登记问题。最终 {verification['rows']:,} 条缓存身份一致、0冲突；原始结果不覆盖。[verification.json](verification.json)。

## 2. 完整候选库及固定构造规则

原G/P/R直接由生产实现提供。新增15类全部实现且全部进入同一筛选；系数固定整局，字段随真实状态重算。这里“固定”指系数和构造规则，不表示运动向量不随状态变化。主参数盒：G∈[.5,1.25]，P∈[-.5,.5]，R∈[0,.75]；T1–T5、P2、Y2、C1∈[-1,1]，其余新增列∈[0,1]。删列是独立族消融，未改原B3合法范围。全部操作为ADD，没有偷偷使用BLEND。

近活跃尺度统一定义为 `tau_v=.05*vmax`。原行 `A_iℓ u_i ≥ b_iℓ` 在安全参考a处的归一化slack为 `(A_iℓ a_i-b_iℓ)/||A_iℓ||`。使用实际行，零行跳过；不依赖未来成功标签。

| 类别 | 可复现规则 |
| --- | --- |
| T1 | JG；G已按原速度限幅，避免重复浮点限幅，保持上轮精确别名 |
| T2 | 接触半径1.25安全直径的连通组；组内共同κ=vmax/max(vmax,max半径)，乘J(p−组中心)；孤点为0 |
| T3 | 只使用下述唯一选定冲突pair；相反的J(p_i−p_j)，共同限幅，不叠加多个pair |
| T4 | 只用slack≤tau_v的原墙行；选择归一化slack最小者、行序打破平局，取vmax乘单位法向的J旋转 |
| T5 | 原近活跃归一化行的零空间投影；主recipe d=JG，另独立测试d=G和d=raw nominal；近速度球也加入其一阶法向 |
| P2 | 使用原P的同一个Q、epsilon=1e-6*vmax，对raw nominal求正交残差并限幅；主定义不加原P的RMS尺度，另做匹配尺度对照 |
| R2 | 原pair行slack≤3tau_v；w=exp(−max(slack,0)/tau_v)，加权(p_i−p_j)/(sum w+1e-12)，最后限幅 |
| R3 | 合格pair中先选slack最小，再按距离、index打破平局，使用该唯一对象的分离方向；无合格对象为0 |
| Y1 | 仅yielder取−G；G=0时保留退化并记录，没有后备方向 |
| Y2 | 仅yielder沿passer当前目标方向的J侧向，左/右由显式系数符号记录；passer目标方向为0则为0 |
| Y3 | 从yielder指向远离passer当前位置—目标线段最近点的方向；共线时固定±J目标方向，两个recipe分别记录 |
| Y4 | yielder软抵消−a，不抵消其余修正，也不增加硬保持约束 |
| S1 | yielder −G、passer +G，其他分量0 |
| C1 | 只识别.5邻域内的两条有限直墙：投影脚点在线段内部，法向相反/切线平行误差≤15°；选总距离最小pair，沿通道轴并以G定正向；没有可信通道即0 |
| C2 | 下述明确的平滑局部净空势函数梯度；使用原安全模型的半径/间距，不改变几何或solver |

角色选择为确定的在线几何规则：首次出现近活跃pair时，按最小slack、index选唯一pair；从其中未完成机器人里选目标距离较短者为passer（同距按index），另一个为yielder。身份冻结到本局结束，仅在pair仍处于.5距离内且至少一方未完成时激活。已到目标者可以成为yielder，Y1退化会记录。没有用Oracle未来标签换角色；没有实现多pair累计或动态角色重选。

新列原则上按原vmax逐行限幅；T2/T3采用共同缩放保留结构。Y4在原可行速度域中限幅不改变−a；本轮未发生restoration/速度放宽。原P保留原尺度且不额外裁剪。不同列的总范数和作用机器人数量并不相同，等候选预算不等于等能量预算，低分不能直接解释为理论无用。

### C2精确定义与数值核验

对原filter实际出现的pair/wall邻居，令

- pair间距δ_ij=||p_i−p_j||−(2r+m_agent+1e-4)；
- wall间距δ_iℓ=dist(p_i,wall_segment)−(r+r_wall+m_wall+1e-4)；
- tau_d=.25×安全直径；空邻域项为0。

定义 `D(x)=vmax * sum_i [-tau_d * log sum_k exp(-delta_ik/tau_d)]`，取 `C2_i=sat_vmax(∂D/∂p_i)`。pair出现在两个局部softmin中，微分时两个端点和两项贡献均计入；原始metadata保存D/vmax。使用稳定log-sum-exp。当前场景没有moving obstacle；该列不处理moving行。邻域集合/最近线段特征切换处仅分段光滑，不宣称全局光滑或无局部极值。

在22个真实状态上，对固定邻域集合做中心差分，epsilon=1e-6；限幅后梯度最大误差 **{s['C2_fd_max_error']:.3g}**。不是凭手工法向近似冒充完整梯度。[mechanism_validation.json](mechanism_validation.json)。

T5在真实分布式域上做等价的二维块SVD，不用巨大稠密联合矩阵；阈值 `max(1e-12,1e-10*smax)`。矩阵分解开销单独计时。近活跃阈值0.01/0.05/0.1*vmax使22个状态中{s['T5_near_threshold_rank_changed_states']}个的rank发生变化；SVD相对阈值1e-8/1e-10/1e-12下rank变化状态数为{s['T5_svd_tolerance_rank_changed_states']}，最大零空间残差6.19e-16。它只取活跃超平面的切空间，**不覆盖整个单侧可行锥**。

## 3. 统一筛选流程与预算

开发队列来自上轮真实证据：14个B3@168残余案例（其中pilot也做过@296），再加8个原B3成功对照，覆盖四场景及两个nominal。基线η为旧结果选出的成功/最接近完成的合法B3；再用共同中点(.875,0,.375)作第二个nuisance配置。不是每个新方向手工挑不同B3基线。

1. 每个新增列主预算4候选：两个B3基线 × 两个增益。带符号列±.75，非负列.5/1。T5另外两种seed各2候选、Y3另一共线侧2候选，附加recipe单列报告，不混入@4。
2. **所有105种两列组合**均有2个Sobol增益配置，在预先按三类困难场景各取2例的6例pilot上测试。没有因单列低分而省略潜在互补组合。
3. 按pilot成功覆盖、碰撞和残余距离自动选择12对，在22例上各扩4配置；再加入24个随机稀疏3–5列组合和4个全15列组合。具体搜索顺序、种子及冻结选择见 `EXPANSION_FROZEN.json`。
4. 从成功记录自动删列，枚举18列支持集的262,144个子集，用bitset动态规划求“能覆盖当前全部已解决案例的最少列数”。随后对最小族的边界做逐列删除；选最多6个接近最大覆盖的低一维子族，只在其未覆盖案例上补16个Sobol点。维数经历4→3→2；再测试G单列/C2单列的剩余案例，未得到单列全覆盖。
5. 选择过程只看开发seed0。第一次独立集为四个新场景states18/19、seed3、两个nominal共16例；压缩后另开states20/21、seed4的全新确认集16例。各方法同初态/nominal使用matched future seed。每组只有8个物理状态块，不做B15/Q16认证。
6. 再从原失败B3轨迹提取最停滞的真实状态，重新初始化同一个物理系统、原horizon和新的匹配未来流，区分从初态预防与从失败状态恢复。

所有候选均包含原B3参考池作为比较/回退证据；“纯压缩族”与“B3池加压缩族扩展”分别报告。Oracle@K只是有限池的成功覆盖，不是连续参数空间最优或不可解证明。原始逐候选η、recipe、角色激活、成本、碰撞、trace及UID保存在 `candidates.jsonl`；每批候选顺序保存在 `*_planned.json`。[预算曲线](budget_curves.csv)。

## 4. 全15类筛选结果

“对照保留”表示该列的4候选里至少一个成功，不能解读为任意固定系数都保留能力。下表的失败状态分母6，是B3@16后仍未恢复的十字口状态；随后B3增至48也没有减少这6例。

{screen}

T5主recipe的12/14是@4结果；加d=G、d=raw的附加recipe后可救13/14，但预算增大。两个附加recipe各只救1例。不能把多recipe联合成绩伪称为单一固定T5。Y3的另一个共线侧recipe也单独保存。

P/P2、R/R2/R3匹配替换结果如下。P2匹配RMS组使用原3.303687尺度，系数会超出主新列盒，因此不纳入主最小族分析。R替换组把原R系数设0后以同样.375/.75增益使用R2/R3。

{matched}

这次真实nominal在很多停滞状态本来就沿目标方向，P2在22个静态状态中仅4个非零；理论上P=0而P2≠0的构型通过了测试，但不能据此说raw nominal是本系统主要缺口。R2直接替换比R3更有证据；R3作为补充项的6例收益与替换组0例的差异说明不能简单把原R删掉。

## 5. 机制：新增方向、投影抵消与恢复

静态帧来自旧失败轨迹后2/3内最大机器人速度最小的未完成时刻；仍然不是数学死锁认证。T1/T2/Y等在此作为一次新激活，冻结角色与原整局的历史角色不同。完整norm、夹角、rank、正负扰动的原solver响应见 [static_states.json](static_states.json)。相对B3独立分量均值仅对norm>1e-8的列统计，避免零列数值噪声误导。

{static}

六个困难十字口帧，每个机器人的T5活跃行rank均为2，T5范数只有约1e-16：SVD正确，但切空间已缩为0。原安全不等式仍可能允许主动增加净空的单侧运动。C2在原solver下使这些状态实际恢复，T5没有。这是“只沿活跃边界相切”过度保守的实证，不能将T5=0解释为联合可行域没有脱困动作。

恢复实验给每个失败状态一个**新的完整原horizon**及匹配未来流，并非精确重放旧RNG后缀。B3@16已恢复8/14（对径及障碍部分），增加32个Sobol后仍为8/14。这个结果本身说明上轮失败不都等于不可解死锁：时间窗、轨迹阶段、未来采样也有作用。以下只分析剩余6个近停滞十字口案例。

{recovery}

局部进展诊断为“所有初始未完成机器人各减少.05m目标距离或到达”；它可能漏掉暂时后退但有效的动作，只作轨迹描述，不替换原任务成功判据。C2单列扩展救回6/6，Y1/S1各5/6，T5为0/6。冻结的压缩族16候选池后来单独在全部14个恢复状态测试，**纯[G,C2]为14/14**，并非只统计B3遗留的6例。

Y2与Y3在许多固定状态上点态共线，但Y3的侧向符号由当前线段净空决定，闭环中不等于整局固定符号的Y2。Y4在静态状态均未显著改变原投影动作，且a本已很小，不能期待仅取消a生成缺失的主动脱困动作。C1只在1/22状态识别出可信通道，当前弱成绩主要是可用性不足；本轮未对旧Double-Bottleneck checkpoint做专项回测，不宣称通道方向普遍无用。

## 6. 自动压缩与独立Oracle

初始成功witness的最小完整覆盖有十个四列族，例如[G,R,R3,C2]、[G,R,S1,C2]、[G,R,T3,C2]及[G,R,T1,T5]。进一步统一删列及低维补搜后，最小有限witness-cover降为 **{minimum['min_columns']}列：{minimum['minimal_families'][0]['names']}**，覆盖22/22。纯G和纯C2经第三轮补搜仍未全覆盖。精确的是“已测试成功记录里的最小支持集”，不是连续空间中一维绝无可能的数学证明。[MINIMAL_FAMILIES.json](MINIMAL_FAMILIES.json)、`COMPRESSION_*`记录全部选择与追加成本。

| 独立队列 | B3@16 | B3@48 | B3@64 | 历史定制B3≤16 | 两个B3池联合≤80 | 新族结果 |
| --- | --- | --- | --- | --- | --- | --- |
| states18/19，seed3，16例 | {oracles['test']['B3_16']} | {oracles['test']['B3_48']} | {oracles['test']['B3_64']} | {oracles['test']['B3_informed16']} | {oracles['test']['B3_union_max80']} | 三个冻结族各扩≤16到B3@48后均16/16 |
| states20/21，seed4，16例 | {oracles['confirm']['B3_16']} | {oracles['confirm']['B3_48']} | {oracles['confirm']['B3_64']} | {oracles['confirm']['B3_informed16']} | {oracles['confirm']['B3_union_max80']} | 纯[G,C2]@16为16/16；B3+该扩展也16/16 |

B3@64由原16菜单+48 Sobol组成，早已成功的案例不重复扩搜。额外加强基线的“历史定制≤16”取对应开发场景族/nominal的states8…15各自最佳合法B3，再加8 Sobol；不使用新测试结果选η。两池联合仍不能消除表中残余。新族16候选池为开发选择的最多8个参数+8 Sobol，跨所有确认案例冻结；不是逐测试状态重新调系数再冒充固定方法。池的训练信息和构造不同，虽然新族候选数更少，仍不能把对比当作连续参数空间的严格支配证明。

这次补跑了**纯[G,C2]候选池的全部确认案例**，没有将“继承B3成功+只救剩余状态”冒充纯两维族的16/16。两个nominal都参与，确认集具体初态和种子不同于第一组；这些小规模结果支持优先研究该低维族，但不覆盖旧四checkpoint、三机器人专用链/闭环、N=100或所有局部围堵机制。

## 7. 固定策略结果：Oracle收益尚未转化为部署优势

第一次独立集的family0=[G,R,R3,C2]，但其选出的最佳固定参数R3=0，实际是G,R,C2；family1=[G,R,S1,C2]；family2=[G,P,R,T5]，其P=0，实际是G,R,T5。全部先在22例上做共同参数迁移，再冻结，测试不选η。

{fixed_tables['test']}

新确认集的压缩固定参数为 `eta_G={final['best_fixed']['eta'][0]:.9f}, eta_C2={final['best_fixed']['coefficients'].get('C2',0):.9f}`，P=R=0。释放策略仅在原horizon/3后将新增列系数置0，仍保留同一G；单列数没有增加，但这是独立的时序策略，不能算固定η的表达收益。

{fixed_tables['confirm']}

{scene}

{ctl}

所有这些固定测试均无碰撞、无数值失败。G+C2固定策略能救回B3的2例，但损失原本成功的6例；释放后仍损失4例。保持净空的修正可能阻碍后期贴近目标，参数/停止时机是后续策略选择问题，不能仅凭候选族覆盖率上线固定参数。

成功条件均值不与不同成功集合混用。全队列截断时间将每个失败赋其原horizon；配对成功时间单独列出：

{paired}

确认集固定B3成功率区间[.625,1]、G+C2固定[.3125,.875]、释放[.5,1]。区间以8个scene/state物理块bootstrap10,000次、保留块内两个nominal，是描述性小样本区间。Oracle16/16不报虚假的[1,1]泛化保证；有限池成功并不等于未来成功概率为1。

## 8. 计算成本与失败记录

下表kernel仅包含候选构造及几何/角色信息，不包含原nominal、两次原安全投影和hook中重复取得原GPR的成本；整步墙钟包含这些操作。T5的SVD成本单独记录，均值包括三个seed recipe。候选动作、事件和成功标准没有为计时而改写。

{cost}

新仿真共{s['sources']['new_rollout']:,}条，逻辑结果{s['records']:,}条，numerical failure={s['numerical_failures']}，restoration steps={s['restoration_steps']}，原线性约束violation steps={s['original_constraint_violation_steps']}。候选中仍有碰撞，完整分布见summary；尤其T1类在十字口可能触发上轮已确认的原墙体模型间距不一致。不能把“原QP约束通过”说成物理碰撞无风险，也没有删除这些不利结果。

CPU作业日志中JAX在导入时尝试发现CUDA插件，会打印CUDA_ERROR_NO_DEVICE；Slurm未分配GPU，`JAX_PLATFORMS=cpu`固定，实际闭环在CPU完成。该导入日志不是闭环numerical failure，原每个rollout的error字段仍独立保留。

## 9. 本轮判断与下一步边界

- **优先保留C2**：它同时有初态覆盖、失败状态恢复、低维压缩及两nominal独立确认支持；不是只有静态rank收益。
- **Y1/S1适合作为角色化恢复备选**：6个强B3恢复失败中各救5个。当前最小族不需要它们，但更广泛阻塞/到达机器人场景尚未排除其价值。
- **T3、Y2、Y3有补充价值**：小预算能救较多初态难例，角色与侧向结构明确；Y2/Y3点态相关但闭环规则不同，不能直接合并成绩。
- **T5更像预防性方向候选**：初态收益强、真实满秩停滞处为0，不能作为完整脱困锥的替代。考虑其用途时必须保留单侧可行方向。
- **P2、Y4、C1在本批没有充分增益证据**：P2经匹配RMS也未补困难覆盖；Y4无法凭零/小a产生主动动作；C1主要缺可信通道输入。不是它们普遍无用的证明。
- **R2值得保留替换对照，T2/R3/T4保留场景性备选**：R2直接替换原R有增益；T4在此近墙定义下弱，仍不能等同于所有障碍切向构造无用。

建议下一阶段以 **a+eta_G G+eta_C C2** 为首选低维研究族，保留原B3候选作为回退，并用现有数据研究系数和退出时机的选择；本轮不训练Generator/Critic。固定参数结果明确不支持直接替换生产B3。尚未证明全局最小Basis，也未解决旧墙体模型不一致；所有结论限定于冻结pipeline和这些实际测试队列。

## 10. 交付与复现

[实现](../../shared_control/basis_dictionary.py)、[生产适配](runner.py)、[统一筛选](screen.py)、[自动压缩](compression.py)、[独立确认](confirmatory.py)、[恢复实验](restarts.py)。

[summary.json](summary.json)、[单列原始汇总CSV](single_screen.csv)、[预算CSV](budget_curves.csv)、[图PNG](results.png)、[导出PDF](results.pdf)、[静态状态审计](static_states.json)、[梯度与SVD数值验证](mechanism_validation.json)、[缓存核验](verification.json)。

全部逐候选原始数据在 `candidates.jsonl`，轨迹在 `traces/`；`*_planned.json`、`*_preflight.json`、`*_execution.json`和`*_merge.log`记录预算、精确身份、复用和执行。`PROTOCOL.json`、各`*_FROZEN.json`保留选择时依据。Git提交代码、文档和小型摘要；JSONL/NPZ/SQLite等原始证据保留本地，Git不是它们的备份。

可恢复命令见 [README.md](README.md) 与 [resume.sh](resume.sh)。旧实验目录和旧生产文件未修改。阶段完成标记和精确UID避免重复rollout；重算统计/图只读取已完成证据。
'''
    if (OUT/'baseline_report.md').exists():text+='\n'+(OUT/'baseline_report.md').read_text()
    (OUT/'report.md').write_text(text);print('Wrote report.md')

if __name__=='__main__':main()
