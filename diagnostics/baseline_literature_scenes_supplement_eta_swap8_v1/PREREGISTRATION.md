# 第二部分预注册：η粗筛及八机对径交换

第一部分COMPLETE已存在，其SHA在本目录INTEGRITY_START中。第二部分开始后不改第一部分封存结果。本轮全部为开发证据；新代码/输出仅本目录，官方journal默认位置及单进程merger例外。外部HJB-GNN仅参数来源，未运行/导入/安装，不训练critic，不改48个controller算法与参数。参数来源和三维模型到二维SI的声明见PARAMETERS.md。CPU总上限18（含1管理），GPU≤2，实际CPU计算；每批squeue检查所有用户，兼顾机器容量与他人作业。

## 几何与能力门

swap8 free/static/moving各32states，0原构型、其余±.1扰动；固定种子2026102311/12/13，moving机器人逐位沿用free；移动路径由父state确定且不受controller/seed影响，solo保留完整父障碍轨迹。前16训练、后16评估。几何净空、单机器人静态连通、容量和联合可解证据分开，未认证联合可解。图7实景没有可靠米制标尺，采用prompt写定4L后备，方向/线段/碰撞余量详见PARAMETERS与SCENES.receipt，不调布局。

48个冻结实例分别对3版本做单机器人能力门：32父state×8个agent×2seed×48=24576逻辑请求/版本，合计73728。成功下界≥95%即合格（unknown下界0、另报上界，不能重试unknown）。同配置、物理state、随机数、障碍路径、名义controller执行完全相同才复用。确定性gradient_mpc/PA/CBS可用明确忽略PRNG代码及回放证据为seed1保存0步别名；不同请求UID仍注册。能力门读取完整32state，按既有v2流程；η选择仍仅用前16训练state上的训练controller。不会因后16能力筛选把η选择称为完全外部测试。

安全层SI：pair各半，静态/移动障碍全责；只纳入当前中心/最近墙线段距离≤.5m；动态半径+.0125但碰撞半径真实.3。执行顺序、到达停止、v2三个修正基和Rc=1.25*d_safe不变。d_safe=.1001,Rc=.125125，控制器/修正仍保留v2全局位置及平均进度/到达位广播假设，不伪称全系统局部通信。

原速度球约束下先使用v2相同问题solver及数值重试；数学无解时先仅放松速度球半径、保持全部安全行；仍无解再对pair/墙/动态安全行最小化最大违反，去掉速度球，与第一部分同词典序原则。不裁剪输出，不把约束违反记unknown；速度超限、restoration、违反分别逐步记录。数值失败仍unknown，全部raw不可覆盖。速度放松的实际动作在新SI几何plant中执行，不以旧plant的速度检查异常掩盖无解回退。物理碰撞按原连续扫掠，moving waypoint步内转折逐段准确扫掠，三类碰撞分别计，任意类失败。

## 主线与η筛选

主线baseline：全部48×32states×2seed=3072/版本；固定η10/13、人定η13→10 progress：每版本合格数Q×32×2×3。最大36864主线请求，实际Q和官方preflight在独立各批CASES/PREFLIGHT收据封存。

粗筛场景：旧narrow/crowd/circle（不做door）与swap8 free/static。16固定η严格取controller_loco_real_flow_v1/protocol.json eta索引0–15的float64值。旧state/controller严格取v2 LEARNING_SPLIT中的前16训练state及每类型第1个合格training controller，保留原列表顺序；新state前16、每类型按v2训练实例优先顺序取第1个合格（无合格类型则该类缺口，不临时替换未合格实例）。seed0；每场景若8类均有则2048固定+768配对=2816训练请求。

固定候选按类型等权成功下界排序，完全平局索引优先。取前3全部6有序配对，theta=.968或round(6*max_steps/7)全体广播同步首次切换；配对平局按(a,b)索引字典序。每场景最佳固定/最佳配对在首条评估前封存，bootstrap不重新选择。旧评估state遵守v2 split；新后16。评估所有合格实例、seed0，与相同state/controller/seed的η13及人定双η配对，已有精确语义直接复用（包括方法标签别名），仅真正缺失执行。移动版不筛η，使用free选出的两个最佳动作在moving评估state、全部moving合格实例、seed0执行/复用。

预估5×2816=14080粗筛训练（扣历史复用），评估实际数由Q和冻结赢家决定。保存16+6完整排名、每类型描述性最佳动作、跨场景赢家是否不同。每类型仅1训练实例，明确描述性限制。

## 诊断、统计、完成

4a′/4a″、联合参照、实际局部乘积锥、九格W=2.5/5/10秒与tau=.02/.05/.1、原互斥类别、实例失败归类都用v2最终口径；static和moving墙前子类分开。只由封存轨迹计算，不调用controller补生成证据。timeout本身不是deadlock，controller级失败上的增益不计解死锁。按类型→实例→state等权，两个seed同簇，5000配对bootstrap，seed2026102003；单seed粗筛评估按同类型/实例/state簇。unknown下界0另报上界/识别区间。主线保留全部实例baseline，修正只合格实例，对比限定同实例/同state/seed，不把未运行填unknown。

每批提交前/后stat扫描v1/v2/第一部分及其他只读目录，任意非允许并行活动写入整体停。并行任务目录只能stat；新增docs/prompt_只记录metadata。原先29个等价回放先提交于慢stat返回前的程序性失序已停止，结果保留，独立完整扫描通过后仅恢复缺3条；这不等于允许越界写入。32旧轨迹逐位通过、strace和写路径审计通过后才允许新研究rollout。科学实现、场景、候选、能力门规则在首条新研究前封存；未知/数值失败不重跑；每批官方preflight，仅genuinely_missing，worker默认journals，单merger，结束postflight、完整性审计、报告后才能COMPLETE。

所需交付遵从总说明，图仅封存证据，不要求视频。精确请求数及复用数记独立REQUEST_LEDGER.jsonl/各批收据，不后改本预注册。
