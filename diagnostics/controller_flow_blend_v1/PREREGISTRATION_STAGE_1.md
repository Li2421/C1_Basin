# Stage 1：连续 Flow 融合族与 regime 边界 pilot

## 目标及历史证据缺口
Stage 0已复用57,344个原始Flow训练标签格；既有regime集中于η10，η13仅两个。需要辨识基础Flow执行程序连续改变时η10/η13 regime能否在可胜任任务的范围内切换。融合controller执行语义是新的；更改表示不会使既有原始Flow标签失效。先对12个精确新UID进行shared DB preflight；未知记录不重跑。该pilot不是逐state selection、泛化或因果根因检验。

## 执行定义（实现前冻结）
A顺序[base_u_macflow,base_u_v10_local_macflow]，B顺序[base_u_plus_w_v4_wide_macflow,base_u_v9_local_macflow]；A外层/B内层，共4pair，每pairλ=[.25,.5,.75]。使用原Flow adapter和native v10 runtime，各checkpoint在自身观测chart和normalizer下查询速度，转换为世界坐标。共同ξ由原key生成相同的8维标准正态；D_X(ξ)是X的动作反归一化及世界chart映射，初值h=(1−λ)D_A(ξ)+λD_B(ξ)。每10个pseudo-step在当前共同h处独立反映射到A/B chart查询，u=(1−λ)u_A+λu_B。

OrthoFlow3的物理修正先按原pipeline从nominal flow的硬投影safe参考计算，之后整个u+δ逐pseudo-step做原admissible-field投影。nominal flow定义为同一个融合速度场在无修正、无pseudo-time投影时的10步Euler终点h；对该h用两个锚点各自的原terminal clipping和speed-bound chart规则处理、世界坐标凸组合、最后原float64 speed re-bound。这样没有新增物理/安全/评价规则，明确了原prompt未指定的basis参考流的terminal chart规则。

λ=0/1使用原锚点执行路径，保留原float32归一化采样和native clipping顺序，避免数学等价world-space Euler在float32舍入下与旧缓存不兼容。边界精确分支公开记录，不将数学连续性宣称为浮点逐值连续；静态audit不仅检查分支与锚点最大误差≤1e−10，还检查绕过分支的world-space公式与锚点执行动作差、及λ=1e−6和1−1e−6附近误差。若未分支公式差>1e−4则视为实现/语义问题停止新增outcome，修复需单独附录。所有查询封锁env.step。环境、H700、dt=.05、FF、安全层、K16、seed与评价器沿用P1/round4；JAX x64=false。声明式identity包括两锚点UID/完整payload、λ、代码SHA、nominal参考规则与数值边界分支。

## 样本与成本门
复用round4 S_train前24个物理初态，不筛选容易state。K4 indices=[2,4,10,13]，原seed0–3。12×24×4×4=4608格，cap5000。约100个timing请求固定为全missing案例列表中等距98个，计入4608，不另造请求。仅运行真正缺失，未知永久保留。锚点的旧标签是每state随机8-head，**不能假设前24state的K4完整**；端点曲线只报告真实请求覆盖，不补跑锚点、不填充缺失。

测量mean/median/p90 episode seconds与初始化成本，预测全Stage1及Stage2 CPU小时和18/24核墙钟时间；不据outcome改变state/η集合。Stage2全量请求61440，含832个R5短episode最多62272<65000；若确实因已冻结计数上限无法实施，唯一减配为S_train96→64。时间成本另报告，不擅自凭任意耗时阈值减配。

## Pilot门及统计
对三个**内部λ**分别报告K4成功上下界、observed-valid成功率、η13−η10配对保守上下界、state bootstrap5000次CI。pair合格需要两个内部λ分别由η13/η10占优（保守差上下界不含0），且两侧赢家的成功下界≥.4。选内部λ是因为Stage2正式源/留出controller正是12个内部λ；仅原始端点翻转而内部controller全部同regime，不提供本次要检验的覆盖变化。另完整描述λ0/1原标签可用曲线，不能将缺失覆盖当完整24state。至少3/4pair合格才进Stage2。CI不另增加显著性门，也不宣称翻转由独立数据确认。若unknown使门不确定，保守视为未通过，不删unknown、不扩大seed。

门失败则Stage2 NOT EXECUTED — GATE FAILED，报告regime地图并结束；门通过才冻结Stage2附加协议、划分、R5、模型/统计细节并提交。不得删K16强η或选择有利state。所有round4state结果为DEVELOPMENT。

## 资源与证据
CPU最多24同时任务，GPU最多8shards，2026-10-14T08:00:00+08:00后CPU18/GPU4；GPU任务各占1CPU。extra作业Slurm时限不得跨截止。worker raw/journal、experiment先登记、单进程merge、postflight；纯Q不要求视频；只写新目录、不commit、不覆盖历史证据。
