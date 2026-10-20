# Agent A — 实际 Toy 几何与 K 边界

结论标记：以下有限墙段自由空间恒等式及其正宽子域为 **PROVEN**（几何解析证明，常量从实际 checkpoint 运行对象提取）；固定 TT/FF 的 Yield、Pass 与任务成功均 **UNKNOWN**，本报告不进行相关推断。没有调用投影、env.step、训练或 rollout。

## 历史证据、版本与运行配置

先读取并复用 `diagnostics/tt_ff_correction_realization_20261017_v1/{geometry_data.json,geometry_certification.md,pre_evaluation_audit.json}`。旧证据已包含精确墙段、checkpoint 参数和几何邻域证书；本次缺口是完整自由空间截面、K 的非独立坐标边界与有限尺寸子域。因此只做新几何算术和 checkpoint 重新加载，不重复行动或轨迹实验。

冻结根 F=`/home/zhihan/research/Basin_C1_flow_field_poc_20261004`。`validation_v2/adapters.py:53–59` 的 `OtherRuntime('toy_give_way')` 真实调用 `load_policy` 并使用 checkpoint `si_metadata.evaluation_environment` 构造环境。checkpoint=`F/validation_v2/assets/toy_ckpt.pkl`，SHA256=`8c17b5d2cf74c8ce94ffa03a192a8c36ae01227465a168e7b74d6e18ff121f32`。本次脚本重新走此加载链，复核旧审计的全部源文件和 checkpoint 哈希完全一致；各绝对路径与哈希保存在 `geometry_contract.json`。

源码 `environment.Config` 默认半长 **2.5 m**，实际 checkpoint **1.3 m**。其余当前默认环境字段与实际评估字段一致。墙坐标必须由实际1.3计算，不能套默认2.5。CBFConfig 为当前冻结源实际默认值。CPU加载过程中本机 CUDA 插件发出初始化警告，但脚本使用 JAX_PLATFORMS=cpu，完成运行；无模型推理需求。

| 参数 | 实际值 | 源码位置（F 下） |
|---|---:|---|
| `wall_layout_scale` | 1 | environment.py:20,123；adapters.py:57 |
| corridor half-length L | 1.3 m | environment.py:19,116；checkpoint实际覆盖默认 |
| corridor half-width w | 0.2 m | environment.py:17,114 |
| bay left/right/top | -0.2 / 0.2 / 0.68 m | environment.py:18,118–123 |
| robot radius rb | 0.16 m | environment.py:16 |
| wall radius rw | 4/600 = 0.006666666666666667 m | environment.py:22 |
| wall collision margin | 0.005 m | environment.py:23 |
| pair collision margin | 0 m | environment.py:24 |
| CBF buffer beta | 0.0001 m | cbf.py:20–22 |
| Rw=rb+rw+mwall+beta | 0.17176666666666665 m | cbf.py:70–71 |
| Ds=2rb+mpair+beta | 0.3201 m | cbf.py:74 |
| V, dt | 0.5 m/s, 0.05 s | environment.py:13,15 |
| gamma, gamma_wall | 1 /s, 1 /s | cbf.py:17–18 |
| horizon | 850 steps = 42.5 s | environment.py:14 |

数值 feasibility_tol=1e-9、speed_tol=1e-10、optimality_tol=2e-6；这些不能再加进物理半径。Toy投影是 cbf.py 的 SLSQP+外切速度球 cutting plane，失败按冻结 retry 链处理；不是直接把通用 hard_projection.py Clarabel结果当Toy。此报告不使用任意后端的行动数值。

全部8墙段，顺序即实际17行CBF中的墙序号：

| 名称 | 起点 m | 终点 m |
|---|---|---|
| left_end | (-1.3,-0.2) | (-1.3,0.2) |
| right_end | (1.3,-0.2) | (1.3,0.2) |
| floor | (-1.3,-0.2) | (1.3,-0.2) |
| left_ceiling | (-1.3,0.2) | (-0.2,0.2) |
| right_ceiling | (0.2,0.2) | (1.3,0.2) |
| bay_left | (-0.2,0.2) | (-0.2,0.68) |
| bay_right | (0.2,0.2) | (0.2,0.68) |
| bay_top | (-0.2,0.68) | (0.2,0.68) |

## 完整中心自由空间，保留口部圆弧分支

定义 a=L-Rw=1.1282333333333334 m，h=w-Rw=0.02823333333333336 m，H=0.68-Rw=0.5082333333333334 m。必须限制在环境 corridor∪bay 的内部连通分支（`environment.py:135–139`）；单独墙距不等式也允许环境外部，不可拿来扩大空间。

在此分支，单机器人安全中心集合恰为

\[
\Omega=\{(x,y):-h\le y\le H,\ |x|\le g(y)\},\quad
g(y)=\begin{cases}
a,&-h\le y\le h,\\
w-\sqrt{R_w^2-(w-y)^2},&h<y<w,\\
h,&w\le y\le H.
\end{cases}
\]

**PROVEN**：下段 floor、end walls 和 ceiling 距离给出第一段；口部中段两条 ceiling 端点 `(±w,w)` 和相接 bay_side 的最近点相同，形成两个圆形排除区，准确给出第二段；上段 bay侧墙内点以及top给出第三段。其余每一墙段均至少Rw：下段对口角的垂距不小于Rw；中段floor距离≥w+h、top距离≥.48、end距离远大于Rw；上段ceiling最近点在口角且水平距≥Rw、floor和end更远。截面在w处连续；**h处不连续**：y=h整条主通道顶安全线|x|≤a合法，而y>h只允许口部圆弧内的|x|≤g(y)，右极限为w。此水平线是二维面积为零的合法边界分支，不能用中段公式覆盖；几何可行集依然闭合。不把端点替换成无限直线：例如 y=.1 时允许宽度是 `2*(.2-sqrt(Rw²-.1²))`，明显不同于bay上段宽度。

两机器人完整几何集合是

\[
\mathcal F=\{(p_1,p_2)\in\Omega^2:s=x_1-x_2,\ z=y_2-y_1,\ s^2+z^2\ge D_s^2\}.
\]

最近点原实现为 `t=clip((p-a)·(b-a)/||b-a||²,0,1)`；见 environment.py:74–78、cbf.py:62–75。墙CBF为线性距离，pair CBF为**平方距离** `s²+z²-Ds²`，不能混淆。

| 变量 | 单独极值 m | 适用范围/不可省略的耦合 |
|---|---:|---|
| x1,x2 | [-1.1282333333333334,1.1282333333333334] | 各自|xi|≤g(yi) |
| y1,y2 | [-.02823333333333336,.5082333333333334] | 各自|xi|≤g(yi) |
| 理论 lower agent y1 | [-.02823333333333336,0] | 额外选择lower角色；不是环境限制 |
| yielding agent y2 | [0,.5082333333333334] | 额外选择upper角色；低于h时未必在bay |
| bay内x2 | [-.02823333333333336,.02823333333333336] | 仅y2≥.2的完整截面；口部实际更宽 |
| z，当y1≤0≤y2 | [0,.5364666666666668] | pair要求s²+z²≥Ds² |
| s，无角色限制 | [-2.256466666666667,2.256466666666667] | 不是独立box；由x1,x2,y1,y2共同限制 |

整个闭安全集合的最小墙CBF净空为0，在floor y=-h、bay side x=±h/y≥.2、top y=H等安全边界取得；相对评估器的碰撞阈值仍多0.0001 m。正净空必须主动选择严格内子域。

## K 的边界哪些可决定、哪些不可决定

原形状可以保留为**几何容器**而非控制保证：`-h≤y1≤0, 0≤y2≤H, b-≤x2≤b+, -S≤s≤D_exit`，加全体`p_i∈Ω`和pair耦合。其中h、H是上述角色分支的真实几何极值；若要求x2整条垂直路线直到高bay，最大的公共b是±h。S和D_exit**不能**由墙面唯一推导。它们是任务阶段、入口位置、退出标准的选择。

若把完整x2∈[-h,h]和s∈[-S,D_exit]强行取成独立box，为使x1=x2+s始终不碰端墙，须S≤a-h=1.1和D_exit≤1.1 m；允许耦合时可达a+h=1.1564667的个别相对位置，不可作为统一box边界。取D_exit≥Ds（例如Ds或.34 m）足以使随后横向回归时水平距保持Ds；这是保守退出选择，并非通过所必需的最小正s。

## 有限尺寸、可追溯子区域

以下只证明几何，**不证明控制实际进入**。

1. `K_entry_box`: x1∈[-.55,-.40], x2∈[-.02,.02], y1∈[-.01,0], y2∈[0,.01]。体积6e-7 m⁴；全域wall h≥.19-Rw=.01823333333333335 m（floor，在y1=-.01取得）；pair center distance≥.38 m（x1=-.40,x2=-.02,y1=y2=0）。可包含零初始横向偏移，纵向宽.15 m、入口x2宽.04 m不是无限小邻域。

2. `K_pass_box`: x1∈[-.55,.55], x2∈[-.02,.02], y1∈[-.01,0], y2∈[.38,.42]。体积1.76e-5 m⁴；wall h≥.18-Rw=.00823333333333336 m（bay侧墙x2=±.02）；pair distance≥.38 m（s=0,y1=0,y2=.38）。其整个水平通过带是几何可行。

3. 更宽的可选积分域，供B/C/D独立评估：x2∈[-.025,.025], y1∈[-.02,0], y2∈[.38,.46], x1∈[-.65,x2+.34]，D_exit=.34。体积7.92e-5 m⁴（积分x1长度.99+x2）；wall h≥.175-Rw=.00323333333333335 m（bay侧墙），pair distance≥.38 m。严格内点均有正体积。该选择牺牲侧墙净空以增加覆盖；不能由几何净空直接推出大通过速度。

**PROVEN** 正宽几何入口、口部通道、高bay和通过域存在。沿x2∈[-.02,.02]从y2=0升高到.4的口角最小墙距为.18 m，因此入口到bay几何连通；若p1保持左侧x1≤-.4，pair水平差≥.38也始终安全。但竖直速度仍受口角CBF限制，须由B计算；不把连续路径直接当TT/FF可实现路径。

## 采样、目标及终止

`environment.py:147–155` 默认reset是p1=(-.85,0),p2=(.85,0)，并检验实际碰撞净空。它本身**不是随机采样器**。`source/scripts/giveway_initial_state.py:4–14` 基础分布为x1≈[-.88,-.82]、x2≈[.82,.88]、yi≈[-.002,.002]，用float32存储，边缘有浮点舍入。

困难面板来自 `/home/zhihan/research/Basin_C1_hard_field_20261006/design.py:29–40,91–94`：先基础采样，再把两个x同乘统一随机scale，tier0∈[.88,.94]、tier1∈[.72,.80]；最后yi加独立[-.018,.018]。故名义包络tier0 |xi|∈[.7216,.8272]、tier1∈[.5904,.704]、yi∈[-.020,.020]，但所有坐标相关，不能视为独立均匀box；具体冻结confirmation状态优先于包络。

目标由 environment.py:124 得(1.09,0),(-1.09,0)。两目标误差同时≤.08才success（:192–194）。墙/机器人碰撞含整个物理步的swept几何（:183–191）；零速不自动deadlock。deadlock需2 s窗口中两机器人目标距离的净变化绝对值均<.01 m，且瞬时最大速度<.025 m/s，并连续保持候选5 s（:196–219）。终止优先级collision→success→deadlock→850步timeout（:223–228）；timeout不是deadlock。

## 复现与成本

从 `/home/zhihan/research/Basin_C1` 执行：

```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/toy_K_quantitative_certification_20261017_v1/agent_a/certify_geometry.py
```

新运行为一次CPU checkpoint对象加载及常量算术；模块导入之外计时0.823 s，进程总约1.47 s。新增投影0、rollout0、物理步0、训练0、GPU0。输出同目录 `geometry_contract.json`；完整源哈希也在其中。子域与自由空间推导是全域解析界，未将有限采样称为完整区域证明。
