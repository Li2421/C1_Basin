# Round17 新配方 critic 前瞻确认：预注册

日期：2026-10-14（Asia/Singapore）。本文及其 SHA256 收据写在任何本轮新成功标签产生之前。全部新文件仅写入本目录；历史代码、checkpoint、数据、DB 证据只读，除依共享 DB 标准流程登记本实验新证据外不改历史资产。不提交 Git。

## 问题、证据及隔离

主问题是在新训练配方得到的 17 个 controller 和 48 个新 Ring state 上，冻结的 round16 A3 能否零样本选择。N1 为新数据集 `base_u_v7_local_plus_drift_dataset` 上、原 v7_local config 的四个 seed（3124+40001..40004）；若该数据集 manifest 或任何列举文件不完整，N1 标记 UNAVAILABLE，不替换。N2 为其余 13 个原配方，原 config 只把 actor_hidden_dims 改为 (512,512,512,512)，seed=原 seed+40001。N2 仅是新架构配方，仍与训练配方共享数据集。新颖性核对参数 hash 与全部可读历史 checkpoint；同一 adapter 的 FP32 加载、确定性前向及 round15 兼容性条件均须通过。

历史证据先审计：round16 的 A0/A3 三 fit checkpoint、标准化/PCA 及 G_E 已冻结，可直接推理，不重训；round15 的八动作 Q4 和 T1/T2/T3 的已有标签可用于 A3-all，旧 controller 的 rollout 不重跑。新 controller checkpoint、R5 探测、48 个新 state 及新 controller×state 的 Q4 标签是缺口；仅该缺口按 DB preflight 后补跑。round5/6 固定 η K16 与本轮时间表动作语义不同，不复用为八动作标签。换 critic 编码不废止语义相同的旧 rollout。

## 时间顺序及模型

1. 核对 `git_43b5264` 包字节与 round14A/15 收据，核对 N1 全部文件完整性与文件 SHA256；生成并冻结训练计划。训练17个 Flow checkpoint，GPU、JAX x64=false、FP32。N1 v7_local config 原值：100000 步、256×3；N2 原 config 所有设置不变，唯 actor_hidden_dims=(512,512,512,512)。所有新 checkpoint 和训练日志落本目录。
2. 在**任何新成功标签之前**完成 checkpoint 新颖性/兼容性审计。R5 用 round15 代码和冻结的16个 probe state×η10/η13×200步无标签探测，得到160维；不读取任务终点成功。使用 round16 `fits/full_transform.npz` 的 R5 缺失填补、列选择、标准化和 PCA2，不重拟合。
3. 为使 controller 级选择有明确评估 state，先依 round10 采样流程生成48个**无标签**新 state，抽样种子 2026172400..2026172447，全部与 DB 可恢复物理状态、历史协议及本批相互去重；若冲突，按整数种子递增替补，记录每次尝试。其后才生成任何新成功标签。
4. A3（主）与 A0 使用 round16 各自三个 fit seed17/23/41的全源 checkpoint，原样及原特征变换。用48个评估 state 的 physical23 推理各动作概率。主规则对每 controller 先在48 state 上平均三个 fit 概率，再对八动作取 argmax，形成固定动作；次规则对每 state 取 argmax。A0 同样两种规则。均值并列按 F10,F13,F2,S200,S300,S400,S500,S600 顺序。G_E 使用 round16 的56源 controller 冻结 Ê ridge（`predictions/G_E_FROZEN_CHOICES.json` 记录系数、均值及阈值0.05），投影新 R5 后选择 F10 或 S500。源平均动作使用 round16 的源标签平均最优动作，固定不随新标签更新。
5. A3-all 为次要比较：在 round15 的56、T1的14、T2的14、T3的8，共92个旧 controller 的所有可用八动作或子集标签上，从头训练三个 fit。完全采用 round16 A3 的64/64/16 SiLU scalar-logit、[physical23,R5-PCA2,one-hot]、源训练 controller 的 R5 缺失填补/标准化/PCA2、AdamW lr0.0008、weight decay0.0001、梯度 clip5、batch128、fit seeds17/23/41、JAX x64=false、FP32。unknown 排除，不可用 arm 排除；均匀抽样有效 controller×state×arm cell，按每 cell 的有效 seed 数/4 加权 BCE。使用 round16 已冻结的 A3 selected step=1000，不看本轮标签调步数。A3-all 的选择同 A3 两规则。
6. 冻结 R5、48 state 身份、A3/A0/G_E/源均值/A3-all 的逐 controller 主动作、逐 state 次动作、A3-all checkpoint、全部三 fit 预测、代码和阈值，写冻结收据并校验。**此时之前不得运行新成功标签或一致性门。**

## 标签采集与判定

冻结后，用 round15 的原始 schedule 执行代码语义（t<t_switch用η13，此后η10；固定动作按原η）生成 17×48×8×4=26,112 个 Q4 case，外加32例历史一致性门。按 shared_rollout_db 标准：manifest → preflight → 只跑 genuinely_missing → append-only journal → 单进程 merger → postflight。EXACT_REUSE须完整语义指纹匹配；AMBIGUOUS不复用；unknown 保留且不重跑，不算失败；timeout 不称 deadlock。一致性门要求旧证据对应的 outcome、episode length 与首动作字节完全一致；门失败停止主 batch。源 N1 不可用时，只运行 N2 的13×48×8×4并明确缺口。

主要分析群体为 N2 的13个新架构 controller；N1 的4个新数据集 controller 单列描述，因为样本量小。每个 controller 每动作成功率为 success/known，unknown 不入分子或分母；若 controller×动作已知数为0，则该 controller 的遗憾与相关分析标记 INCONCLUSIVE，不用其他动作替代。Oracle 为同一 controller 八动作中已知成功率最高者；所有策略按实际冻结选择的动作计值。遗憾为 oracle−策略成功率，单位pp；最大遗憾为 N2 controller 最大值。真实 E=S500−F10，L=S500−F13；与 A3/A3-all 推出值作 Pearson 与 Spearman，常数向量记未定义。

K1：N2 A3 controller 固定规则减 A0 controller 固定规则的**平均遗憾差**；从48个 state 同步有放回抽样10000次（seed1717），每次对同一批 state 重算各 controller 的动作成功率、oracle 和平均遗憾，取配对百分位95% CI；上界严格<0 为 SUPPORTED，否则 REJECTED，缺标签导致无法计算为 INCONCLUSIVE。K2：N2 A3 平均遗憾≤G_E平均遗憾+0.5pp且最大遗憾≤5pp 才 SUPPORTED，否则 REJECTED。K3：N2 任何新 controller 的 A3 固定动作成功率都不得低于 F10 超过10pp；达到或超过10pp的劣势则 REJECTED。另报告 N1 同样数值但不作主门槛；A3逐 state vs固定、A3-all vs A3、每 controller 选择及其已知 seed 数、unknown 总数与按动作分布。逐 state 的动作对各 state 单独计结果；固定动作始终用冻结的48 state 选定，不因 bootstrap 重选。

若 K1/K2/K3 全通过，判为新配方前瞻确认；K1通过而K2失败，则只确认 controller 依赖、结构化 G_E 更好；K1失败则本轮未确认新配方上的 controller 输入选择收益。新 controller+新 state 的结果称前瞻；N2 不称完全独立数据配方。Q 预测质量和 E/L 相关是描述，不代替选择门槛。

## 交付与资源

本轮纯 Q 数据生成和 critic 评估按 AGENTS.md 视频例外不运行视频入口。交付预注册/收据、训练与数据审计、17 checkpoint、R5及冻结收据、选择、state去重、一致性门、DB pre/postflight、results、三类图、REPORT、HANDOFF。资源在每次提交前按 squeue 核对全部用户任务，合计≤18 CPU shard、≤2 GPU shard；不将 Slurm partition 的物理 GPU 数误当作 shard 授权。
