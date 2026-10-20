# Round18 Ring η2 缺口：预注册

日期：2026-10-14（Asia/Singapore）。本文件与 UTC/SHA256 收据先于本轮训练、探测和新成功标签。只在本目录写文件，不修改或删除历史资产，不做 git commit。本轮为纯 Q 数据生成、训练和评估，适用 AGENTS.md 视频例外，不运行视频入口。

## 证据审计与目的

先复用 Round15 的 56 个 controller 的 Q4 标签、R5 特征，Round17 的 17 个 controller 的 Q4 标签、R5 特征，以及 Round16 的 A3 三 fit 与预处理作为 C0。按 Round15 `dataset.jsonl` 重算，56 个 controller 中只有 2 个 F2 最优，领先最佳非 F2 分别为 0、0.595pp；Round17 `results.json` 中 v3 宽网络版 F2 约95.72%，A3 选 S300，遗憾6.36pp。旧证据不能回答新 η2 探测和新 v3 家族的因果析因检验。更换 critic 输入不使 execution semantics 相同的旧标签失效；旧 controller 标签不重跑。所有 rollout 先按 shared_rollout_db 标准 design/preflight，只跑 genuinely_missing，journal，单进程 merge，postflight。unknown 保留、不重跑、不当失败；timeout 不称 deadlock。

## A. checkpoint 与固定划分

训练包 `git_43b5264` 的18个文件先按 Round15 历史审计逐字节核对，GPU，JAX x64=false，参数 FP32；用 Round14A/Round17 的兼容性、确定性前向和 checkpoint 参数哈希门。v3 与 plus_w_v3 两配方分别用 actor_hidden_dims 256×3、384×3、512×4，原 seed +50001..+50004，共24个。其余12个原配方用384×3、原 seed+50001，共12个。除架构与 seed 外保持原 config、数据、训练步数、batch及归一化语义；宽网采用 Round14A 的 v4_wide 兼容修复。每个新参数哈希需与可读的所有历史 checkpoint 及本批其他 checkpoint 不同。

TRAIN-new 为 v3 家族每个配方×架构的 +50001/+50002（12个），及非 v3 配方名字典序奇数位（1,3,...,11，共6个）。TEST 为 v3 家族 +50003/+50004（12个）及非 v3 偶数位（6个）。名字典序使用原 run 名称；该划分在任何新标签前冻结。两个 v3 组共享配方与架构，TEST 只有训练 seed 不同，不能称独立配方。

## B. 无标签探测与冻结

探测对象为 Round15 56 + Round17 17 + Round18 36，共109 controller。旧 R5 直接复用；新36个 R5 用 Round15 冻结代码，在16个固定 probe state×η10/η13×200步运行。η2 探测用同一16 state、相同 seed 与冻结的5种过程统计量，对109个 controller 各运行16次×200步；不读取终止成功。数值失败按 Round15/7 同类缺失规则记 NaN，不重跑。R5 原始160维删除冻结的5个无信息列得155维，并应用 Round16 已冻结的训练集缺失填补、z-score 与 PCA2，得到 C0/C1 的 R5-PCA2。

新表示 R5η2 为上述155维与η2探测的16×5=80维拼接，共235维。训练 controller 仅为 Round15 56 + Round17 17 + TRAIN-new 18（91个）；TEST18绝不参与新表示的均值、尺度、缺失填补或PCA拟合。每列用91个训练 controller 的有限值均值填补；全缺失列舍弃；标准差<1e-6则取1；对 z-score 矩阵做 SVD，取前2主成分，按绝对值最大载荷为正定向。C2/C3共用这套仅由训练 controller 无标签探测拟合的变换，以隔离标签覆盖效应。特征、变换、所有新 state 身份与探测代码在任何本轮新成功标签之前冻结并留哈希收据。

## C. TRAIN-new 标签与一致性门

按 Round10 sampler 以整数种子2026181000起抽32个真正 t0 新 state；另以2026182000起抽48个TEST新 state。按 DB 中所有可恢复物理状态、历史协议及本批 state 去重，撞库则递增种子替补并留审计。两组互不重叠，统一 H700、dt=.05、FF、原安全投影及原成功判定。动作F10/F13/F2/S200/S300/S400/S500/S600；S-t 在步骤索引<t 用η13，此后η10，只改变η。每个 TRAIN-new controller×32 state×8动作×seed0..3，共18,432条；先运行32例字节级一致性门，要求历史同例 outcome、episode length、首动作 float64 字节相同。若门失败，停止新标签。TRAIN-new 可用于模型拟合；在 critic 与 TEST 选择冻结之前不得生成或读取任何 TEST 标签。

## D. 模型、验证、选择与冻结

C0 是 Round16 冻结的 A3（三 fit）及其原R5变换，不重训。C1用Round15+Round17+TRAIN-new标签与原R5变换，只增加覆盖。C3用Round15+Round17标签与新R5η2-PCA2变换，只改表示。C2（主）用Round15+Round17+TRAIN-new标签与新R5η2-PCA2。A0用与C2相同标签，不输入controller特征。所有新模型采用Round16 A3的64/64/16 SiLU MLP、one-hot动作、AdamW lr0.0008/weight_decay0.0001、clip5、batch128、fit seed17/23/41、最多12000步、JAX x64=false。标签cell中unknown排除；四seed的已知成功/已知数按Round16的加权BCE，不可用动作排除。

内部验证按原14配方加N1新数据配方的根配方分组，组内所有架构/seed留在同一折；按排序后的根配方轮转分成5折。对各模型分别训练5折×3 fit，500,1000,...,12000步评估 held-out 已知标签 NLL，选择跨折和fit平均NLL最低的步数，并列取较早步数。变换使用上一节的冻结无标签变换；state23标准化仅用该模型训练标签集的训练state（在折内仅用训练折相关state，跨源共享state时按训练折可用state）。选完step后用该模型全部规定训练标签重训3 fit，不看TEST标签。所有超参数、折法、输入、动作、阈值、TEST state及其无标签预测，在TEST任何成功标签前冻结并留收据。

TEST选择按Round17主规则：对每模型的3 fit预测先求平均，再在48个TEST state上平均，按F10,F13,F2,S200,S300,S400,S500,S600次序argmax，逐controller固定动作；逐state argmax只作次要分析。同一测试state对所有模型相同。冻结每个TEST controller×模型主/次动作、预测E/L、所有checkpoint与预处理哈希。

## E. TEST标签与分析

只在第二次冻结后运行18 TEST controller×48新state×8动作×4seed=27,648条。先DB preflight，只跑缺失；journal、单进程merge、postflight。成功率=成功/已知seed，unknown不算成功也不算失败；若某controller×动作没有已知值，相关门记INCONCLUSIVE。oracle为八动作中成功率最高者；遗憾=oracle−冻结策略成功率，单位pp。

η2型定义：TEST中 F2 成功率至少比最佳非F2高3pp。headroom为至少3个η2型；不足则B1=INCONCLUSIVE，其余照常。CI按48个TEST state成簇同步有放回抽样10000次，seed1818，配对百分位95%CI；固定动作不随重采样重新选择。B1：η2型上C2−C0平均遗憾差CI上界严格<0且C2最大遗憾≤5pp，headroom不足优先INCONCLUSIVE。B2：全18个C2平均遗憾≤2pp、最大≤5pp、无任一比F10低至少10pp。B3：非η2型C2平均遗憾≤C0+0.5pp。B4：全18个C2−A0平均遗憾差CI上界严格<0。析因描述性比较C1−C0、C3−C0、C2−C1、C2−C3及其配对CI；另报v3 η2出现率按架构/seed、各模型对η2型所选动作、η2探测特征与F2−最佳非F2的Pearson/Spearman。任何TEST标签都不得用于重新拟合、选择step、改阈值或改动作。

若四门全过，称η2缺口补上；B1过但B2未过则指出其他尾部controller；headroom足而B1未过则报告两种补救不足及析因方向；headroom不足则说明罕见性。TRAIN-new数值是开发数据，TEST为新seed与新state的前瞻证据，不称新配方独立。资源提交前用squeue核对用户所有任务，合计≤18 CPU shard/2 GPU shard。
