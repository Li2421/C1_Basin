# Critic 是否真正学到 state–eta interaction？

**结论：Ring / Four-Way 目前不能宣称已可靠学到“随 state 正确改变 eta 相对排序”。但也不能把联合 critic 简单等同于完全不看 state 的网络。**

本轮检验的是冻结模型的函数行为，不使用 raw weight magnitude 作为证据。第一阶段不训练；第二阶段只拟合隔离的诊断 eta-only / additive 基线，未重训练任何现有 critic、generator 或控制器。新增 rollout 仅供诊断，不进入训练。

## 1. 资产、分层与定义

读取当前本地 LOSO root-cause 完成资产，而不是只沿用较早 prose。A=single_scene supervised；B=joint_enriched supervised（并保留 joint_generic 控制）；C=source-only LOSO。每个模型选用此前 source/validation 冻结的 checkpoint，不按本轮结果选模型。四个冻结候选集是 Ring60、Four24、Double24、Toy48，均为已归档的16个随机候选，排除 mean；无重新抽样。模型/数据位置见 ALREADY_TESTED.md、provenance.json。

定义 Δ=[Q(a,i)−Q(a,j)]−[Q(b,i)−Q(b,j)]。A(h)+B(eta) 在概率空间的 Δ 恒为0；但 sigmoid(A(h)+B(eta)) 的概率 Δ 可以非0，且仍不能翻转 eta 排序。因此同时报告概率双差分、logit 非可加性，以及真正反向排序。|Δ|>0 本身不等价于发生 ranking reversal。

预注册：20个场景内独立 derangement seeds；强交互 |Δ_true|≥0.5；清晰符号比较 |Δ_true|≥0.125；反转要求两边 Q gap 均≥0.25 且方向相反。真 Q 使用完整证据，数值失败不填0。所有 quadruples 共享 state/eta，不能当独立数千次统计试验。

## 2. State shuffle：保留原 state 的候选与真 Q

| 场景 | 模型 | 原 B15 / 未认证 | shuffle B15均值 [min,max] | top1一致率 | score相关 | 原→shuffle Q下界 | severe FP均值 |
|---|---|---:|---:|---:|---:|---:|---:|
| ring | scene_specific | 47 / 0 | 46.35 [45,48] | 0.966 | 0.840 | 0.833→0.831 | 0.00 |
| ring | joint_supervised | 58 / 0 | 57.15 [56,58] | 0.803 | 0.981 | 0.977→0.975 | 1.30 |
| ring | joint_generic | 11 / 0 | 7.60 [6,10] | 0.696 | 0.579 | 0.242→0.172 | 49.50 |
| ring | LOSO | 0 / 0 | 0.00 [0,0] | 0.978 | 0.987 | 0.014→0.014 | 59.65 |
| four | scene_specific | 22 / 2 | 22.00 [22,22] | 1.000 | 0.428 | 0.987→0.987 | 0.00 |
| four | joint_supervised | 19 / 5 | 19.80 [17,21] | 0.523 | 0.740 | 0.961→0.966 | 0.00 |
| four | joint_generic | 22 / 2 | 23.20 [21,24] | 0.119 | 0.097 | 0.982→0.990 | 0.00 |
| four | LOSO | 21 / 3 | 20.70 [19,23] | 0.283 | 0.275 | 0.974→0.955 | 0.60 |
| db | scene_specific | 24 / 0 | 24.00 [24,24] | 0.973 | 0.935 | 1.000→1.000 | 0.00 |
| db | joint_supervised | 24 / 0 | 24.00 [24,24] | 0.940 | 0.960 | 1.000→1.000 | 0.00 |
| db | joint_generic | 24 / 0 | 24.00 [24,24] | 0.927 | 0.922 | 1.000→1.000 | 0.00 |
| db | LOSO | 20 / 0 | 20.25 [19,22] | 0.556 | -0.103 | 0.922→0.907 | 0.00 |
| toy | scene_specific | 45 / 0 | 34.60 [31,38] | 0.218 | 0.169 | 0.978→0.776 | 6.60 |
| toy | joint_supervised | 44 / 0 | 35.25 [31,40] | 0.255 | 0.205 | 0.970→0.795 | 7.55 |
| toy | joint_generic | 45 / 0 | 35.60 [30,41] | 0.224 | 0.227 | 0.966→0.801 | 6.30 |
| toy | LOSO | 30 / 0 | 30.05 [29,31] | 0.970 | 0.971 | 0.689→0.688 | 0.00 |

Ring enriched 58/60→57.15/60，说明原 K16 部署中的大部分好选择不需要正确匹配具体 state；但其 top1 约80%一致，仍存在 state 影响。Toy44/48→35.25/48 是检验有效的正控制。Four 的数值未认证不能当失败，不能仅比较已认证 B15 数量就断言某选择更差。

## 3. Eta shuffle / joint shuffle

| 场景 | enriched 控制 | score MAE变化 | score相关 | 相同候选位置 top1 |
|---|---|---:|---:|---:|
| ring | state_shuffle | 0.0378 | 0.981 | 0.803 |
| ring | eta_shuffle | 0.4545 | 0.027 | 0.073 |
| ring | joint_shuffle | 0.4578 | 0.015 | 0.066 |
| four | state_shuffle | 0.0020 | 0.740 | 0.523 |
| four | eta_shuffle | 0.0160 | 0.097 | 0.102 |
| four | joint_shuffle | 0.0161 | 0.009 | 0.075 |
| db | state_shuffle | 0.0102 | 0.960 | 0.940 |
| db | eta_shuffle | 0.0779 | 0.058 | 0.050 |
| db | joint_shuffle | 0.0772 | 0.026 | 0.058 |
| toy | state_shuffle | 0.3046 | 0.205 | 0.255 |
| toy | eta_shuffle | 0.3611 | 0.235 | 0.083 |
| toy | joint_shuffle | 0.3849 | -0.034 | 0.085 |

Eta/joint shuffle 的 top1 是候选位置比较，不是相同 eta 身份。recipient state 上的 donor eta 没有对应真 Q 时，B15/Q/regret 均为 N/A；绝不挪用 donor state 的标签。详见逐 seed 的 shuffle_metrics.json。

## 4. 共享 eta 直接 interaction 测试

已有 VAL 缓存中 Ring 仅1个、Four仅30个完整四元组，且没有强交互；不能把 state-specific K16 中相近但不同的 eta 当同一 eta。于是按固定 hash 从各场景既有独立确认队列取8个状态，使用共同8个预注册 Sobol eta及 eta0 参考；模型预测在新结果前冻结。该队列已被查看过，属于诊断，不是新的 untouched generalization 声明。

| 场景 | 模型 | 完整四元组 | Δ相关 | 清晰符号准确率 | 强交互数 / 方向准确率 | 清晰反转数 |
|---|---|---:|---:|---:|---:|---:|
| ring | scene_specific | 759 | -0.065 | 0.473 | 164 / 0.421 | 0 |
| ring | joint_supervised | 759 | 0.089 | 0.584 | 164 / 0.683 | 0 |
| ring | joint_generic | 759 | -0.022 | 0.481 | 164 / 0.585 | 0 |
| ring | LOSO | 759 | -0.168 | 0.424 | 164 / 0.451 | 0 |
| four | scene_specific | 735 | -0.126 | 0.386 | 0 / N/A | 0 |
| four | joint_supervised | 735 | -0.042 | 0.410 | 0 / N/A | 0 |
| four | joint_generic | 735 | 0.213 | 0.643 | 0 / N/A | 0 |
| four | LOSO | 735 | -0.013 | 0.467 | 0 / N/A | 0 |

上表包含eta=0诊断参考；它不是部署候选，也不在非零eta冻结域内。必须另看仅8个域内eta：Ring enriched Δ相关0.078、强子集117个、符号准确69.2%；Four enriched Δ相关0.163、符号准确49.0%，仍无强交互。详见 full_domain_only 指标，不能把域外参考的误差完全归咎于部署分布。

Ring 真 Δ 范围约[-1.0625,1.0625]，enriched 预测仅约[-0.153,0.231]：明显低估状态相关的差值幅度。含参考时整体相关0.089，强子集相关0.262、符号准确68.3%，属于弱信号，不能包装成可靠 ranking learning。该68.3%的state-cluster区间约47.5%–85.1%，包含50%，且尚未计入rollout seed噪声。Four没有预注册强交互样本，含参考时enriched 双差分相关约−0.042。

两主场景面板均无达到预注册幅度的真 ranking reversal。反转 recall/direction accuracy 是 N/A；“全部预测无反转”所得接近100% detection accuracy 不构成正证据。完整 Δ 分布及强子集指标在 probe_interaction_metrics.json；按8个 state 重采样的不确定区间在 state_cluster_uncertainty.json，尚未计入 seed 估计误差。没有为了找出漂亮反转而自适应扩大全网格。

Toy 复用验证证据含252个经验反转，enriched正确反转方向约91.3%，Δ相关0.754；LOSO Toy正确反转方向为0。这说明同类共享编码/critic在有相关监督时能够表达交互，但该能力没有被证明可迁移。Toy标签含不同seed预算，其结果是有限经验Q对照，并非每个格子均Q16认证。Double已有8072个四元组、仅2个清晰反转，enriched两者均未识别。

## 5. Eta-only、additive 与 full 基线

相同各模型 TRAIN/验证家族划分和Q标签：保留已有eta-only MLP；另外用固定128维eta Fourier特征、固定ridge .01拟合 eta-only 和 A(h)+B(eta)，其中A使用同一冻结critic的h。只用TRAIN拟合，不选超参数。后者不是重新训练的同容量神经网络，不能用其欠拟合单独证明full学会interaction；优势证据必须结合直接双差分/反转。additive预测不截断后选argmax，避免制造clipping ties；仅概率误差计算clip。

| 场景 | enriched对照 | B15 | 未认证 | 完整Q MSE |
|---|---|---:|---:|---:|
| ring | eta_only_RFF | 49 | 0 | 0.3283 |
| ring | additive_RFF | 46 | 0 | 0.1807 |
| ring | frozen_eta_only_MLP | 55 | 0 | 0.2441 |
| ring | frozen_full | 58 | 0 | 0.0255 |
| four | eta_only_RFF | 23 | 1 | 0.0207 |
| four | additive_RFF | 21 | 3 | 0.0115 |
| four | frozen_eta_only_MLP | 23 | 1 | 0.0460 |
| four | frozen_full | 19 | 5 | 0.0011 |
| db | eta_only_RFF | 23 | 0 | 0.0243 |
| db | additive_RFF | 20 | 0 | 0.0238 |
| db | frozen_eta_only_MLP | 24 | 0 | 0.0185 |
| db | frozen_full | 24 | 0 | 0.0034 |
| toy | eta_only_RFF | 44 | 0 | 0.1352 |
| toy | additive_RFF | 36 | 0 | 0.1765 |
| toy | frozen_eta_only_MLP | 41 | 0 | 0.1295 |
| toy | frozen_full | 44 | 0 | 0.0690 |

Ring full的校准/回归明显优于eta-only，但只有3个额外B15（58 vs55），且state shuffle影响很小；回归优势不等价于学会反转。Four full19+5未认证与eta-only23+1未认证不能解释成四个确定的ranking错误。所有A/B/C逐模型表见 baseline_metrics.json；原始eta是否见于TRAIN及state overlap见 unseen_state_eta_audit.json。

## 6. Feature mechanism：哪些输入参与排序

| enriched场景 | feature group | permutation top1一致 | selected B15均值 |
|---|---|---:|---:|
| ring | Flow_reference | 0.870 | 56.60 |
| ring | agent_relative | 0.857 | 57.80 |
| ring | entity_masks_counts | 1.000 | 58.00 |
| ring | goal_relative | 0.883 | 58.20 |
| ring | obstacle_geometry | 0.907 | 58.00 |
| ring | policy_frame | 1.000 | 58.00 |
| ring | remaining_time | 1.000 | 58.00 |
| ring | safety_quantities | 0.953 | 58.20 |
| four | Flow_reference | 0.667 | 17.80 |
| four | agent_relative | 0.750 | 18.40 |
| four | entity_masks_counts | 1.000 | 19.00 |
| four | goal_relative | 0.825 | 18.80 |
| four | obstacle_geometry | 0.558 | 19.60 |
| four | policy_frame | 0.875 | 17.80 |
| four | remaining_time | 1.000 | 19.00 |
| four | safety_quantities | 0.917 | 18.40 |

Ring共享探针中，强交互方向准确率原68.3%；打乱agent relations降至约49.0%，goal-relative约52.7%，geometry约55.0%，Flow约60.1%。因此不能说联合模型完全不使用物理state：这些字段参与其弱交互信号。但这不是已证实的正确反转机制。

同场景TRAIN均值替换与5个permutation seeds均已完成，详细score/top1/Q/interaction结果见 feature_metrics.json 和 feature_interaction_metrics.json。LOSO不使用目标TRAIN均值，无法合法构造同条件source reference时记N/A。各feature组可能重叠，混合字段可能离开物理流形；因此这是函数依赖诊断，不是物理因果干预。time/count在该t0固定N/M队列中恒定，零影响意味着不可辨识，不能解释为不重要。没有用梯度或权重大小替代上述证据。

## 7. Generator 与 critic 严格分开

state-conditioned generator已先过滤候选，所以K16 oracle高、critic shuffle影响小可以同时成立。系统整体依赖state并不能证明critic依赖state。共享eta面板去掉了候选生成这一混淆；LOSO仅指critic目标标签未参与拟合，不表示此前目标监督的generator也是zero-shot。

## 8. 最终四项判断

- STATE_USAGE：联合模型确实使用h；Ring scene-specific基本eta排序，Four scene-specific在冻结K16上top1完全不随shuffle改变。
- STATE_ETA_INTERACTION：Ring joint有弱的差值幅度信号，但可靠偏好反转未被建立；Four暂无支持性直接证据；Toy有经验正证据。
- INTERACTION_GENERALIZATION：主场景未见state/eta上的支持薄弱，LOSO不支持交互迁移；不能因架构可表达而声称迁移成功。
- FEATURE_MECHANISM：关系、goal、geometry、Flow参与Ring弱信号；常量time/count不可判断；依赖输入不等于用对输入。

整体分类 **UNDERRESOLVED**。这不是“没做完”，而是主场景中还缺少独立、充分的真排序反转证据，且目前直接差值预测质量不足。scene-specific Ring/Four 在本轮范围可称 ETA_ONLY_EFFECTIVELY；joint Ring/Four不能硬贴精确additive标签，因为其函数存在非可加性，但非可加性未等价为可靠科学能力。Toy joint的有限经验结果支持 STATE_ETA_INTERACTION_GENERALIZES_WITHIN_SCENE。没有任何层次支持 STATE_ETA_INTERACTION_TRANSFERS_ACROSS_SCENES。

## 9. 数据库、数值与边界

请求2304 continuation；执行统计 {'completed_seeds': 2304, 'existing_numerical': 0, 'numerical_unresolved': 17, 'physical_attempts': 2355, 'requested': 2304, 'reused': 0}；新增数据库唯一seed记录 2304；未认证seed 17，涉及11个eta tuple；有效碰撞0。full-Q计算排除未完整认证格子，保留上下界及全部numerical记录；这种缺失可能相关于eta，不隐藏其选择偏差。未使用提前停止k/n冒充Q16。

所有新增证据由标准cache preflight、journal和DatabaseSink持久化；没有生成训练标签或改变模型/归一化/OrthoFlow3/安全/成功条件。现有checkpoint哈希校验通过，函数分解与完整网络一致，合成additive/interaction回归通过。

最小下一步（本轮不自动执行）：在独立train/dev状态上，以共享eta边界面板取得足够、经独立seed块复核的排序反转，再判断是监督缺少反转还是critic拟合不了；不是先换architecture，也不是把更多高oracle K16当interaction证据。
