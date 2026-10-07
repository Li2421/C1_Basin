# Common-eta state–eta interaction panel v1

## 设计与执行

使用 unified v3 中 TRAIN/validation 的 correction-needed 状态；每场景 24 个，固定共同 eta 面板 24 个，完整 canonical seeds 0–15。没有使用任何 frozen test state，没有修改/训练 generator 或 critic。Q16 仅在 16 个有效 seed 全部完成时报告；数值失败保留为区间并不作成功/失败插补。B15 为至少 15/16 成功。

| 场景 | 候选 correction-needed | 入选 | 不同 parent | train/validation | 可复用有效 seeds | 已缓存数值失败 | 新缺失 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| four_way_intersection | 77 | 24 | 24 | {'train': 16, 'validation': 8} | 3077 | 5 | 6134 |
| ring_exchange | 24 | 24 | 24 | {'train': 22, 'validation': 2} | 3173 | 10 | 6033 |

共同面板顺序：eta=0、train/dev fair Four-Way eta、train/dev fair Ring eta、canonical primary Sobol design 的前 21 个有效点。每个 rollout 在 shared DB 中按 state×eta×controller×future-index 精确键存储。

## TRUE Q 与 state-dependent ranking

| 场景 | 完整 Q16 cells | 数值未决 seeds | B15 真/假/未决 tuples | 碰撞 | 不同 state-best eta | best eta 不同于全局 eta | state-pair Spearman 中位数 | strong reversal quadruples | 含 strong reversal 的 state-pair | fraction | state-specific oracle Q | global eta Q | oracle gap | interaction energy ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| four_way_intersection | 553/576 | 24 | 121/455/0 | 0 | 1 | 0.000 | 0.920 | 28 | 24/276 | 0.087 | 1.000–1.000 | 1.000–1.000 | 0.000–0.000 | 0.022 |
| ring_exchange | 531/576 | 52 | 119/457/0 | 0 | 2 | 0.083 | 0.762 | 709 | 191/276 | 0.692 | 1.000–1.000 | 0.992–0.995 | 0.005–0.008 | 0.213 |

## 每场景详细结果

### four_way_intersection

判断：**UNDERRESOLVED**。

- Q16 矩阵：553/576 个 cell 完整；12/24 个 state 行全部完整；24 个 seed 数值未决。未决 cell 使用 Q_lower/Q_upper；状态间 rank 相似度用每对状态共同拥有的精确 eta 子集。所有状态均精确的 eta 列为 [0, 1, 3, 10, 14, 18, 19, 20]。
- State ranking：1 种 observed-best eta 身份；5/24 个状态可能被未决 eta 的 Q 上界改变 observed-best；0.000 状态的 observed-best Q 严格高于全局 eta 的 Q；state-pair Spearman/Kendall 中位数分别为 0.920/0.899（276 对）。
- Strong reversal（双侧差异至少 4/16）：28 个 eta-pair×state-pair quadruples；24/276 state pairs (8.7%)，涉及 24 个 states。较弱反转另计 139 个。
- Oracle：state-specific mean Q bounds=1.000–1.000，B15 coverage bounds=100.0%–100.0%；global eta [0.5291672244202346, -0.011185371316969395, 0.01067163236439228] mean Q bounds=1.000–1.000，B15=100.0%–100.0%；STATE_SPECIFIC_ORACLE_GAP bounds=0.000–0.000。
- Eta-only：top1 对 observed state-best identity accuracy=100.0%；top3 hit=100.0%；mean regret lower bound=0.000；B15=100.0%–100.0%。
- Frozen critic：mean per-state Spearman=0.791；top1 mean true Q bounds=0.992–1.000，regret bounds=0.000–0.008；top3 hit=0.0%；B15 selection=100.0%–100.0%；strong-reversal direction accuracy=0.0；B15 change vs eta-only bounds=0.0–0.0。
- State-score shuffle：
| shuffle seed | top1 agreement | shuffled-selection true Q | original true Q | score correlation |
|---:|---:|---:|---:|---:|
| 20261003 | 1.000 | 0.992–1.000 | 0.992–1.000 | 0.996 |
| 20261004 | 1.000 | 0.992–1.000 | 0.992–1.000 | 0.997 |
| 20261005 | 1.000 | 0.992–1.000 | 0.992–1.000 | 0.996 |
| 20261006 | 1.000 | 0.992–1.000 | 0.992–1.000 | 0.998 |
| 20261007 | 1.000 | 0.992–1.000 | 0.992–1.000 | 0.997 |


### ring_exchange

判断：**TRUE_INTERACTION_PRESENT**。

- Q16 矩阵：531/576 个 cell 完整；1/24 个 state 行全部完整；52 个 seed 数值未决。未决 cell 使用 Q_lower/Q_upper；状态间 rank 相似度用每对状态共同拥有的精确 eta 子集。所有状态均精确的 eta 列为 [0, 1, 4, 8, 10, 11, 14, 16, 18, 22]。
- State ranking：2 种 observed-best eta 身份；2/24 个状态可能被未决 eta 的 Q 上界改变 observed-best；0.083 状态的 observed-best Q 严格高于全局 eta 的 Q；state-pair Spearman/Kendall 中位数分别为 0.762/0.701（276 对）。
- Strong reversal（双侧差异至少 4/16）：709 个 eta-pair×state-pair quadruples；191/276 state pairs (69.2%)，涉及 24 个 states。较弱反转另计 1601 个。
- Oracle：state-specific mean Q bounds=1.000–1.000，B15 coverage bounds=100.0%–100.0%；global eta [0.8806116178166121, 0.02854111511260271, 0.024503270164132118] mean Q bounds=0.992–0.995，B15=100.0%–100.0%；STATE_SPECIFIC_ORACLE_GAP bounds=0.005–0.008。
- Eta-only：top1 对 observed state-best identity accuracy=12.5%；top3 hit=100.0%；mean regret lower bound=0.005；B15=100.0%–100.0%。
- Frozen critic：mean per-state Spearman=0.617；top1 mean true Q bounds=0.826–0.826，regret bounds=0.174–0.174；top3 hit=45.8%；B15 selection=79.2%–79.2%；strong-reversal direction accuracy=0.07334273624823695；B15 change vs eta-only bounds=-0.20833333333333337–-0.20833333333333337。
- State-score shuffle：
| shuffle seed | top1 agreement | shuffled-selection true Q | original true Q | score correlation |
|---:|---:|---:|---:|---:|
| 20261003 | 0.292 | 0.872–0.872 | 0.826–0.826 | 0.899 |
| 20261004 | 0.500 | 0.836–0.836 | 0.826–0.826 | 0.920 |
| 20261005 | 0.375 | 0.818–0.820 | 0.826–0.826 | 0.911 |
| 20261006 | 0.500 | 0.909–0.909 | 0.826–0.826 | 0.915 |
| 20261007 | 0.375 | 0.891–0.891 | 0.826–0.826 | 0.913 |


## 解释边界

该分析比较同一 state cohort 上的真实 fixed-eta continuation Q，不是把 state 难度变化误作 eta-ranking 交互。双侧强反转是直接的 state×eta ranking 证据；state-specific oracle gap 衡量 24 点共同面板内的可用性差异。加性分解是描述统计，不能单独作因果结论。critic 分数是冻结模型对同一共同面板的逐格评分；shuffle 仅为诊断。结果只覆盖此固定面板和训练/开发状态，不是总体概率的无偏保证。

## 文件与完整性

- `states_manifest.json`、`common_eta_panel.json`：outcome-blind 冻结 cohort/panel。
- `Q16_matrices.json`、`q16_records.json`：矩阵、逐 seed 成功与数据库 rollout UID。
- `critic_scores_frozen.json`：冻结 critic score/logit 矩阵。
- `analysis.json`：排序、强/弱 reversal 原始记录、oracle、加性分解、eta-only、critic 与 shuffle 指标。
- `cache_preflight.json`：rollout 前兼容缓存检查。
- `run_shard*of18.json` 与 `execution/*/raw/`：增量执行摘要和可审计原始记录。
- `analysis_preregistration.json`：分析阈值和 ties 规则。

执行合计：1,152 个 state×eta tuple、18,432 个 canonical seed 槽位；有效结果 18,356，数值未决 76，缺失/未运行 0。Rollout DB 在本批次前复用 6,250 个有效 seed 结果及 15 个已达重试上限的数值结果；本批新执行 12,350 次（含 183 次相同 seed 的数值重试），留下 61 个新数值未决。所有 76 个未决 seed 均不插补；所有 tuple 的 B15 二元分类均可判定。有效 rollout 碰撞数为 0。

