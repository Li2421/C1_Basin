H_cov_L REJECTED，H_switch REJECTED；H_cov_E REJECTED。controller regime 判定：配方差异为主。N 的配方 ICC=0.813；14 个配方中有 3 个配方，其四个训练 seed 的 N 符号不一致。

证据级别：DEVELOPMENT。完整波次 4/4，controller 56/56；新 outcome case 的 unknown 数 376。下界与上界均存于 results.json；unknown 保留，不重跑，timeout 未称为 deadlock。

## 事实与门槛

H_cov_L：REJECTED；H_cov_E：REJECTED；H_switch：REJECTED。门槛及特征/分析收据见 PREREGISTRATION.md、ANALYSIS_FROZEN.receipt.json、ANALYSIS_DETAIL.receipt.json、R5_FROZEN.receipt.json、R6_FROZEN.receipt.json。
H_cov_E 折 bootstrap 差值 95% CI=[-0.010736527433929458, 0.009683315740981163]；H_switch 配对遗憾差 95% CI=[0.0008370535714285714, 0.012137276785714286]。

## 历史证据、兼容性与前瞻范围

历史代码与配方来自 round14A 的 GPU 复现门；本轮 56 个 checkpoint 的训练、参数新颖性和兼容性审计见 TRAINING_AUDIT.json、CHECKPOINT_NOVELTY_AUDIT.json、COMPATIBILITY_AUDIT.json；32 个新 state 的去重见 STATE_DEDUP_AUDIT.json。新 controller 和新 state 的 Q4 标签为前瞻采集；交叉验证同属该数据集，故学习曲线为 DEVELOPMENT。
缓存：预检 {'total_requested': 57344, 'exact_reusable': 0, 'partial_reusable': 0, 'aggregate_reusable': 0, 'ambiguous': 0, 'incompatible': 57344, 'genuinely_missing': 57344}；完整波次合并后标准复检 {'aggregate_reusable': 0, 'ambiguous': 0, 'exact_reusable': 55900, 'genuinely_missing': 376, 'incompatible': 0, 'partial_reusable': 1068, 'total_requested': 57344}。标准复检把 376 条已在 DB 中的 numerical_failure 按可复用性标为 missing；POSTFLIGHT_UNKNOWN_AUDIT.json 逐条核对 DB、raw 和 journal，真实缺失=0，这些案例保留为 unknown，不重跑。32 例一致性门通过=True。

## 数值结果

下表 RMSE 与遗憾用成功率比例表示，乘 100 即百分点；未知结果的上界敏感性在 results.json。

|特征|k|E RMSE|L Spearman|L RMSE 比|G_E 遗憾|G_EL 遗憾|
|---|---:|---:|---:|---:|---:|---:|
|R5-PCA2|1|0.117|0.047|0.904|0.013|0.014|
|R5-PCA2|2|0.117|0.050|0.913|0.013|0.014|
|R5-PCA2|3|0.117|0.068|0.909|0.013|0.013|
|R5-PCA2|4|0.117|0.061|0.907|0.013|0.014|
|R6-Delta|1|0.352|0.711|2.863|0.042|0.046|
|R6-Delta|2|1.741|0.721|2.292|0.057|0.053|
|R6-Delta|3|0.530|0.738|1.119|0.036|0.041|
|R6-Delta|4|0.376|0.758|0.764|0.036|0.038|
|R5-PCA2+R6-Delta|1|0.887|0.713|3.338|0.027|0.034|
|R5-PCA2+R6-Delta|2|1.115|0.733|1.812|0.028|0.037|
|R5-PCA2+R6-Delta|3|0.330|0.722|1.153|0.028|0.037|
|R5-PCA2+R6-Delta|4|0.318|0.731|0.829|0.028|0.034|

### 新训练 seed 泛化

|特征|训练 seed 数|E RMSE|L Spearman|L RMSE 比|t* ±100 比例|
|---|---:|---:|---:|---:|---:|
|R5-PCA2|1|0.117|0.395|0.827|0.589|
|R5-PCA2|2|0.111|0.415|0.830|0.607|
|R5-PCA2|3|0.110|0.389|0.834|0.643|
|R6-Delta|1|0.401|0.709|5.659|0.536|
|R6-Delta|2|0.517|0.764|2.188|0.554|
|R6-Delta|3|0.371|0.838|0.528|0.679|
|R5-PCA2+R6-Delta|1|0.531|0.752|4.605|0.518|
|R5-PCA2+R6-Delta|2|0.332|0.776|2.218|0.482|
|R5-PCA2+R6-Delta|3|0.190|0.830|0.580|0.589|

### 方差分解

|标签|配方 ICC|95% CI|配方占比|同配方 seed 占比|
|---|---:|---:|---:|---:|
|E|0.902|0.755 ～ 0.960|0.902|0.098|
|L|0.894|0.774 ～ 0.939|0.894|0.106|
|N|0.813|0.361 ～ 0.922|0.813|0.187|
|t*|0.660|0.186 ～ 0.910|0.660|0.340|

预注册描述性预期为至少 3/14 配方的四个训练 seed 对 N 符号不一致；实际为 3/14，此项不是判定门。

### 外部 E 验证（描述性）

|组|状态|n|Spearman|RMSE|
|---|---|---:|---:|---:|
|round10_best14|DESCRIPTIVE|14|0.855|0.080|
|round12_siblings8|DESCRIPTIVE|8|0.881|0.118|
|round14A_independent14|DESCRIPTIVE|14|0.760|0.101|

## Split-half 信度及噪声参照

|标签|Split-half Spearman|Split-half Pearson|推算相关上限|
|---|---:|---:|---:|
|E|0.973|0.984|0.996|
|L|0.971|0.980|0.995|
|N|0.960|0.971|0.993|
|t*|0.701|0.775|0.934|

相关上限采用正的 half-Pearson 经 Spearman–Brown 修正后开方；负值或常数标签不推算。它是独立噪声假设下的参照，不是门槛。所有详细折、seed 泛化、符号准确率、t* 误差、四种规则及各动作 split-half 信度在 results.json。图见 LEARNING_CURVES.png、E_L_SCATTER.png 和 VARIANCE_DECOMPOSITION.png。

## 解释与限制

H 假设仅依据预冻结门槛；若少于四波，k=4 门无法检验。Q4 是低预算数据集而非 B15 认证。纯 Q 数据生成与评估按 AGENTS.md 的预注册例外免视频入口。

解释（描述性）：有 3/14 个配方的不同训练 seed 对 N 的符号判断不一致，单凭配方无法确定每个 seed 的偏好；总体 N 方差仍以配方之间为主。

## 资源记录

提交前、训练中及各波 worker 启动时的 squeue/进程快照保存在 RESOURCE_USAGE.jsonl 和 wave*/RESOURCE_worker_*.json；快照最高 CPU=24、GPU=3 shard。各波开始的占用与时间还保存在 START.receipt.json。08:30 前上限 24/6，08:30 起上限 18/2。
