# 跨type1b：按state的生成器与controller-conditioned critic

依据 docs/prompt_crosstype1b_generator_critic_pipeline_20261018.md 全文与共同协议第一至六节。全部新代码/输出仅本目录，旧源码、跨type1、冻结生成器/critic、PoC、官方仓库与venv只读；禁读目录按decentralized_/teacher_student_/multiscene_/crosstype2_前缀剪枝。DB仅官方INSERT OR IGNORE身份登记、默认journal、单进程merger。不开git写操作，不安装包。新评估state冻结前不执行任何闭环rollout。现有任务优先，全局18 CPU、2 GPU shard。

范围为原Ring H700、集中安全投影、x64=False，Flow用FF，其余用TT。OrthoFlow3基、物理、安全层、事件与成功标准不变。直接复用跨type1全部62个合格controller和24维无成败特征；隐式能量类型不处理。未知保留不重跑，timeout不称为deadlock。

训练state为跨type1的16调参+32评估，共48个；锚点标签和原seed/RNG语义按精确身份复用。新评估24state使用round10生成器，种子从202610180801起顺序拒绝采样，物理初始状态与DB全部可恢复state去重；不按任何模型预测或成败筛选。训练/新评估state均使用原FrozenNewRuntime.conditioning定义：固定参考Flow观测和参考动作，h100与c33不看被测controller；生成器归一化固定原版本。

生成器冻结seed41/step1900；原critic冻结seed23/step2700。每state候选为原变换均值和4个SHA256确定性样本，原stable_int('generator-v1-proposals',41,'ring_exchange',state_uid)；每η形成全程固定与按进度切η10两个动作，外加安全基线、前500步η13后η10、按进度切换三个锚点。共13候选，另报nominal-only。进度阈值0.9681620884830543，最晚600步切换，直接复用原ProgressSwitch。η10/η13用历史完整精度，不用文档三位近似；生成器输出float32精确提升float64后登记，不取整。候选先计算/封存后跑标签，无critic连续优化或额外搜索。

预算初定：训练生成器动作62×48×10×seed0=29760条，新评估62×24×2×14=41664条，共71424条，扣精确复用；改变候选的开发追加上限10000。先做Flow、扩散行为克隆、参数共享独立近端策略优化三类试点，再补其余类型；试点不改变后续设定。

原prompt有训练期交叉拟合的证据冲突：生成器动作只给seed0，而全13候选的按state交叉拟合需要seed0和seed1；即使32个旧评估state的锚点有两seed，新增生成器动作仍没有seed1。已向用户请求选择是否在精确preflight后增加最多19840条以补齐；用户回答前不执行该增量，不伪造第二seed或把锚点oracle冒充全候选oracle。若预算保持原状，相关训练期oracle和依赖目标明确列缺口，新评估state全部候选两seed按原设计完成。

critic输入h100/c33/C24与动作10维（前η3、后η3、三种切换独热、safety标志）；源类型内填补/标准化，缺失controller特征保留明确标志或源均值填补，处理只拟合源类型。起点为state适配96、context32、action32、controller32、共享128/64、单logit、SiLU；从头训练和冻结原critic权重warm start均报告。warm start仅拷贝形状相容状态/上下文/共享参数，η编码第一3维继承、其余维置0，新增controller到共享层权重置0；完整映射另存收据。所有controller标签来自已观察的成功/失败计数，unknown不作0；二项NLL使用min(已观察seed数,16)权重。

AdamW lr1e-3、wd1e-4、梯度范数5；ensemble seeds17/23/41。初始训练最多2000步，每100步记录源内留一类型NLL和同state候选排序准确率（经验成功率差至少0.25的严格配对，预测平局计0.5），保存候选checkpoint。batch256，类型/controller/state/candidate均衡采样；实际观察seed数作为损失权重，不以unknown填补。步数和warm/cold选择只用源类型内留一的ensemble NLL，精确并列优先更早步数，再更高排序准确率，再从头训练。外层留出标签只评分/画监控曲线，不用于该折超参数选择。后续结构迭代逐次记日志，冻结前封存实际最终config。

主选择为ensemble平均概率最大；另报均值减1倍标准差，若最高保守分数低于按进度锚点平均预测概率则回退按进度锚点。预测精确并列优先按进度锚点，然后候选清单顺序。两种规则实际主版本只按源内验证选择，完整保留每种结果。对照为两个统一时间表、原跨type1冻结controller选择器、原生成器+原critic仅5固定提议、生成器均值进度切换、去controller分支、去state/context分支；消融同训练预算/源内选择程序。

外层九折留一类型、内层只在八源类型中留一；最后额外拟合全部九类型的模型。开发首报全标签可选空间与生成器覆盖；两seed完整支持的state才算按seed交叉拟合，不把单seed样本内oracle作为交叉拟合。开发目标与新state终点原样依prompt，不改阈值。类型等权主、controller等权辅；5000次配对type→controller→state成簇bootstrap，seed保留同簇，oracle每次重算，unknown报下/上界。bootstrap种子202610180901。

冻结九个外折模型、全类型模型、输入预处理、候选与选臂规则、代码/参数/UTC/sha256后，为新评估state生成全部方法选择并封存；收据必须早于新评估第一个rollout提交。然后全13候选加nominal-only全部两seed跑齐，仅一次确认。没有证据的类型或终点保留缺口，不以子集替代完整结论。

mandatory gates：先复现跨type1的16条覆盖9类型原始记录，首动作/全位置/终止/步数逐位一致；重算冻结Ring验证提议要求逐位或绝对误差≤1e-6并解释；原critic分数重算对照原记录，保留逐位/误差/argmax检查。任何复现失败停止，不补跑新实验。所有历史sources和checkpoint在导入/复制前登记SHA；开工/结束审计和源码复制diff交付。视频不要求。
