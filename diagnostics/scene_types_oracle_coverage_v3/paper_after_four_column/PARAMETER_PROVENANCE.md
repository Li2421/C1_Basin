# 已执行参数的来源及修复约束

用户2026-10-28明确要求：修复必须根据作者代码或论文叙述，不准凭成功率或个人想法修改。任何未能提供原文/代码定位的公式、参数、机制改动都不能作为正式论文复现修复。已有结果保持原始身份；审计标签修正不更改已有数值。

此前回答将“reference_default”简称为“默认”不够准确：Wang与自适应风险是历史公式端口参照值，并非已确认的作者默认。应称为“已跑参照配置”；其余方法也包含向共同物理环境的适配，不能把整个配置称为作者原实验配置。

|方法|已跑参照配置|v1开发集选参|可追溯依据|
|---|---|---|---|
|Wang一致扰动、完整消解|alpha=1,gamma=1|alpha=5,gamma=1|历史公式端口构造参数；尚不能将这组数值称为作者默认|
|Zhang自适应风险|gamma=1,q=1,c=9.75|gamma=1,q=0.01,c=9.75|历史公式端口参照；需逐项对论文核验，不是作者完整配置|
|Zhang DAC|alpha=2,agentCA阈值=0.1,obsCA阈值=0.1|相同，合并执行|作者vendor/dac/agents/circle2DAgent.py:47,53|
|Lee MGR|tau=0.5,K_P=0.05|tau=1.0,K_P=0.05|作者vendor/mgr/main.py:81设tau=0.5，params.py:7设K_P=0.05；Simulator构造函数另有tau=1.0默认值，应区分入口脚本与类默认|
|Chen IMPC-OB|K=8,scale_lengths=false,epsilon_gain=1|相同，合并执行|作者vendor/impc_ob/src/SET.py:27的K=8；scale_lengths与epsilon_gain是端口选项，不是作者原参数名称|

所有vendor路径相对只读 `diagnostics/paper_baselines_20261028/`。精确作者提交见本轮 `PAPER_REGISTRY.json`。

“v1开发集选参”来自只读 `diagnostics/scene_types_basis_dimension_v1/PAPER_PARAMETERS_FROZEN.json` 的 `selected_development`，在旧60案例开发集上选取：先成功数，再碰撞、数值未知、全队列截断平均时间。它不是论文推荐参数，也不是本轮17场景上重新调出的参数。`historical`与`selected_development`两个字段不能混称；当前使用后者。

另一个Chen配置将 `scale_lengths=true`，是旧端口的物理长度缩放方案（当前机器人半径/0.3），不是另一种论文方法，也不能标为作者默认。

上述表仅列配置间变化/明确覆盖的字段，并非完整有效参数：

- Wang端口还有 kp=1,kv=2,kleft=2,kright=0.5,kdelta=1,zero_tolerance=1e-6。这些数值与触发判据仍需逐项核验。
- 风险端口还有 p=100,alpha=1,beta=1,phi=10,t=10,epsilon=1e-5,psi_scale=0.25,omega_recovery=0.1。这些参数及净空函数选择不能仅因满足一般性质就视作论文原设定。
- 作者控制器与共同场景的dt、速度、半径、墙体表示、路线引导、异常处理之间的适配也必须审计。例如Chen作者SET.py设dt=0.15，而本轮共同动力学dt=0.05；K相同不意味着预测时长相同。

下一步修复的准入依据：原文公式/算法/参数段或固定提交的代码位置；记录旧行为、修正行为、对照证据与影响范围。求解器返回失败本身不授权放宽约束、更换机制、提高迭代上限或调整增益。真正不满足论文假设的案例要报告适用性缺口。

已确认的诊断事实：DAC大量ExitFlag=2；作者qpSWIFT/include/GlobalOptions.h:56定义其为迭代上限，DAC作者代码使用MAXITER=30。这解释错误类型，但**不构成擅自将30改大或更换求解器的依据**。尚未据此修改控制器。
