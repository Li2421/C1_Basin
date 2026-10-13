# 实验前证据与成本清单

本清单在本轮新增计算前核对。主体只做缓存读取、16-key 既有动作表重算及静态作图；补充了四个冻结初态/η/key 的 **action-only Flow 子步前向探针**，没有新增完整或短 continuation。

|旧资产|位置|本轮兼容性/用途|仍缺证据|
|---|---|---|---|
|四链控制器和原 Flow、CBF|`/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_stage2/factorial_controller.py`、`poc/field_controller.py`|与 hard-field frozen descriptor 对应；重建公式和监视器语义|无新代码改动|
|49,152 条确认记录|`/home/zhihan/research/Basin_C1_hard_field_20261006/{confirm_q16_tasks.json,results/,rollout.sqlite}`|同 state/η/seed 四链，正式 success/B15 与已存动作/轨迹；本轮只取 Toy FT/FF、Ring TT/TF|缺独立新初态的前瞻标签，但 320 continuation 不足以广泛重建 16-seed B15|
|动作传递 16-key 旧表|`.../transfer_mechanism_v2/state_eta_chain_features.csv`、`results/`|与上述相同控制器/key，首动作 pre/post、投影与方向特征；初态每 state×η×chain 有 16-key 平均|旧 NPZ 仅有子步投影形变，没有两修正条件的完整 `h_k`；也没有 A/B 中介的独立安全闭环介入|
|旧 H1–H4|`diagnostics/placement_boundary_mechanism_audit_v1/`、`diagnostics/placement_strong_hypotheses_v1/`|复用 TT/FF 局部/边界、Ring 形状变化和 Toy 非光滑结论；H1 的动作族交换只覆盖 TT/FF，不冒充本轮主 pair 的介入|TT→TF、FT→FF 的特定 mediator 因果拆解仍缺|
|上一轮 Part 2|`diagnostics/part2_closed_loop_sensitivity_validation_20261012/`|复用 `b15_by_candidate.csv`、`gain_loss_audit.csv`、`gain_loss_mechanism_seed0.csv`、S/M/G/K、boundary 诊断；已核 49,152 记录，本轮不重做 M/G/K|旧 ρ 是代理，不能当新 B15 预测标签|

**补充缺口及成本审计：** 为直接观察 `h_k(Bη)-h_k(0)`，仅在预定 UID 01/02、η[4]、seed0 上运行每例两次 10 子步冻结 Flow 积分；四例共 8 次积分，加上真实动作重建检查。没有 `env.step`、训练或 GPU 使用。该探针是从已知历史示例选取，不能用来估计总体频率或验证新预测条件。

**本轮成本：**完整 continuations 0/320，短 continuations 0；CPU 单进程读取缓存及四例 action-only forward，GPU 0。所有主证据路径与读入行数见 `verification.json`、`cache_index.json`。监视器从已保存位置/动作独立重放 768 条 Toy seed0 轨迹，与正式 `max_stuck_timer` 全数相符。旧结果身份不合共享 DB 的新精确请求时不强制并库；此次不产生新 rollout 请求，无需运行 rollout preflight。
