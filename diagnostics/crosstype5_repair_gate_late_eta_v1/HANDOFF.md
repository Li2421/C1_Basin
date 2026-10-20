# 跨类型5交接

本轮主方法 = 学习双 η·critic 判修不修·扩充提议。全部19个类型都是开发证据。

从 REPORT.md、FINAL_RESULTS.json、COVERAGE.md、GAPS.md、INTEGRITY_COMPARISON.json 开始阅读。冻结身份和全部外层/最终模型在 FROZEN_PIPELINE.json；动作及选择在 NEW_EVAL_CHOSEN_ACTIONS.json，收据早于新state rollout。训练原始来源为跨类型4的34动作槽，本轮 TRAINING_BUNDLE.npz 扩为44槽；安全投影有第二seed，其他第二seed为结构性未观测，不当成失败。

可直接复用：schedule_generator.py 与 schedule_generator_fits/、critic_model.py 与 critic_fits/、candidate_menu.py、gate_selection.py 与 SELECTED_GATES.json；实际执行链为 execution_v3.py/requery_execution_v3.py，冻结生成器在独立CPU helper内推理，实际controller进程XLA_FLAGS必须unset，防止旧controller算术改变。已有given动作身份不变。执行证明以 EXECUTION_REPRODUCTION.json、REQUERY_EQUIVALENCE_v3.json 和 EARLY_TERMINATION_EQUIVALENCE_v3.json 为准。

每个阶段 *_BUDGET.json / *_COMPLETENESS.json 分开保存真实执行、精确缓存和本地推断。*_ALL_LOCAL_EQUIVALENCE.json 仅本地，不进入共享DB，不可用作下一层推断来源。后续新问题先查历史、精确preflight与现有预测/checkpoint；不因critic编码改变重跑旧物理标签。

用户持续执行要求见USER_EXECUTION_RULES.md：不影响实验有效性的缺包、绘图或格式化问题必须记缺口并继续训练与评估，跨轮次持续生效，不再次询问。coordinator.py 是本轮顺序调度与恢复入口；排队与claims不自动重试。2026-10-20 09:00新加坡时间起，CPU/GPU上限为18/2（此前24/5）；以RESOURCE_RELEASE_POLICY.json为准，每次提交查全用户占用。旧目录、源码、PoC、官方仓库和venv只读，禁止前缀只能stat。
