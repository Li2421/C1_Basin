# 历史证据与必要缺口

- `diagnostics/eta_diverse_scene_test_v1/COMPLETE.json`、`INTEGRITY_COMPARE.json`：完成与完整性通过；复用其私有执行图，数值模块不改。两局本轮私有图逐位重放通过才启动。
- `diagnostics/cross_scene_correction_v1/SCENES.json`、`LABEL_TABLES.json`、`SELECTED_CONTROLLERS.json`：直接复用五个已有场景的三控制器×49候选标签；对穿原0–11状态与配置逐字段核对。能力门第一批四状态的384条旧单机器人请求直接复用，不重跑。
- `diagnostics/cross_scene_pipeline_v2/REPORT.md`、`COMPLETE.json`、`INTEGRITY_COMPARE.json`、学习代码：旧11场景上主方法无明显收益；方法和训练超参数逐字复用，旧checkpoint、预测的训练场景集与本轮7场景不符，不能直接替代本轮折模型。仅对精确同源六场景的内部λ拟合去重。
- `diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json`：复用49候选精确float64与顺序，不增候选搜索。
- 旧结果不能回答三个新几何场景与对穿12–27新状态的成功率，必须补这些请求；探索结果不作为证据，也不导入其模型或标签。三新场景0–7参与过探索，8–23作核对与留出计分。
- 共享数据库官方预检只补真正缺失的UID。执行图位置变化不使旧标签失效：已有标签与能力门均直接复用；新物理状态才新跑。生成器只跑全部折冻结后主方法实际选中的提案，至多384条。
- `diagnostics/eta_diverse_scene_test_v2/`、`diagnostics/cross_scene_pipeline_v2_learning_tests/`仅登记元数据；`diagnostics/cross_scene_prospective_v1/`仅os.stat，不打开或hash。

几何有效、单机路径、联合可解、控制器成功和死锁解决分开报告。timeout不标死锁，不替换无效状态或不通过能力门的场景。

旧模型与预测的实际兼容性核对见HISTORICAL_LEARNING_COMPATIBILITY.json：检查了旧留一场景折定义、选中checkpoint与RANKING_SCORES预测，旧折10个源场景，与本轮7个源场景不同，不能直接复用模型；旧标签继续有效。
