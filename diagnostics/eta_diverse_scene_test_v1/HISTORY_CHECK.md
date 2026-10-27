# 历史证据检索与复用

本轮只出标签与统计，不训练、不读取旧学习模型权重、不做第二阶段。cross_scene_prospective_v1仅os.stat元数据。

| 来源 | 兼容性与复用 | 仍缺证据 |
|---|---|---|
| diagnostics/cross_scene_correction_v1/SCENES.json、LABEL_TABLES.json、SELECTED_CONTROLLERS.json、pilot_CASES.json及原轨迹 | 复用对穿6机的12个初始状态与基础配置；11个现有场景的指定3控制器×25候选直接取旧标签；2局快速控制器逐位重放已通过 | 四个单旋钮场景改变几何、半径或时限，原场景标签不能替代新请求；精确缓存由官方预检决定 |
| diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json | 精确float64的预定25候选 | 不增补或搜索候选 |
| diagnostics/cross_scene_pipeline_v1/、cross_scene_pipeline_v2/ | 复用已验证隔离构建方式、数值执行封装、调度与成本证据；学习模型及预测不用于本测试 | 四个新场景的能力门与候选标签 |
| shared_rollout_db/ | 官方只读历史匹配与精确请求预检；仅真正缺失结果补跑 | 新状态UID含场景配置，RNG沿用原state_uid规则；不能把不同配置旧轨迹直接当作新seed0轨迹 |

控制器口径核对：说明中的圆形五机历史最优线索来自原合格控制器集合；限制到指定3个控制器后，全49候选最优与本轮25候选最优略有差异。完整数据见HISTORICAL_COHORT_CHECK.json。本轮严格使用预定25候选，结论只限该集合，不追加候选。

共享数据库同名历史场景检索见HISTORICAL_MATCH_SEARCH.json；注册前仍按完整场景、状态、η、控制器和种子UID预检。几何有效、单机器人路径、联合可解、控制器成功、死锁解决分别报告；不替换非法状态，timeout不等于deadlock。
