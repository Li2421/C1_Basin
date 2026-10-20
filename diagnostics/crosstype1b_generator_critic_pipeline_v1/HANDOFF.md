本轮已完成开发、冻结和一次性新state评估。结论与完整性审计见REPORT.md，数据限制见GAPS.md。

下一条跨type2线可只读复用：critic_model.py与action_only_model.py（结构/源类型预处理）、train_critic.py与train_action_only.py（源集合缓存与GPU三seed训练）、select_models.py/select_models_v2.py/select_matched_v2b.py（源内嵌套选参、正controller权重收缩、匹配消融）、new_evaluation_choices.py（冻结推理与归一化dtype恢复）、statistics_pipeline.py（配对分层bootstrap）、CANDIDATES.json与condition_candidates_v2.py（冻结候选规则）、FROZEN_PIPELINE.json（九外层与全类型模型路径/sha）及NEW_EVAL_CHOICES.receipt.json（选择时间收据）。

历史执行链和数据均保留原位；不得直接运行有旧目录写副作用的构造器。需要改变实现时复制到新轮目录并记录来源；必须再次按新prompt做源保护、历史证据核查和DB preflight。

本轮训练生成器标签仅seed0，因此训练完整候选交叉拟合不可识别。新评估24state全部十四动作两seed齐全。不能利用确认标签重新调本轮模型后继续称原确认有效。

资源最初18CPU/2GPU shards；用户后授权没有其他任务时24CPU，GPU仍2shards。resource_guard.py在提交前检查全队列并按条件执行24/18CPU上限。旧critic初始化有原生Flow预训练范围，跨类型表述需保留这项限制。

典型轨迹图为仿真证据的静态读取，不是另一次实验。新评估逐候选/逐seed UID和数据包在NEW_EVAL_BUNDLE.npz，raw/traces位于new_evaluation/；共享DB官方journal仍在默认目录。
