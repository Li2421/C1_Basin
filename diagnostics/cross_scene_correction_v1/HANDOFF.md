# 自动续跑说明

任务依据只读原说明 docs/prompt_cross_scene_correction_v1_20261024.md 与本目录 PROTOCOL_AMENDMENT.md。甲乙丙丁不变，全部写入限制不变。

advance.py 正在运行，依次完成 capability → pilot → main → 模型冻结 → extrapolation → statistics → postflight → deliver。各阶段只执行官方预检 genuinely_missing；已完成raw、unknown和确定性种子引用保留，不重跑。COMPLETE.json 只在最后数据库、轨迹、完整性核对与中文报告全部通过后生成。

查看：PROGRESS.json、SCHEDULER_EVENTS.jsonl、advance.log、REPORT.md。异常写ADVANCE_ERROR.json；不要把原始STOP.receipt.json（用户修订前的历史停止）当作当前停止信号。

资源：用户已授权用满其他任务不使用的CPU；当前17个CPU worker+1个编排/审计份额，无GPU任务。worker约3小时，结束前55分钟不再领取新组。调度器检查实际RUNNING和三分钟状态、15分钟排队/30分钟无新结果；仅对真实CPU/内存缺口让出，不为缺GPU份额让出。

历史文献六场景26304请求已在官方库中，8192请求确实缺失（只在窄通道/拥挤随机各4096）。其中先核验26048个raw与15.7GB轨迹，另256安全基线已单独核验。新场景标签仍须经过能力门和pilot。

固定图数量扫描已修正为所选N2源状态的episode_seed，N2→N4→N6起点及目标前缀逐位一致。旧设计1条能力门结果保留但不用于新计划。原有10110条兼容能力门结果精确复用，修订计划剩16258个缺失请求，见TASK_PREFIX_CORRECTION.json。

恢复命令（只在advance没有运行且尚无COMPLETE.json时）：

    /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/cross_scene_correction_v1/advance.py

不修改旧源码、docs、其他diagnostics、数据集、环境或共享库源码；不执行git写命令。旧Gap标签不进入本轮，排序相关固定N/A。完成交付在REPORT.md及COMPLETE.json，运行中不宣称假设判定已完成。
