# Round30 补充预注册（第三次运行）

UTC: 2026-10-15T06:49:38.300273+00:00

原PREREGISTRATION.md及收据保持字节不变，原副本保存在attempt2_gate_tolerance/。第一次归档保持不动；第二次全部产物原样移入attempt2_gate_tolerance/，见ATTEMPT2_ARCHIVE.json。尚未训练学生或读取学生结果。

唯一修改：按当前任务第二节，取消调用nominal(traced=True)的FP64容差断言；在新目录以原解码方式独立检查trace终点与native nominal的最大绝对差≤1e-5 m/s，并报告最大值、99.9分位。噪声逐位一致、native经相同to_world与再限速后的动作逐位一致两项不变。仍为原计划14教师各前72个train rollout第0状态，共1,008样本，相同key。协议x64=False，原代码与内置断言均不改。上一轮约1e-8差异与FP32精度相容，本次修改由任务明确指定，与学生闭环结果无关。

原训练数据/80,000步/batch256/验证8批/14seed/观测/动作空间/能力门/评估/统计判据全部保持原预注册不变。若任一新门失败立即停止，不以其他样本替换。本补充仅替代原预注册中对traced=True内置断言的要求。
