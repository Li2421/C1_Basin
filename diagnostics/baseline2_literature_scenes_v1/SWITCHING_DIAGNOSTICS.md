# 进度切换与预算回退诊断

只读取既有位置、switch_step与封存阈值，未生成新rollout，未改切换定义。真正触发发生在动作执行前；最后一步到达后已终止而没有再次检查进度的局数单列为未切换。恰在预算点同时满足进度时记both，不猜测原因。

|批次|场景|类型|state0|局数|触发计数|初始平均进度|
|---|---|---|---|---|---|---|
|fixed_lit_crowded_random|二维拥挤随机场景|交叉熵方法模型预测控制|False|336|{'never_before_termination': 230, 'budget_fallback_only': 105, 'progress_only': 1}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|优先级时空A*规划|False|336|{'never_before_termination': 318, 'budget_fallback_only': 18}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|冲突搜索多机规划|False|336|{'never_before_termination': 316, 'budget_fallback_only': 20}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|分布式交叉熵方法模型预测控制|False|336|{'budget_fallback_only': 107, 'never_before_termination': 229}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|分布式模型预测控制|False|336|{'budget_fallback_only': 322, 'never_before_termination': 14}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|梯度下降模型预测控制|False|336|{'never_before_termination': 200, 'budget_fallback_only': 136}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|集中式模型预测路径积分控制|False|336|{'budget_fallback_only': 287, 'never_before_termination': 49}|0.0|
|fixed_lit_crowded_random|二维拥挤随机场景|集中式随机采样模型预测控制|False|336|{'budget_fallback_only': 326, 'never_before_termination': 10}|0.0|
|fixed_lit_doorway|门口场景|交叉熵方法模型预测控制|False|756|{'progress_only': 756}|0.0|
|fixed_lit_doorway|门口场景|交叉熵方法模型预测控制|True|12|{'progress_only': 12}|0.0|
|fixed_lit_doorway|门口场景|优先级时空A*规划|False|756|{'progress_only': 756}|0.0|
|fixed_lit_doorway|门口场景|优先级时空A*规划|True|12|{'progress_only': 12}|0.0|
|fixed_lit_doorway|门口场景|冲突搜索多机规划|False|756|{'progress_only': 756}|0.0|
|fixed_lit_doorway|门口场景|冲突搜索多机规划|True|12|{'progress_only': 12}|0.0|
|fixed_lit_doorway|门口场景|分布式交叉熵方法模型预测控制|False|756|{'progress_only': 756}|0.0|
|fixed_lit_doorway|门口场景|分布式交叉熵方法模型预测控制|True|12|{'progress_only': 12}|0.0|
|fixed_lit_doorway|门口场景|分布式模型预测控制|False|756|{'progress_only': 756}|0.0|
|fixed_lit_doorway|门口场景|分布式模型预测控制|True|12|{'progress_only': 12}|0.0|
|fixed_lit_doorway|门口场景|梯度下降模型预测控制|False|756|{'progress_only': 756}|0.0|
|fixed_lit_doorway|门口场景|梯度下降模型预测控制|True|12|{'progress_only': 12}|0.0|
|fixed_lit_doorway|门口场景|集中式模型预测路径积分控制|False|756|{'budget_fallback_only': 515, 'progress_only': 239, 'both': 2}|0.0|
|fixed_lit_doorway|门口场景|集中式模型预测路径积分控制|True|12|{'budget_fallback_only': 8, 'progress_only': 3, 'both': 1}|0.0|
|fixed_lit_doorway|门口场景|集中式随机采样模型预测控制|False|756|{'progress_only': 630, 'budget_fallback_only': 126}|0.0|
|fixed_lit_doorway|门口场景|集中式随机采样模型预测控制|True|12|{'progress_only': 10, 'budget_fallback_only': 2}|0.0|
|fixed_lit_narrow_passage|窄通道场景|交叉熵方法模型预测控制|False|756|{'budget_fallback_only': 652, 'never_before_termination': 104}|0.0|
|fixed_lit_narrow_passage|窄通道场景|交叉熵方法模型预测控制|True|12|{'budget_fallback_only': 7, 'progress_only': 3, 'never_before_termination': 2}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|优先级时空A*规划|False|756|{'budget_fallback_only': 168, 'never_before_termination': 588}|0.0|
|fixed_lit_narrow_passage|窄通道场景|优先级时空A*规划|True|12|{'progress_only': 6, 'never_before_termination': 6}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|冲突搜索多机规划|False|756|{'never_before_termination': 754, 'budget_fallback_only': 2}|0.0|
|fixed_lit_narrow_passage|窄通道场景|冲突搜索多机规划|True|12|{'never_before_termination': 12}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|分布式交叉熵方法模型预测控制|False|756|{'budget_fallback_only': 568, 'never_before_termination': 188}|0.0|
|fixed_lit_narrow_passage|窄通道场景|分布式交叉熵方法模型预测控制|True|12|{'never_before_termination': 2, 'progress_only': 6, 'budget_fallback_only': 4}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|分布式模型预测控制|False|756|{'budget_fallback_only': 756}|0.0|
|fixed_lit_narrow_passage|窄通道场景|分布式模型预测控制|True|12|{'budget_fallback_only': 12}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|梯度下降模型预测控制|False|756|{'never_before_termination': 372, 'budget_fallback_only': 384}|0.0|
|fixed_lit_narrow_passage|窄通道场景|梯度下降模型预测控制|True|12|{'never_before_termination': 4, 'progress_only': 2, 'budget_fallback_only': 6}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|集中式模型预测路径积分控制|False|756|{'budget_fallback_only': 756}|0.0|
|fixed_lit_narrow_passage|窄通道场景|集中式模型预测路径积分控制|True|12|{'budget_fallback_only': 12}|0.6666666666666666|
|fixed_lit_narrow_passage|窄通道场景|集中式随机采样模型预测控制|False|756|{'budget_fallback_only': 756}|0.0|
|fixed_lit_narrow_passage|窄通道场景|集中式随机采样模型预测控制|True|12|{'budget_fallback_only': 12}|0.6666666666666666|
|learning_eval_lit_doorway|门口场景|交叉熵方法模型预测控制|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|交叉熵方法模型预测控制|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|优先级时空A*规划|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|优先级时空A*规划|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|冲突搜索多机规划|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|冲突搜索多机规划|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|分布式交叉熵方法模型预测控制|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|分布式交叉熵方法模型预测控制|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|分布式模型预测控制|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|分布式模型预测控制|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|梯度下降模型预测控制|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|梯度下降模型预测控制|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|集中式模型预测路径积分控制|False|228|{'progress_only': 217, 'budget_fallback_only': 11}|0.0|
|learning_eval_lit_doorway|门口场景|集中式模型预测路径积分控制|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_doorway|门口场景|集中式随机采样模型预测控制|False|372|{'progress_only': 372}|0.0|
|learning_eval_lit_doorway|门口场景|集中式随机采样模型预测控制|True|12|{'progress_only': 12}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|交叉熵方法模型预测控制|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|交叉熵方法模型预测控制|True|12|{'never_before_termination': 12}|0.6666666666666666|
|learning_eval_lit_narrow_passage|窄通道场景|优先级时空A*规划|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|优先级时空A*规划|True|12|{'never_before_termination': 12}|0.6666666666666666|
|learning_eval_lit_narrow_passage|窄通道场景|冲突搜索多机规划|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|冲突搜索多机规划|True|12|{'never_before_termination': 12}|0.6666666666666666|
|learning_eval_lit_narrow_passage|窄通道场景|分布式交叉熵方法模型预测控制|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|分布式交叉熵方法模型预测控制|True|12|{'never_before_termination': 12}|0.6666666666666666|
|learning_eval_lit_narrow_passage|窄通道场景|分布式模型预测控制|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|分布式模型预测控制|True|12|{'never_before_termination': 12}|0.6666666666666666|
|learning_eval_lit_narrow_passage|窄通道场景|梯度下降模型预测控制|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|梯度下降模型预测控制|True|12|{'never_before_termination': 12}|0.6666666666666666|
|learning_eval_lit_narrow_passage|窄通道场景|集中式模型预测路径积分控制|False|192|{'never_before_termination': 192}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|集中式随机采样模型预测控制|False|372|{'never_before_termination': 372}|0.0|
|learning_eval_lit_narrow_passage|窄通道场景|集中式随机采样模型预测控制|True|12|{'never_before_termination': 12}|0.6666666666666666|

窄通道扰动最多sqrt(2)*.05=.07071米，小于原到达容差.08米，因此两台挡路机器人初始即在到达区。其扰动距离非零时，原clip(1−distance/d0)初始为0；到达停止并不把这个值自动设为1。仅精确d0=0时使用预注册的未定义处扩展。state0与扰动state的进度结构因此不同，阈值切换可能主要依赖6/7预算回退。保留原定义，不在观察结果后把“已在容差内”强制改为进度1。此表是解释性诊断，不作新的效果估计。
