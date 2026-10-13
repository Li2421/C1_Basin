# P3 开环静态查询固定细节

在 P2′ 新 outcome 前记录。使用与执行程序一致的 `key_for(state, seed=0, step=0)` 查询η0投影动作a0；无需调用环境step，按已知plant公式更新 `p′=p+0.05*a0`、`v′=a0`，目标不变；使用 `key_for(state,0,1)` 在该开环预测state查询η0动作与中央差分Jacobian，ε=0.001。输入把原R4_action的t0 32维和预测state的32维拼接为64维。该probe只代表一次开环动作后的局部执行响应，不能推断完整未来安全事件或闭环Q。任何静态查询异常保留为异常，不按测试结果替换probe。
