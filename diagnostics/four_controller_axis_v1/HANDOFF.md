# Round19 handoff

阶段 0 等价门未通过：G1 的历史记录没有首动作字节，无法完成规定的逐位比较；G0/G2 在此硬停止后未运行。Four 是否存在跨 controller 的 η 泛化问题为 INCONCLUSIVE；H_compress_Four 与由无标签探测学习 E 均为 INCONCLUSIVE（未进入相应阶段）。

工作目录 `/home/zhihan/research/Basin_C1`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。本轮文件仅在 `diagnostics/four_controller_axis_v1/`。
先读预注册及收据，再读 `G1_EVIDENCE_AUDIT.json` 和 `GATE_REPORT.json`。`audit_g1_evidence.py` 只读共享 DB 与历史 journal；`finalize_stop.py` 生成停止报告。
G1 不能被同代码重新计算的首动作补足；需要独立保存的历史首动作字节或全程轨迹，或者另行预注册不同的验证门。当前授权下不得越过此门启动阶段 1–4。
旧 v12/v10-wide intervention 采用 canonical v13 t0 再切到替代 Flow 的 conditioning，不能直接当作本轮纯替代 Flow 的成功标签。旧 round14B 的 v13 G0 8/8 通过，但 Round19 新适配器未建立。
视频交付未完成；Four 不具备可用的正式 deadlock 判据，timeout 不可代替死锁证据。
