# 留出失败队列事后补查

旧实验 `../scene_types_basis_dimension_v1/` 已封存，只读复用。本轮不训练 Generator/Critic，不用 GPU。

历史审计：原四列 G/R/T1/C2 的 Oracle@16 在120个 controller×初态案例中成功77、失败43。43例各已有121个不同线性参数配置的旧 rollout，其中11例已被其他旧线性池补救，32例在这些有限池中仍失败。不得将它们解释为连续参数空间不可解。

唯一新增证据缺口：同场景类型、同 nominal 的开发初态0..3中，确实成功、非零支持不超过5、但尚未作用于这32例的离散配置。逐个保留原 recipe、角色规则、η、额外列系数；共56个 case×配置。候选来自原开发集，不利用测试集成功标签拟合参数。由于失败队列选择已看过原留出结果，本轮属于事后救援筛选，不能用于更新原77/120、维度推荐或声称新增独立泛化验证。

执行保持原 scenario/state、nominal、动力学、两次原安全投影、原成功/碰撞/超时判据、规划预算、episode horizon 和 matched physical RNG。固定系数 ADD，不增加调度。runtime、routes、加速数值模块逐字节复制；guard/core/runner/dispatch 仅改变输出命名空间、缓存记录和本轮预算。新代码哈希产生新 controller UID，明确记录这种工程迁移；事先对参数配置去重，不借命名空间重复旧 rollout。

复用旧200局逐位加速认证，另回放64条旧轨迹的初步动作，逐字节比较 nominal、第一次投影、最终动作、route guide 和 remaining；没有 env.step，没有新 rollout 标签。先跑56项中的两个完整 smoke（各 nominal 一个），然后只跑其余缺失 UID；保留 unknown，不自动重跑完整异常结果。

所有请求先登记官方数据库并 preflight；worker 单CPU/BLAS单线程写 journal，单协调者合并且读回。CPU用户总限额18（09:00已释放6个），GPU0。局部硬上限256次尝试；加旧36832仍低于原45000预算。最多三个基础设施补交，只补缺失 UID。trace/raw 全保留。

正式论文对比仅包括五篇指定论文的六项机制：Wang CP、Wang完整消解、Chen IMPC-OB、Zhang风险自适应、Zhang DAC、Lee MGR。三/四/五列和启发式参照属于内部消融。直接复用已完成的论文端口60初态结果，不重复执行。端口异常、碰撞和规划信息差异必须与成功率一并披露，不能据此做原论文能力排名。

恢复：`PYTHONDONTWRITEBYTECODE=1 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python diagnostics/scene_types_remaining_oracle_v2_20261028/screen.py`。恢复前查自身 `str_` 作业，防止两个协调者同时运行。
