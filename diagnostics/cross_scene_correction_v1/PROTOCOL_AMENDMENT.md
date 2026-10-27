# 用户修订（2026-10-24，优先于原说明）

原说明只读来源：docs/prompt_cross_scene_correction_v1_20261024.md。甲乙丙丁判定、样本量、49候选、执行语义与资源上限不变。initial_stop/ 保留首次停止报告，STOP.receipt.json保留历史事件；本轮现已恢复。

1. 单进程准许官方 rollout_db.connect/transaction 只 INSERT 本轮新的 experiment/scenario/state/controller_config/eta；同UID内容不一致即停；登记前后原有行逐表哈希一致。rollout仍只走默认官方journal和官方单进程merger。
2. 旧Gap1排序相关记 N/A（无共同eta），不补跑、不近似配对；旧数据只用地图与物理状态。
3. 窄通道、拥挤随机保留 centralized_mppi0 共8类；圆形与三个八机7类。场景内控制器类型等权。
4. 完整性结束时剔除本轮目录新增的一条status；其余原有status、两仓git diff哈希与全部审计输入哈希须一致。
5. 在本轮目录建立隔离执行副本；临时和缓存路径指向本轮；原生CBS仅改缓存根目录并重新编译。记录全部来源SHA，逐位重放通过才能使用。

从第四节第1步继续，不覆盖或改写原说明、旧源码与旧证据。
