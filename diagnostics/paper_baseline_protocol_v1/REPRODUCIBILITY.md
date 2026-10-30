# 复现与恢复

代码在独立分支 `research/paper-baseline-protocol-v1-20261030`，本轮提交只包含本目录。原始源码副本和五个数值执行版本的代码、manifest均版本管理；原文安全距离参照实际使用的v001适配器与当前v005都保留。数值源码身份由EXECUTION_FROZEN.json绑定，不使用可变Git HEAD作缓存身份。各选参、划分和分析冻结另外记录SHA。

完整恢复需要本工作区保留的大型只读输入、checkpoint、原始轨迹及官方SQLite/实验journal。它们未随Git提交或声称远程备份；INPUT_SNAPSHOT.json列出输入来源、目标和SHA，candidates.jsonl及plans/保留逐候选语义，import_manifests/保留实际导入文件。不能仅凭Git push认定这些数据已备份。没有删除大型证据。

恢复命令：

```bash
cd /home/zhihan/research/Basin_C1
bash diagnostics/paper_baseline_protocol_v1/resume.sh
```

入口拒绝并行启动第二个协调器。协调器先采用已存在的同阶段worker，再合并journal，执行官方preflight，补交缺失UID。已有数值结局不因结果差而重跑；基础设施补交有界。原环境复现不进入共享数据库但计入本轮50000次预算。当前用户上限18CPU，包含协调器；其他作业不撤销。每个worker单核、BLAS单线程、GPU0。

主线固定策略先于可选Oracle。Oracle为直接搜索冻结η/论文参数前16组，首次成功即停，不生成/训练Generator。非主线从已有证据提取统计或依照预登记规则补时限、旧放宽几何；缺失、预算未搜索完与有限池未找到成功分别保存。统计结果缓存绑定精确输入和算法源码SHA，改变输入不会复用旧CI/p值。

原研究目录封存只读。主仓库原有未提交改动保留，完整性审计比较排除本目录后的原始差异，不会把它们混入本轮提交。PoC目录无Git元数据的限制如实保留。
