# Gap 1 规模试验目录

顶层四个 Slurm 脚本是最终 N=2/10/50 数据生成与训练配置；N=50 分成 CPU 数据和
依赖其成功完成的 GPU 训练任务。`pilots/` 保留早期失败配置的脚本，供核对
development 决策，不应误当作最终结果运行。实验输出（dataset、checkpoint、
trace、视频、日志）只在本地保存，普通 Git 提交不含这些大文件。

评价口径、结果、视频与剩余限制见
[规模试验报告](../../docs/gap_flow_scaling_20261007.md)。
