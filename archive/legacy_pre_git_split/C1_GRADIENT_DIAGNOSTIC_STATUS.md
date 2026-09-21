# 小规模闭环梯度诊断：2026-09-17

最新进展：Slurm GPU 作业已成功验证 JAX `CudaDevice(id=0)`，诊断已作为 Job 19 启动。下文权限阻塞为历史诊断；现已确认管理员有意要求 CUDA 只能经 Slurm 使用，因此无需、更不应为本实验修改登录会话的设备白名单。

用户要求每 15 分钟检查一次。`scripts/monitor_c1_gradient_job.py` 已以独立后台进程启动，每 900 秒记录作业与进度；诊断完成后自动运行 `scripts/analyze_c1_gradient_closed_loop.py`，生成 `analysis.json` 和 `ANALYSIS_ZH.md`。当前阶段是逐例梯度诊断，不是共享 G_phi 的正式训练。该监控只检查并汇总，不会自动启动未经结果审视的训练方案。

用户已重新授权使用 GPU，仅进行约 400 条轨迹量级的诊断，不恢复历史大场景队列。

## 已准备的实验

`scripts/probe_c1_gradient_closed_loop.py` 固定选取旧 TRAIN calibration 中按编号排序的 20 个死锁与 20 个非死锁初始状态；两组不重复初始状态并交错执行。不是按人口分布抽样，不能将其中的死锁比例当作部署死锁率。

每例执行 Safety，以及原 sum 风险、新 bounded union 风险各 4 个干预：负梯度方向的三个尺度、正梯度方向的一个对照，共 360 次真实闭环回放。扰动作用于原 G_phi 参数，不增加行动规则、路径候选选择或新控制模块。尺度按前 5 秒同状态 residual JVP 的每动作分量 RMS 归一化至 0.001、0.005、0.01 m/s，参数位移范数上限 0.1；正向对照使用 0.005。实际施加动作的差异另外记录。

新 bounded union 的固定参数为 delta=0.5、eta=0.01、tau=0.02。此处 eta 只是诊断设置，不能据此声称训练约束 epsilon=0.01 有可用误差余量。深死锁梯度可能饱和；零梯度原样记录，不用启发式方向替代。

真实环境重新执行两次硬投影与原事件判定，保存每条 trace、前 5 秒动作差异、风险变化、死锁转成功、死锁转普通 timeout、以及新引入的死锁。回放累计预算 1800 秒，达到预算停止；梯度计算时间单列。这些是逐例拟合的局部诊断，不是共享 phi 的完整 C1 训练，也不能证明独立测试有效。前期动作变化与最终成功同时出现，也不能单独证明只有早期动作导致了成功。

## 本轮验证与实际阻塞

- 两个新脚本/模块的语法检查通过。
- `test_c1_reachability_margin` 和 `test_c1_reachability_union` 共 8 项 CPU 测试通过：事件上界、terminal timeout 门控、窗口长度、padding 不变性及局部导数等。
- 未更改历史训练、硬投影、环境和 checkpoint 的源代码。
- **尚未启动 GPU 梯度实验或 360 条回放，没有新增闭环改善结果。**

`nvidia-smi` 能看到 RTX PRO 6000，但 CUDA Driver API `cuInit(0)` 返回 999，JAX 回退到 CPU。独立 `os.open('/dev/nvidia-uvm', O_RDWR)` 返回 EPERM；nvidia0 与 nvidiactl 可打开。

`systemctl show user-1004.slice` 显示 DevicePolicy=closed，现有设备白名单未包含 nvidia-uvm。相关配置为 `/etc/systemd/system/user-.slice.d/50-gpu-lockdown.conf`，另有 `/run/systemd/system.control/user-1004.slice.d/` 的运行时配置。普通用户的 systemd service 中同样无法 cuInit，所以更换启动方式不能解决此账号的权限问题。

管理员需要为该用户追加 `DeviceAllow=char-nvidia-uvm rw`，保留其它必要设备规则，并使配置生效。当前 sudo 要求密码；本轮未修改管理员设备限制、未重载驱动、未重置 GPU。

权限恢复后先验证：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-c1/bin/python -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())'
```

随后可用新结果目录执行已准备脚本：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-c1/bin/python scripts/probe_c1_gradient_closed_loop.py --out results/c1_gradient_closed_loop_v1 --cases 40 --replay-budget-seconds 1800
```

脚本拒绝 CPU 回退，也拒绝覆盖已有输出目录。旧大型训练/测试队列继续保持停止。
