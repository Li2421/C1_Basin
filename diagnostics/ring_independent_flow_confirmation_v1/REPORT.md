# Round13b independent Ring Flow confirmation

C1 **INCONCLUSIVE**、C3 **INCONCLUSIVE**；C2 的 headroom 门 **未评估**。训练复现门失败，按预注册和任务要求停止。没有训练 14 个新 seed controller，没有生成新 Flow 的 R5 探测或成功标签，也没有运行 21,504 条 rollout。整批交付未完成。

## 训练复现门

原配方 `base_u_v2_macflow`，原 seed 3104、12,000 步、batch 256、同一 dataset manifest SHA256 `b49dcafe393b4f5f4103cad8e633c589d0dd8ee08200d52ce512a62941a57d1b`。重训写于 `reproduction_gate/base_u_v2_macflow/`。原 run 与本次的 best fixed dev loss 分别为 0.129786164、0.121099007，相对误差 **6.693%**；final fixed dev loss 分别为 0.129786164、0.124628182，相对误差 **3.974%**。两项均超过预设 1% 上限。

原 best 参数 SHA256 `42c447321b5dab25cca8fefc9ef76384aaa2cefa6ecb74f05a011a5d2172ee74`，重训 best `8f1b2837be2e0136d773e7d6400b739099788fb2929f89e5a3274b678a3e6630`；原 final 参数 SHA256 `42c447321b5dab25cca8fefc9ef76384aaa2cefa6ecb74f05a011a5d2172ee74`，重训 final `7a800fafc49a2e642299a9227a824bf483f792e15f6ed5e3c75d5c8815744707`。原 best step 12000，重训 best step 11750。按参数叶子的路径、形状、dtype、字节计算 SHA256；完整检查点文件 SHA 和逐共同 step 的 dev 曲线在 `TRAINING_REPRODUCTION_GATE.json`。两条曲线从 step 0 已有差异。训练代码或环境、数据内容中哪一项导致差异尚未确定；不能据此声称新 checkpoint 与历史 run 同配方可比。

## 历史证据与执行边界

预注册 `PREREGISTRATION.md` 和收据先于本轮训练写入；14 个原训练配置、种子、数据集 manifest 路径与 hash 在 `TRAINING_PLAN.json`。round12 `fp32_n8` 的冻结 R5/Ê/G_E、32 例门、64 state 方法及最终数值结果可作流程来源，但其目标仅是同 run final 兄弟，不能回答独立训练问题。原始旧 rollout 标签不因 critic 输入编码变化而失效；本轮目标 checkpoint 尚不存在，无可复用的目标成功标签。未触发新 rollout，故无 DB preflight、journal、merge 或视频入口可执行。

## 状态

- C1：**INCONCLUSIVE**，无独立训练目标的成功率数据。
- C1b：**INCONCLUSIVE**，无遗憾数据。
- C2：**INCONCLUSIVE**，真实 E 与 headroom 未产生。
- C3：**INCONCLUSIVE**，目标 R5 与 E 未产生。
- C4：**INCONCLUSIVE**，描述性符号统计不可计算。

未生成任何新的成功标签；冻结时序因此未被打开。资源上仅用一个本地 CPU 训练进程，未提交 Slurm 作业或占用 GPU。应先定位并解决复现偏差，形成新的门报告及有时间戳的补充预注册，再考虑独立 seed 训练。当前不宣称 C1/C3 通过或失败。
