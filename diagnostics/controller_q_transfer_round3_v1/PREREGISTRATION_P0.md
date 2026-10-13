# P0 设计校准预注册

登记在本轮新增拟合、重算开发集结果之前。全部 P0 结果属于 DEVELOPMENT：256 个测试 state 曾被历史研究打开过。

数据：原生 Ring v10，自由 t0、FF、H700、K16。源 512 state×16 η×4 seed；验证 32 state×16 η×16 seed；开发 256 state×16 η×16 seed。开发 seed 0–7 定最佳固定 η，8–15 评估。数值未知保留上下界。

P0a：总标签预算 2048 个 state–η 格子，分别取 128×16、256×8、512×4。随机种子 2026101201/02/03 决定各自三个子集与每 state 的 η 分配。16-head、physical23、64×64 SiLU；只对实际有标签的 head 求 Bernoulli NLL。η 抽样须保证每个 head 有基本覆盖；不能使用开发/验证标签决定 η 子集。三个优化 seed 17/23/41；AdamW 0.0008、wd 0.0001、clip5、batch128、最多12000步；所有配置共用训练步数网格 50、100、200 及其后每200步。输出每个子集/fit 的预测、checkpoint 与完整验证史。

P0b：比较按验证 NLL 与验证 selected-success 选共同 step；并列报告两规则，不按开发结果反选。selected-success 用验证集经验成功概率，平分 argmax 时取该规则对应的验证固定 η prior 中最高者，再按 η 索引。主选择规则优先在验证集上对 3 个 fit 的平均 selected-success 最大化；相同则取验证 NLL 最小的最早 step。NLL 主规则是3fit均值最小的最早step。优化不按测试信息改变。

主要指标：开发 seed8–15 上 top-1 成功率，减去全部16个固定 η 的最强者；按物理 state 成组 bootstrap5000 次，bootstrap中每次重估16个固定 η 的最强成功率，报告同时区间。依次报告平均3子集×3fit，以及子集/fit敏感性、未知区间、oracle regret、Q MSE辅助。P0预算优选：首先要求同时 CI 下界>0；若多个满足，选成功率点估计最高的等预算方案；点估计差 <1pp 时优先覆盖更多 state。P0b 在所选P0a预算上冻结；若两规则差<1pp且CI跨0，沿用NLL，避免不稳的选择目标。所有选择只在开发阶段，正式P2必须独立确认。

复用判断：历史N1完整16-head checkpoint可作128×16对照，但缺少对应随机三子集与稀疏η训练；因此只补必要的新拟合，旧rollout全部复用。研究直接回答Controller训练数据分配问题；P0自身固定Controller，不构成跨Controller证据。
