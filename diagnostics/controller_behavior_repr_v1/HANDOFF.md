# Round7 独立Handoff

状态COMPLETE，UTC 2026-10-13T05:43:40.033473+00:00，全部 **DEVELOPMENT**。工作根目录 `/home/zhihan/research/Basin_C1`；输出 `diagnostics/controller_behavior_repr_v1/`；Python `/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python`。不依赖本对话。

目标：区分真实未见Flow的η选择迁移瓶颈是表示、锚点邻域覆盖还是学习通路。遵循AGENTS.md与冻结PREREGISTRATION。新增rollout/probe=0，DB连接/写入=0，只写新目录，无commit，纯Q无需视频。

结论：**表示（探测设计）仍不足：当前200步、两个η的R5没有建立真实Flow regime的可迁移编码**。真实四格全无SPT/CT；ρ(Δprog,ΔS)真实14Flow=-0.297，CI[-0.854,0.347]，未建立正单调信息；所有7面板Δprog>0，零参数P_rule全选η13，possible E3=5/7。安全R5规则多回退η10，在base_u/v10上相对prior收益0。学习通路瓶颈未被本轮单独识别，独立η13真实controller覆盖仍INCONCLUSIVE。I4静态配方距离预测SUPPORTED、行为regime距离预测REJECTED，均描述性。

融合LOPO次检验：R3-PCA2 SPARSE悲观/保守保留SPT，R5主/插补均无SPT。融合家族结果不推广真实Flow。所有数值和family CI见REPORT及三个*_results.json。不能用Q误差、regime分类准确率代替η选择收益；不能把Ring timeout称deadlock。

已完成：阶段0从冻结checkpoint重放15task，R0/F2κ1/保守/F4全部选择/成功与round6一致；831完整probe重算与raw统计误差0、float32特征逐值一致，唯一失败base_u_v3/state10/η13/132步保留。主删除5维得到155、delta删整个state10用15state；敏感性仅阶段1/3源均值插补。阶段1I1–I4全部主/敏感性、散点/混淆/距离CSV。105真实+30融合主+30插补seed fit=165，82个task×3seed预测及checkpoint齐全，SPARSE共享模型不重训。F4及P_rule参数冻结，终点只真实seed4–7、融合seed2–3，unknown配对保守且同η对消。

历史事实差异：round6真实R3均值实际128TRAINstate、融合96，非提示概述中的全部96。真实R3 baseline精确复用128；增广保留真实128、融合96，state组成差异为局限。预注册先已记录。用户随后将资源更新为共同18CPU/3GPU、本轮优先；资源补充收据先于新终点评价。真实GPU、融合CPU，x64false和科学超参数保持。历史prompt仅资源段在外部更新，唯一hash差异已解释；3067个其余历史资产不变。原审计保留，最终PASS_WITH_DOCUMENTED_RESOURCE_UPDATE。

预注册UTC 2026-10-13T05:09:44.888555+00:00，sha256 `05314f4387a577ea32b5d5c9d4f5e2156f1bb01ec1eceea3c85d569c65029463`。MODEL_FREEZE在新终点评价之前；FINAL_AUDIT验证hash、无泄漏、165fit、155复用行与η直方图、unknown算术。全共享DB未作字节不变声明；无DB连接。

下一步：先从已有raw源TRAIN内预注册新probe统计，0新rollout。若候选冻结，4个新controller（至少2个候选η13真实Flow）/全新state前瞻：384发现完整episode+16,384终点完整episode+128短probe=16,896请求，未执行。原R5每C6,400步；round5 k1每C32完整H700且需要目标成功标签。报告直接保留round5少样本校准表，其比较对象是3个必要融合controller，source prior78.65%，SPARSE R0 78.65%、DENSE R0 78.82%，不是7真实Flow。

继续工作先读REPORT.md、PREREGISTRATION.md及资源补充、FINAL_AUDIT.json。结果入口results.json；阶段文件、checkpoints索引、figures/PNG-PDF、tables/CSV、运行/代码均在本目录。REPORT末尾有只从冻结缓存复算命令；不得运行历史rollout/probe/merge，也不得修改旧资产或用终点调参。边界Ring/FF/H700/K16固定几何、真实必要C仅2；所有结论DEVELOPMENT。
