# 同数量比较的等分布计算（覆盖标签产生前固定）

点估计严格按预注册的每个controller/state/count/repeat种子，生成200个从56个冻结有序非对角提议对中不放回抽取的子集，并保存全部下标。bootstrap的每次type/controller/state成簇重抽后，对每个state出现位置重新生成200次抽样的最大值。

为避免构造约2.4亿个不影响最大值的完整子集，bootstrap使用完全相同的概率分布：若该state有s个成功标签、每次抽k个，则子集最大值为Bernoulli(p)，p=1-C(56-s,k)/C(56,k)。200次独立子集最大值的平均严格服从Binomial(200,p)/200。unknown下界/上界共享同一个子集，故联合计数为Multinomial(200,[p_lower,p_upper-p_lower,1-p_upper])；两个端点并非独立抽样。每个bootstrap重抽的state出现位置都重新抽计数，不使用固定跨state提议序号，也不将200次抽样替换为无Monte-Carlo误差的期望。seed仍为预注册的覆盖bootstrap seed。

比较的unknown敏感性端点保守报告，并明确标注保守性；同一真实UID的普通成对差另行抵消。此计算选择在覆盖标签及模型训练前写定，不改变估计目标。
