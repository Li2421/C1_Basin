# 历史证据检查

第一部分已完成，COMPLETE SHA256=763112f66a88441896833801afaec302d60823a504024fa12892a00eab5df7da，0新研究重跑；其双积分器轨迹不能回答本轮SI η筛选。

v2单积分器分布式baseline、修订fixed、修订15动作training labels按完整物理state/controller/ηfloat64/schedule/seed匹配。方法名和critic编码不改变物理执行语义；已有unknown保留并作为失败下界，不能重试。旧controller参数/代码、state拆分、case fingerprint保留。

V2的旧原修正语义非零η与本轮contact-gated执行不兼容；只加安全投影可兼容。集中式/双积分器不同不复用。新swap8几何/局部感知/移动障碍不存在旧证据，必须新能力门与rollout，不训练新controller或critic。

已发现27477个兼容历史语义键（只索引本轮可能用到的训练/评估state）；实际复用须官方DB preflight精确UID及raw/trace SHA核验。HISTORY_SOURCES列9份case清单及SHA，原raw在首次使用时另记SHA。

lit_narrow_passage：前16训练state；评估32state，选中8/8训练类型。缺类型[]。

lit_crowded_random：前16训练state；评估32state，选中8/8训练类型。缺类型[]。

lit_circle_five：前16训练state；评估16state，选中7/8训练类型。缺类型['centralized_mppi']。

圆形MPPI的v2训练实例0/1均不合格，严格照原训练controller范围无法提供第8类；不从评估controller补入，7类筛选为明确协议缺口，不冒充要求的8类完整比较。其余独立工作继续。
