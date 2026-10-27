"""Reporting annotations; never consulted by execution or model selection."""
LABELS={
 'lit_narrow_passage':'论文证据未复现：右手规则活锁未观察到，真实t0未知',
 'lit_crowded_random':'论文证据未完整复现：官方参数差异、unknown及部分不一致',
 'lit_circle_five':'论文证据未完整复现：部分收敛/卡死代理一致，去中心化unknown较多',
 'lit_doorway':'单实例定性证据复现；保留官方活性代码及时间语义差异',
 'pooled_scenes':'含论文证据未复现或仅部分复现场景；流程版独立证据'}
