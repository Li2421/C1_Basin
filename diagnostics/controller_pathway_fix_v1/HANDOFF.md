# 独立Handoff — Controller pathway fix round6

任务COMPLETE，全部证据 **DEVELOPMENT**；2026-10-13T03:47:40.335143+00:00 UTC。本文件不依赖对话。

研究问题：高维controller输入与controller数稀少是否导致critic对未见controller外推、选择被高估η，以及降维、收缩、保守选择或两段式regime是否能消除伤害并获得迁移收益。

实际实施：复用round4真实Flow LOCO和round5连续Flow融合LOPO全部兼容标签/静态特征/冻结checkpoint，重放38任务；物理state23维+controller填充513、16-head MLP，源-onlyPCA/normalization/regime，源P1 NLL选step，fit seeds17/23/41，unknown保留、sameη对消，5000次配对物理statebootstrap。六个F1/F2重训arm和F4/F3均执行；真实LOCO只重训源E1/E3门合格架构。oracle-low原始留出Flow需未知regime，标不适用，禁止目标标签泄漏。

复现：argmax/success全一致；辅助logit门14/38未过，max.0016608，原FAIL记录与补充协议保留，F3用原cache避免后端微小数值改变规则。新增rollout **0**，DB写 **0**；复用197,632标签槽位（含unknown）；新独立MLP fit 174、预测task 76。

核心结论：修复能够消除伤害，按预注册操作映射支持H_method。融合族SPARSE的有界残差+悲观/保守规则有正收益；真实Flow尚无同时满足安全与正收益的规则。高维过参数化是否为唯一根因仍INCONCLUSIVE。 融合操作判定 **SUPPORTED: operational repair eliminates harm**；真实 **SUPPORTED: operational repair eliminates harm**。

融合族SPARSE：F2-PCA2+κ=1悲观选择，在3个必要controller上相对源prior及R0的保守收益为+7.81pp（上界+8.33pp），描述性95% CI [1.56,14.06]pp；E1同时CI下界−1.48pp超过−2pp非劣界，E3=0/12。F2-PCA2+保守偏离收益+4.69pp，CI [1.04,8.33]pp，E1下界−1.04pp，E3=0/12。分别闭合target固定η差距55.6%和33.3%；因2次bootstrap分母非正，融合比例CI记NA。

真实Flow：F2-PCA2+保守偏离在2个必要controller上相对源prior提高+8.79pp，描述性CI [4.30,13.48]pp；相对R0提高+7.49pp，CI [3.06,12.11]pp。但在非必要的v9_local上相对源prior损失−10.16pp，E3=1/7，故不能判为安全正迁移。该收益来自v10_local（+17.58pp），base_u为0，两个η13型controller的不对称仍在。B1+保守偏离、F1-PCA1/2/3+保守偏离、F4可满足安全门，但在必要controller上与源prior完全相同（E2=0）。因此尚没有已验证“既安全又有效”的真实Flow修复，不应只看平均收益。

机制诊断：融合DENSE的旧R3/oracle被选η的平均预测高估上界分别+5.17/+12.99pp，凸包距离与相对R0损失的描述性Pearson为.647/.975；有界残差PCA2 argmax的对应高估为−6.57pp。降维/收缩大幅减少误选η1/η8；请求头数本身平衡，不能将错误简单归为这些head没标签。相关性不是因果，各目标明细及solver状态见D1–D3机器结果。

| 测试床 | 安全修复规则数（含敏感性） | 安全修复涉及架构 |
| --- | --- | --- |
| 融合 | 18 | B2, F1_PCA1, F1_PCA2, F1_PCA3, F1_oracle_low, F2_PCA2, F2_oracle_low, F4 |
| 真实 | 5 | B1, F1_PCA1, F1_PCA2, F1_PCA3, F4 |

| 测试床 | 主要正迁移候选（DEVELOPMENT） | κ=.5敏感性候选 |
| --- | --- | --- |
| 融合 | [{'coverage': 'SPARSE', 'method': 'F2_PCA2__pessimistic1'}, {'coverage': 'SPARSE', 'method': 'F2_PCA2__conservative'}] | [{'coverage': 'DENSE', 'method': 'F2_oracle_low__pessimistic0.5'}, {'coverage': 'SPARSE', 'method': 'F2_PCA2__pessimistic0.5'}] |
| 真实 | [] | [] |

| 假设 | 证据判定 |
| --- | --- |
| H_method_harm | SUPPORTED: operational repair eliminates harm |
| H_positive_blend_transfer | SUPPORTED DEVELOPMENT |
| H_positive_real_Flow_transfer | INCONCLUSIVE: no primary eligible arm established positive real transfer |
| H_high_dimension_is_unique_rootcause | INCONCLUSIVE: repair interventions combine representation, shrinkage, uncertainty and fallback; no unique causal identification |
| H_statewise_generalization | INCONCLUSIVE: not a prospective state-selection confirmation |

结论边界：安全回退并不证明controller→Q已学会，正迁移候选要求必要controller同时优于源prior与R0；候选仍需前瞻确认。低维表示改变有效输入维数与descriptor的state变化，残差同时改变容量/收缩/惩罚，3fit分歧不是校准不确定性；不能唯一归因高维过参数化。coverage/representation与NLL目标偏移仍可能限制正迁移。固定Ring/FF/H700/K16，不能外推跨geometry、连续basin或任意真实Flow。

证据：/home/zhihan/research/Basin_C1/diagnostics/controller_pathway_fix_v1/REPORT.md完整表格；results.json全结果；blend_results.json/real_results.json含每controller E1/E2/E3、D1–D3、bootstrap与unknown边界；BACKGROUND_FACT_AUDIT.json历史数字；HISTORICAL_ASSET_INDEX.json数据hash；PREREGISTRATION*.md及UTC/hash收据；STAGE0_REPRODUCTION.json；STAGE2_ELIGIBILITY.json条件分支；fits/与shared_sparse_fits/逐seedcheckpoint/logits；logistic/分类参数；figures/24张PNG/PDF；FINAL_AUDIT.json最终审计。

最终全库hash审计最初FAIL，原记录保留为FINAL_AUDIT_INITIAL_FAIL.json。只读调查发现本轮期间Drone3D独立任务导入66条source行标签，并登记其独立experiment；本目录experiment/source行均0。DB_CONCURRENT_ACTIVITY.json保留基线/当前hash及外部来源。无法证明共享DB每个字节变化的完整归属，故全库不变性标NOT VERIFIED；本轮实际使用的737个历史NPZ/checkpoint/代码文件哈希全一致，新增脚本不写DB，最后调查仅以mode=ro、immutable、query_only执行SELECT。最终审计以限定范围PASS_WITH_NOTES交付，不能伪称共享DB全局未变化。

可复算：使用REPORT G节命令，evaluate.py从缓存重算、不生成rollout。未经冻结的新policy不能在已打开终点上选出后称确认。后续最有价值的是对候选做新controller/newstate冻结验证；如无正候选，先源数据内审计regime识别与selection校准，避免仅扩大controller标签。前瞻最小设计见REPORT F节，未在本轮运行。
