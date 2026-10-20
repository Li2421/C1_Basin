"""Close the monitor with supplemental review, preserving all prior receipts."""
from common import *
import subprocess

def main():
    assert read(HERE/'FINAL_VERIFICATION.json')['passed']
    assert (HERE/'SUPPLEMENTARY_DEVELOPMENT.receipt.json').exists()
    confirmed=read(HERE/'CONFIRMATORY_RESULTS.json');summary=confirmed['endpoints']['combined'];missing=read(HERE/'UNSEEN_FEATURE_GAPS.json');history=read(HERE/'SUPPLEMENTARY_DEVELOPMENT.json');comparison=history['original_nine_types_regression']['type_equal']['paired_comparisons']['pipeline__minus__legacy_pipeline']
    lines=['# 最终监视与交付复核\n','完整性审计通过：保护源码和历史输入均无改动。主评估、确认性终点、事后诊断、图表和收据核查全部完成。结论仍是：仅部分未见类型获得修正收益，不能宣称普遍泛化。\n','主报告见[REPORT.md](REPORT.md)，逐类型表见[AXIS_TABLE.md](AXIS_TABLE.md)，诊断见[FAILURE_ATTRIBUTION.md](FAILURE_ATTRIBUTION.md)。本补充复核只解释已有证据，不改变已封存的确认结果、模型、候选或主方法。\n','## 合并口径与逐类型结论\n','主报告终点1的+30.56pp [19.44,41.75]为“需要修正的类型”合并，即Transformer行为克隆与冲突搜索两个类型。梯度下降模型预测控制在能力门中六实例均不需要修正，按预设规则不计入该修正有效性合并。三个未见类型全部合并时，主方法84.72%、只加安全投影65.05%，差+19.68pp [12.15,27.55]。两种合并口径都不能掩盖冲突搜索上的负收益。\n','| 未见类型 | 只加安全投影 | 主方法 | 配对差与95% CI |','| --- | ---: | ---: | --- |','| Transformer行为克隆策略 | 0.00% | 68.75% | +68.75pp [47.74,89.06] |','| 梯度下降模型预测控制 | 99.65% | 97.57% | −2.08pp [−4.86,1.39] |','| 冲突搜索多机规划 | 95.49% | 87.85% | −7.64pp [−13.89,−0.69] |','\n主方法比只加安全投影低至少10pp的实例有2个，均为冲突搜索实例。能力门标为不需要修正的全部实例上，主方法−安全投影为−4.17pp [−8.16,0.87]；这不能表述为已经证明“不伤害”。\n','未见类型三类等权合并：相对较好人工时间表−8.56pp [−16.49,−1.74]，未达到−1pp非劣界限；相对整局单η+9.38pp [4.63,15.39]；相对仅学习型训练的旧冻结pipeline−1.39pp [−5.79,3.01]；自学切换时刻相对固定500步+0.06pp [−2.31,2.55]；相对不看控制器的critic+0.81pp [−0.98,2.66]。因此尚无明确证据支持增加训练族、学习切换时刻或控制器特征在这些未见类型上带来额外收益。\n','## 诊断与证据缺口\n','诊断26动作样本内最佳：Transformer87.50%、梯度控制100%、冲突搜索100%；相同8状态seed0上的主方法分别75.00%、95.83%、87.50%。主方法与最佳26动作的差CI均触及0，不符合“CI不含0才算明显”的预设归因标准。可以报告候选中存在成功动作和选择损失的点估计，但不能把原因⑤写成已证实；也不能据此断言双η修法整体无效或生成器一定缺乏覆盖。\n',f'另有{len(missing)}个未见实例的机制探测因原安全投影的CBFSolverError不完整，24维向量保留NaN：'+ '、'.join(r['controller'] for r in missing)+'。它们没有重跑，推理沿用仅由源类型计算的填充值。历史另有5个缺失向量，两者分别记录于[UNSEEN_FEATURE_GAPS.json](UNSEEN_FEATURE_GAPS.json)和[FEATURE_REUSE_CHECK.json](FEATURE_REUSE_CHECK.json)。unknown保留上下界，超时不直接称死锁。\n','## 开发证据补充\n',f"原九类、62实例、同48训练状态和同26动作的配对回归：新方法−旧跨类型1c为{comparison['gain_pp']:+.2f}pp，95% CI {comparison['CI95_pp']}。未观察到明确下降或改善；这不是新状态确认。原九类不混入保守强化学习，详细数值与三族留一描述检验见[SUPPLEMENTARY_DEVELOPMENT.md](SUPPLEMENTARY_DEVELOPMENT.md)。\n",'## 工程与资源复核\n','原交付绘图因指定venv无Matplotlib失败，保留原脚本与错误日志，随后使用上一轮已采用的纯SVG绘图原语生成66张科学图并全部XML解析通过。没有安装包、修改冻结代码、重新训练或补跑rollout。来源SHA和实际绘图脚本见SVG_HELPERS_SOURCE.json、plots_svg.py及FIGURES.receipt.json。\n','资源记录解析器遗漏一条gres/shard:格式的排队GPU计数。按每次提交保存的原始squeue重算，全部已观察到的提交仍在18 CPU/2 GPU内，未发现实际超限；见[RESOURCE_ACCOUNTING_AUDIT.md](RESOURCE_ACCOUNTING_AUDIT.md)。用户随后允许5个GPU时，本轮GPU训练和推理已经完成，此授权没有引入新训练或新评估。冻结解析器保持不变，下一轮应使用修正后的解析函数。\n','原协调器在绘图失败后停止；交付由delivery_svg_recovery作业继续并完成。旧错误记录是保留的历史现场，实际完成状态以FINAL_VERIFICATION和本复核收据为准。没有更改任何已封存结果或旧收据。']
    (HERE/'FINAL_REVIEW.md').write_text('\n'.join(lines)+'\n')
    links=0;digests={};fail=[]
    for receipt in sorted(HERE.rglob('*.receipt.json')):
        for path,h in read(receipt).get('files',{}).items():
            links+=1
            if path not in digests:digests[path]=sha(path)
            if digests[path]!=h:fail.append(dict(receipt=str(receipt),path=path))
    assert not fail,fail[:10]
    assert sha(HERE/'CONFIRMATORY_RESULTS.json')==read(HERE/'FAILURE_ATTRIBUTION.json')['confirmation_sha256']
    write(HERE/'MONITOR_COMPLETION.json',dict(utc=utc(),complete=True,receipt_file_links_verified=links,unique_files_hashed=len(digests),main_evaluation_completed=30084,diagnostic_genuinely_missing_completed=3545,diagnostic_cached=199,training_missing_completed=44736,static_scientific_figures=66,integrity_passed=True,confirmation_unchanged_after_diagnostics=True,supplementary_development_complete=True,resource_audit_passed=True,no_frozen_source_modified_during_monitoring=True,coordinator_delivery_error_resolved_by_job='26212',review='FINAL_REVIEW.md'))
    seal('FINAL_REVIEW.receipt.json',[HERE/n for n in ('FINAL_REVIEW.md','MONITOR_COMPLETION.json','final_review.py','FINAL_VERIFICATION.receipt.json','DELIVERY.receipt.json','SUPPLEMENTARY_DEVELOPMENT.receipt.json','RESOURCE_ACCOUNTING_AUDIT.receipt.json','GPU_AUTHORIZATION_5.receipt.json')],all_requested_stages_complete=True)
    print('Monitor review complete;',links,'receipt links valid',flush=True)

if __name__=='__main__':main()
