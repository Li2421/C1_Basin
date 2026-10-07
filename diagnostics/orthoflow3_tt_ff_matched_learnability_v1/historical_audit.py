"""Evidence ledger, not a retrospective claim that every TT repair succeeded."""
from __future__ import annotations
import csv,json
import numpy as np
from .design import ROOT,MAIN,read,write,sha,freeze
OLD=MAIN/'diagnostics/orthoflow3_critic_rootcause_resolution_v1'


def main():
    rows=[];snapshot=ROOT/'legacy_completed_evidence_snapshot.json';legacy=read(OLD/'working_state.json')
    fields=('breadth_completed','cached_precision_result','goal_response_independent_confirmation',
        'motion_factorial_result','response_dispersion_result','input_ablation_complete','raw_confirmation_complete')
    selected={k:legacy[k] for k in fields}
    if snapshot.exists():assert read(snapshot)['completed_evidence']==selected
    else:freeze(snapshot,dict(original_path=str(OLD/'working_state.json'),original_sha256_at_capture=sha(OLD/'working_state.json'),completed_evidence=selected))
    def add(name,tag,scope,result,*paths):
        pp=[p if hasattr(p,'exists') else MAIN/'diagnostics'/p for p in paths]
        pp=[snapshot if p==OLD/'working_state.json' else p for p in pp]
        assert all(p.exists() for p in pp),(name,pp)
        rows.append(dict(repair=name,status=tag,evaluation_scope=scope,result=result,
            evidence_paths=' | '.join(str(p) for p in pp),evidence_sha256=' | '.join(sha(p) for p in pp)))
    add('负例/early-stop partial-count NLL','IMPROVED_NLL / NO_STABLE_GAIN','source VAL改善；严格四折LOSO',
        'FW source失败监督NLL 0.8792→0.0356；Ring VAL选定0/60（三seed0/0/6）、eta-only27/60；没有一折稳定Full>eta-only。',
        'orthoflow3_loso_partial_count_v1/final_decision.json')
    add('名义/eta-conditioned短期controller context','NO_STABLE_GAIN','严格LOSO，非source内验证',
        '已知controller-swap特征64/64可分；Ring仅1/60 vs eta-only27/60；能分辨输入不等于可泛化选择。',
        'orthoflow3_controller_context_loso_v1/final_decision.json')
    add('H20→H80','NO_STABLE_GAIN','source-controller holdout，36 fits',
        'H80−H20 NLL +0.04250（95%CI −0.01042,0.10013）；B15 +3.125pp但CI含0；未通过独立确认开启门槛。',
        OLD/'response_horizon/adjudication.json')
    add('增加controller函数families（4→12）','NO_STABLE_GAIN','两个独立controller，各64 families、K2',
        '两个target eta-only=56、32；Full=53/55/56、23/24/17；oracle61、37。增加controller数量未稳定修复。',
        OLD/'controller_function_confirmation/metrics.csv',OLD/'controller_function_confirmation_b/metrics.csv')
    add('增加state breadth（64→160 TRAIN families）','NO_STABLE_GAIN','source held-controller CV',
        'NLL 0.53864→0.54542；B15 [119,115,122]→[116,124,123]；没有一致增益。',OLD/'working_state.json',OLD/'state_breadth_training/selection_frozen.json')
    add('Q4→已有Q16 label refinement','NO_STABLE_GAIN','18 cached-data fits；source held-controller CV',
        'NLL 0.53973→0.54260；B15 [122,123,124]→[125,124,124]；幅度小、未独立确认或推广。',
        OLD/'working_state.json',OLD/'motion_cached_precision/dataset_manifest.json')
    add('structured + historical wide/off-anchor + W1','IMPROVED_NLL / IMPROVED_SELECTION','历史Toy continuous-Q TEST；不是LOSO或新K16',
        '联合数据W0→W1：NLL 0.4411→0.3997，平均B15 top1 93.66%→95.09%；structured-only W1反而弱于W0，不能归因于W1普适优势。',
        'orthoflow3_nll_weighting_ablation_v1/final_report.md','orthoflow3_nll_weighting_ablation_v1/final_decision.json')
    add('single/dual/separate rank-loss','NO_STABLE_GAIN / HARMFUL','冻结Toy unseen-state+eta、三seed',
        'K16 NLL-only B15=93.75%；single/dual=89.58%；separate ranker=87.50%。排序loss未稳定改善top1。',
        'orthoflow3_ranking_aware_critic_v1/final_report.md')
    add('5-critic ensemble、LCB、virtual-neighborhood','IMPROVED_SELECTION / NO_STABLE_GAIN','冻结Toy hard200、K16；VAL合法选择',
        '原181→ensemble183；LCB与local-score均183，未超过ensemble；8/13错误top1仍被全部成员高估>0.9。',
        'orthoflow3_toy_critic_uncertainty_local_v1/final_decision.json')
    add('goal-neighborhood controller fingerprint','IMPROVED_NLL / NO_STABLE_GAIN','两个新controller、128独立families、K2',
        'Full NLL≈0.341–0.360优于eta≈0.388–0.401；B15 [105,105,100] vs eta105、oracle113；概率增益不等于选择增益。',
        OLD/'working_state.json',OLD/'goal_response_confirmation/metrics.csv',OLD/'goal_response_confirmation_b/metrics.csv')
    add('crossed phase/motion controller intervention','INCONCLUSIVE','source交叉监督/trajectory诊断，不是新TEST',
        '晚期可学习因果contrast（相关≈0.76、skill≈0.56），但VAL选中早期step25；motion B15未胜已知program eta-prior16/32。不能称state泛化已解决。',
        OLD/'motion_factorial_support/source_selection.json',OLD/'motion_causal_fit_trajectory/audit.json',OLD/'working_state.json')
    add('matched extra-capacity + response dispersion','IMPROVED_NLL / HARMFUL','same-capacity control、source CV',
        'NLL 0.54116→0.53697；B15 [122,121,124]→[112,108,116]；新增信息没有带来选择增益。',
        OLD/'response_dispersion_training/selection_frozen.json',OLD/'working_state.json')
    add('raw context skip / encoder freeze-unfreeze','IMPROVED_SELECTION / INCONCLUSIVE','两个独立controller K2；并列旧architecture matched-step对照',
        '两个panel Full总101–103/128 vs eta79、oracle112；但旧trunk/matched-expanded同等强，未证明raw skip本身更优。多数收益来自controller适配，不是精细state-specific K16。',
        OLD/'motion_independent_confirmation_88136/metrics.csv',OLD/'motion_independent_confirmation_88137/metrics.csv')
    add('正确/错误controller、state/context shuffle','INCONCLUSIVE','上条独立K2 confirmation，输入因果诊断',
        '88136正确context55/64→错误context39–41/64；h-only shuffle不掉，joint-state/context shuffle约54/64。controller输入确实生效，但独立h增益很小。',
        OLD/'motion_independent_confirmation_88136/metrics.csv',OLD/'motion_independent_confirmation_88137/metrics.csv')
    add('agent-slot / entity-attached response control','NO_STABLE_GAIN','source-controller held-out folds，matched capacity',
        'role-zero B15[43,43,41]→role-aware[44,43,44]；response-zero[43,42,43]→entity[43,43,43]，NLL后者0.63383→0.65796。没有稳定选择修复。',
        OLD/'role_control/pooled_metrics.csv',OLD/'response_control/pooled_metrics.csv')
    add('去冗余h / H20 feature-group ablation','IMPROVED_NLL / NO_STABLE_GAIN','source CV + 已打开target回归；非新确认',
        '移除h几乎不改变B15；移除H20 source NLL0.53858→0.49504，但已打开target B15 46–48→43–45。不能只按source loss宣称输入更好。',
        OLD/'raw_input_ablation/selection_frozen.json',OLD/'minimal_response_cached_diagnostic/audit.json',OLD/'working_state.json')
    add('新controller K16 + ranking-reversal benchmark','NO_STABLE_GAIN','独立controllers88138/88139、各16families、8seen+8unseen eta',
        'K16 Full[15,15,16] vs eta15；另一个Full[6,6,11] vs eta6，oracle均16。strong reversal准确率仍约随机；unseen8子集改善不能外推到完整K16。',
        OLD/'held_controller_k16_v1/metrics_stage1.csv',OLD/'held_controller_k16_v1/stage1_eligibility.json')
    src=read(OLD/'source_joint_support_v1/source_results.json')
    vals={}
    for arm in ('matched_repeat','true_t0_base','crossed_controller'):
        rr=[r['augheld'] for r in src if r['arm']==arm and r['kind']=='full_context']
        vals[arm]={'NLL':round(float(np.mean([r['NLL'] for r in rr])),5),'B15':[r['B15'] for r in rr]}
    add('补回1440缓存true-t0×wide-eta×controller pairs','IMPROVED_NLL / INCONCLUSIVE','本轮收尾的旧TT source-only ablation；6 held-source VAL families，oracle5',
        f'{json.dumps(vals)}；same init/order/steps；不把这6个选择checkpoint的VAL families当独立确认。TT仅保留历史ablation，不继续无止境修旧范式。',
        OLD/'source_joint_support_v1/source_results.json',OLD/'source_joint_support_v1/models_frozen.json',OLD/'source_joint_support_v1/cache_audit.json')
    with (ROOT/'old_paradigm_repair_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(ROOT/'old_paradigm_repair_audit.json',dict(rows=rows,principle='Reported only completed artifacts; no plan promoted to evidence. Source VAL, opened regression, K2 independent confirmation and frozen K16 kept distinct.',
        dedicated_network_width_or_regularization_sweep='Not asserted: no standalone verified matched sweep was used in this audit; actual capacity-matched/encoder/raw-skip controls are listed.'))
    print(dict(verified_repairs=len(rows),audit=str(ROOT/'old_paradigm_repair_audit.csv')))


if __name__=='__main__':main()
