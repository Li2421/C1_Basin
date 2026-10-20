"""Once-only full-precision source decision after engineering gates."""
from common import *
from schedule_c6 import THRESHOLDS,TRIGGERS

def main():
    assert read(HERE/'EXECUTION_REPRODUCTION.json')['passed']
    assert read(HERE/'EXECUTION_EQUIVALENCE.json')['passed']
    p=D/'crosstype5_repair_gate_late_eta_v1/COVERAGE_RESULTS.json'
    comparison=read(p)['groups']['类型等权合并']['comparisons']
    write(HERE/'ETA_SOURCE_RAW_COMPARISONS.json',dict(source_path=str(p),source_sha256=sha(p),comparisons=comparison))
    enabled={k:v['CI95'][0]>0 for k,v in comparison.items()}
    assert enabled['时间表生成器提议对相对同数量冻结提议对'] is False
    assert enabled['切换时重查相对同数量冻结提议对'] is True
    timings={}
    for path in (HERE/'reproduction/raw').glob('*.json'):
        r=read(path);timings.setdefault(r['case']['family'],[]).append(r['elapsed_seconds'])
    estimated=dict(source='16 mandatory bitwise C5 replays, original C5 identities and physical semantics',seconds_per_episode_by_type={k:float(np.mean(v)) for k,v in timings.items()},CPU_wall_hours_before_reuse_at18=[103456*3/18/3600,103456*10/18/3600],limitation='3–10s observed episode range extrapolated; model training, queue, implementation and audits additional; update after probe throughput, not a promised deadline')
    write(HERE/'ETA_SOURCE_DECISION.json',dict(utc=utc(),frozen_proposals=True,schedule_generator=False,switch_requery=True,full_precision_decision=enabled,source_path=str(p),source_sha256=sha(p),once_only=True,C5_frozen_reference_preserves_all_original_pools=True))
    write(HERE/'OPENING_CONFIG.json',dict(utc=utc(),thresholds=THRESHOLDS,triggers=TRIGGERS,training_seed=2026102061,new_state_seed=2026102062,equivalence_seed=2026102063,bootstrap_seed=2026102064,bootstrap_draws=5000,training_request_cap=52704,probe_request_cap=3904,evaluation_request_cap=46848,implementation_rerun_cap=8000,all_types='DEVELOPMENT',resources=dict(CPU=18,GPU=2),eta_sources=['initial_frozen8_ordered_pairs','isolated_requery_at_actual_late_switch'],raw_group_definition_receipt='PROBE_FEATURE_DEFINITION.receipt.json',time_estimate=estimated))
    (HERE/'OPENING_RECORD.md').write_text('# 跨类型6开工记录\n\n完整重读10月20日修订，前置条件与16条复现、72条新增执行验证均通过。全部19类型为开发证据。\n\n按 COVERAGE_RESULTS.json 未舍入值：时间表生成器提议对CI下界 −9.877447916666664，不启用；切换时重查下界 +0.0020312500000123812，启用；冻结8提议有序对始终保留。本轮不重训时间表生成器，跨类型5冻结参照仍保留其原全部提议池。\n\n网格、抽样/新state种子、分组与预算见OPENING_CONFIG.json、PREREGISTRATION.md与PROBE_FEATURE_DEFINITION.json；各阶段精确preflight后再封存实际上限。按本次C5复现实测单条约3–10秒估算，未复用10.4万条在18 CPU下约4.8–16小时纯闭环；训练、排队、实现、审计另计。这个范围会按探测真实吞吐更新。\n')
    seal('OPENING_RECORD.receipt.json',[HERE/n for n in ('ETA_SOURCE_RAW_COMPARISONS.json','ETA_SOURCE_DECISION.json','OPENING_CONFIG.json','OPENING_RECORD.md','PREREGISTRATION.receipt.json','PROBE_FEATURE_DEFINITION.receipt.json','EXECUTION_EQUIVALENCE.receipt.json')],before_new_labels=True,source_decision_never_retuned=True)
    print('η source sealed: frozen pairs + switch-time requery; no schedule-generator retraining',flush=True)

if __name__=='__main__':main()
