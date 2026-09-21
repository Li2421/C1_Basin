"""Summarize completed local interventions without selecting deployment actions."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.training.persistence import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    complete = json.loads((args.folder/'complete.json').read_text())
    records = json.loads((args.folder/'records.json').read_text())
    gradients = json.loads((args.folder/'gradients.json').read_text())
    protocol = json.loads((args.folder/'protocol.json').read_text())
    candidate = protocol.get('candidate','bounded')
    baseline = {r['rid']: r for r in records if r['variant']=='Safety'}
    dead = lambda r: r['any_deadlock'] or r['label']=='stalled_deadlock'
    output = dict(scope=complete['scope'], status=complete['status'], replays=len(records),
                  elapsed_seconds=complete['elapsed_seconds'], candidate=candidate, variants={})
    for name, counts in complete['variants'].items():
        group = [r for r in records if r['variant']==name]
        result = dict(counts)
        if name != 'Safety':
            risk_key = 'risk_original' if name.startswith('original') else 'risk_bounded'
            result.update(
                risk_decreased=sum(r[risk_key] < baseline[r['rid']][risk_key]-1e-12 for r in group),
                risk_decreased_but_still_deadlocked=sum(
                    r[risk_key] < baseline[r['rid']][risk_key]-1e-12 and dead(r) for r in group),
                early_action_rms_median=statistics.median(r['early_applied_delta_rms'] for r in group),
                early_action_changed_cases=sum(r['early_applied_delta_rms']>1e-8 for r in group))
        output['variants'][name.replace('bounded_',candidate+'_')] = result
    output['gradient_norms'] = {}
    for name in ('original','bounded'):
        norms = [g['gradient_norm'] for g in gradients if g['risk']==name]
        output['gradient_norms'][candidate if name=='bounded' else name] = dict(n=len(norms), exact_zero=sum(v==0 for v in norms),
                                             median=statistics.median(norms), maximum=max(norms))
    lookup = {(g['rid'],g['risk']):g['gradient_norm'] for g in gradients}
    ratios = [lookup[rid,'bounded']/lookup[rid,'original'] for rid in baseline
              if (rid,'bounded') in lookup and lookup.get((rid,'original'),0)>0]
    output['candidate_over_original_gradient_norm_median'] = statistics.median(ratios) if ratios else None
    atomic_save(args.folder/'analysis.json', output)
    rows = ['# C1 局部梯度闭环诊断', '',
            f"完成状态：{complete['status']}；{len(records)} 次回放，总用时 {complete['elapsed_seconds']/60:.1f} 分钟。", '',
            '每例分别计算自己的梯度；这是 TRAIN 中分层案例的局部干预，不能作为共享策略的泛化死锁率。', '',
            '|干预|案例数|死锁|成功|普通 timeout|原死锁转成功|新增死锁|风险下降|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in output['variants'].items():
        rows.append(f"|{name}|{r['n']}|{r['deadlocks']}|{r['successes']}|{r['ordinary_timeout']}|"
                    f"{r['deadlock_to_success']}|{r['new_deadlocks']}|{r.get('risk_decreased','—')}|")
    rows.extend(['', '梯度范数：', '', '```json', json.dumps(output['gradient_norms'],indent=2), '```', '',
        '前 5 秒动作变化见 analysis.json；它与成功同时出现不等于已经证明仅前期扰动具有因果作用。',
        '负梯度与正梯度的对照按相同步长比较，不能按每例结果挑一个动作或策略部署。',
        '正式 C1 结论仍需单个共享 G_phi、原 primal-dual 流程、验证可行选择及独立配对测试。'])
    # This is generated analysis output, not an edit to a maintained source file.
    (args.folder/'ANALYSIS_ZH.md').write_text('\n'.join(rows)+'\n')
    print(json.dumps(output, ensure_ascii=False))


if __name__=='__main__':
    main()
