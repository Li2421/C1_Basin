"""Once final main numbers exist, verify the early complete-scope outcomes."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
import time


def main():
    verify_receipt('COMPLETED_SCENE_NUMBERS.receipt.json')
    verify_receipt('OUTCOME_SNAPSHOT_COMPARISON_CODE.receipt.json')
    while not (HERE/'MIGRATION_NUMBERS.receipt.json').exists():time.sleep(30)
    verify_receipt('MIGRATION_NUMBERS.receipt.json')
    early=read(HERE/'COMPLETED_SCENE_RESULTS.json')['results']
    final=read(HERE/'MIGRATION_RESULTS.json')['results'];issues=[];rows=0
    keys=('success_rate','ci95','success_upper','upper_ci95','controllers',
          'delta_vs_safety','delta_vs_fixed','delta_vs_best_aggregate_fixed')
    for scene,entry in early.items():
        for scope in ('fixed_full','learned_evaluation'):
            if entry[scope].get('status')!='complete':continue
            index={(r['type'],r['layer'],r['method']):r for r in final[scene][scope]['statistics']}
            for old in entry[scope]['statistics']:
                key=(old['type'],old['layer'],old['method']);new=index[key];rows+=1
                for field in keys:
                    if old.get(field)!=new.get(field):issues.append(dict(scene=scene,scope=scope,key=key,field=field,early=old.get(field),final=new.get(field)))
    result=dict(utc=utc(),passed=not issues,compared_rows=rows,issues=issues,
        early_receipt_sha256=sha(HERE/'COMPLETED_SCENE_NUMBERS.receipt.json'),
        final_receipt_sha256=sha(HERE/'MIGRATION_NUMBERS.receipt.json'),new_rollouts=0)
    write(HERE/'OUTCOME_SNAPSHOT_COMPARISON.json',result)
    if issues:
        write(HERE/'OUTCOME_SNAPSHOT_COMPARISON_ERROR.json',result)
        raise AssertionError('Review early outcome snapshot against authoritative final main numbers')
    seal('OUTCOME_SNAPSHOT_COMPARISON.receipt.json',[HERE/'OUTCOME_SNAPSHOT_COMPARISON.json',HERE/'compare_outcome_snapshot.py'],passed=True,comparison='Exact numerical equality on shared complete scopes')


if __name__=='__main__':main()
