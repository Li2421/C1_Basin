"""Retrospective exact parsing of saved Slurm GRES strings; no runtime edits."""
from common import *
import re
def actual_gpu(gres):
    result=0
    for token in gres.split(','):
        token=token.removeprefix('gres/')
        if re.match(r'^shard(?::[^:]+)?:\d+$',token):result+=int(token.split(':')[-1])
        elif re.match(r'^gpu(?::[^:]+)?:\d+$',token):result+=8*int(token.split(':')[-1])
    return result
def main():
    rows=[];mismatches=[]
    snapshot=HERE/'RESOURCE_CHECKS_AUDIT_SNAPSHOT.jsonl';snapshot.write_bytes((HERE/'RESOURCE_CHECKS.jsonl').read_bytes())
    for line in snapshot.read_text().splitlines():
        r=json.loads(line);cpu=gpu=0
        for record in r['rows']:
            if record['counts']:cpu+=record['cpu'];gpu+=actual_gpu(next(x.split('|')[4] for x in r['raw'].splitlines() if x.split('|')[0]==record['job']))
        ok=cpu+r['requested_cpu']<=18 and gpu+r['requested_gpu']<=2
        row=dict(utc=r['utc'],cpu_before=cpu,gpu_before=gpu,requested_cpu=r['requested_cpu'],requested_gpu=r['requested_gpu'],admitted=r['admitted'],within_limit=ok,original_gpu_count=r['gpu_shards'])
        rows.append(row)
        if gpu!=r['gpu_shards']:mismatches.append(row)
    assert all(r['within_limit'] for r in rows if r['admitted'])
    assert actual_gpu('gres/shard:1')==1 and actual_gpu('shard:2')==2 and actual_gpu('gres/gpu:a6000:1')==8
    write(HERE/'RESOURCE_ACCOUNTING_AUDIT.json',dict(utc=utc(),passed=True,checks=rows,recorded_gpu_count_mismatches=mismatches,all_observed_admission_events_within18cpu2gpu=True,reason='Original parser accepts shard: but omitted gres/shard: prefix; corrected read-only audit uses saved raw squeue strings',no_frozen_execution_or_model_file_modified=True,remaining_gpu_work=False))
    (HERE/'RESOURCE_ACCOUNTING_AUDIT.md').write_text('资源记录复核：原资源统计器遗漏Slurm的gres/shard:前缀，导致一条保存的排队GPU占用显示为0。按每次提交前保存的原始squeue文本重新解析后，全部提交时实际占用加请求均在18 CPU/2 GPU内，没有发现超限。GPU训练和推理已完成，剩余阶段仅用CPU。为保持冻结模型与执行代码的SHA不变，本轮未改动冻结统计器；下一轮应使用修正后的解析函数。此项只影响资源记录，不依赖或改变任何锁定类型结果。\n')
    seal('RESOURCE_ACCOUNTING_AUDIT.receipt.json',[HERE/'RESOURCE_ACCOUNTING_AUDIT.json',HERE/'RESOURCE_ACCOUNTING_AUDIT.md',HERE/'resource_accounting_audit.py',snapshot],no_model_or_execution_semantics_changed=True)
    print('Resource GRES audit passed;',len(rows),'admissions;',len(mismatches),'recorded-prefix mismatches',flush=True)
if __name__=='__main__':main()
