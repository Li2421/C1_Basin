"""Recompute v2 collision counts and elapsed costs from immutable raw records."""
from audit import HERE,V2,sha,write
import json
from collections import defaultdict
import time

def main():
    result={}; evidence=(HERE/'EVIDENCE_INPUTS.jsonl').open('a')
    for scene in ('lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_doorway'):
        groups=defaultdict(lambda:dict(n=0,collision=0,unknown=0,success=0,seconds=0.,steps=0))
        traces=set();physical_seconds=0.;physical_count=0
        for p in (V2/f'optional_dynamics_{scene}'/'raw').glob('*.json'):
            b=p.read_bytes();digest=__import__('hashlib').sha256(b).hexdigest()
            evidence.write(json.dumps(dict(path=str(p.resolve()),sha256=digest))+'\n')
            r=json.loads(b); c=r['case'];g=groups[c['method']]
            g['n']+=1
            for k in ('collision','success'):g[k]+=r[k]
            g['unknown']+=r['numerical_failure'];g['steps']+=r['episode_length'];g['seconds']+=r['elapsed_seconds']
            key=r.get('trace_sha256') or r['rollout_uid']
            if key not in traces:
                traces.add(key);physical_seconds+=r['elapsed_seconds'];physical_count+=1
        result[scene]=dict(methods=dict(groups),unique_trace_seconds=physical_seconds,unique_traces=physical_count,
                           naive_17cpu_hours=physical_seconds/17/3600)
        evidence.flush();write(HERE/'HISTORY_RAW_COUNTS.json',result)
        print(scene,json.dumps(result[scene]),flush=True)
    evidence.close()
    hours=sum(r['naive_17cpu_hours'] for s,r in result.items() if s!='lit_doorway')
    write(HERE/'RUNTIME_FORECAST.json',dict(old_execution_sum_17cpu_hours=hours,
          limitation='Old collisions/unknown terminate early. Safer new execution may run longer; estimates are not deadlines.',
          provision_hours=[max(2,hours*1.5),max(6,hours*4)]))
    (HERE/'HISTORY_CHECK.md').write_text('''# 历史证据检查

v2 已封版：`../baseline_literature_scenes_distributed_v2/V2_COMPLETE.json`。
已核查第二阶段 DYNAMICS_IMPLEMENTATION、DYNAMICS_DIAGNOSTICS、PREREGISTRATION、CORRECTION_REVISION、HANDOFF、DYNAMICS_STAGE；原始 raw 的逐做法碰撞数、unknown、步数与运行耗时重新计算于 HISTORY_RAW_COUNTS.json，输入 sha256 逐条记在 EVIDENCE_INPUTS.jsonl。

窄通道、二维拥挤随机、圆形围场安全层发生改变，原第二阶段标签不能替代修复后的标签；计划分别 22528、39424、5760，共 67712 个请求。新身份须通过官方注册及 preflight，仅执行 genuinely_missing。单元门和试跑门是前置条件，不满足时不运行全量。

门口第二阶段 23040 请求和全部第一阶段：执行语义未变，按 v2 原始 state/controller/eta/seed 身份直接引用，不重跑。原 raw、trace、checkpoint、预测、配置与报告保留原地只读。无需 critic 训练，无需新 rollout 来计算旧结果的碰撞率。

gamma 不按研究结果调参；网格、头对头初始条件及随机快照在单元门中冻结。碰撞、约束无解与数值 unknown 分别处理；timeout 不等于死锁。
''')
if __name__=='__main__':main()
