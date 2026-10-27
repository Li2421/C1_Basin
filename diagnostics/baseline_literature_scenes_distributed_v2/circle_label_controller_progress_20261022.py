"""Read sealed preflight and saved raw only; no science runs or database writes."""
from experiment import *
from atomic_write import write
from resources_corrections_release_20261023 import snapshot
import collections


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    stage='learning_labels_revised_20261021_lit_circle_five'
    cases=read(HERE/f'{stage}_CASES.json');missing=read(HERE/f'{stage}_MISSING_CASES.json')
    missing_ids={c['rollout_uid'] for c in missing}
    nominal={c['rollout_uid']:c['controller_payload']['nominal'] for c in cases}
    rows={}
    for c in cases:
        n=c['controller_payload']['nominal'];name=n['controller']
        r=rows.setdefault(name,dict(type=n['type_code'],controller=name,planned=0,exact_reuse_at_preflight=0,new_raw_landed=0,remaining_new=0,active_missing_cases=0))
        r['planned']+=1
        if c['rollout_uid'] not in missing_ids:r['exact_reuse_at_preflight']+=1
        elif (HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists():r['new_raw_landed']+=1
        else:r['remaining_new']+=1
    state=snapshot();live={j['job_id'] for j in state['jobs']}
    for claim in (HERE/stage/'fine_claims').iterdir():
        owner=claim/'OWNER.json'
        if not claim.is_dir() or (claim/'COMPLETE.json').exists() or not owner.exists():continue
        o=read(owner)
        if str(o['job']) not in live:continue
        for rid in o['rollout_uids']:
            if (HERE/stage/'raw'/(rid+'.json')).exists():continue
            rows[nominal[rid]['controller']]['active_missing_cases']+=1
    values=sorted(rows.values(),key=lambda r:(r['type'],r['controller']))
    for r in values:r['completed_logical']=r['exact_reuse_at_preflight']+r['new_raw_landed'];assert r['completed_logical']+r['remaining_new']==r['planned']
    out=dict(utc=utc(),stage=stage,scope='16 training controllers;32 states and15 actions each; active incomplete episodes are not counted as complete',controllers=values,total={key:sum(r[key] for r in values) for key in ('planned','exact_reuse_at_preflight','new_raw_landed','completed_logical','remaining_new','active_missing_cases')})
    write(HERE/'LIVE_CIRCLE_LABEL_CONTROLLER_PROGRESS_20261022.json',out)
    lines=['# 圆形围场修订学习标签进度','',out['utc'],'','每个训练实例480个标签请求；已复用结果与新落盘结果分列，正在运行的局不算完成。','','| 实例 | 精确复用 | 新落盘 | 剩余新局 | 正在运行 |','|---|---:|---:|---:|---:|']
    lines += [f"| {r['controller']} | {r['exact_reuse_at_preflight']} | {r['new_raw_landed']} | {r['remaining_new']} | {r['active_missing_cases']} |" for r in values]
    (HERE/'LIVE_CIRCLE_LABEL_CONTROLLER_PROGRESS_20261022.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
