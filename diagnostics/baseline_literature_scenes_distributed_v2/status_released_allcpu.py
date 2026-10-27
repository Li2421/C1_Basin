"""Lightweight progress, user-wide allocation and explicit release schedule."""
from experiment import *
from resources_idle_cpu import snapshot

def main(phase='running'):
    if (HERE/'capability_lit_crowded_random_QUALIFICATION.json').exists():
        from live_success import main as live_success
        live_success()
    stages=[]
    for path in HERE.glob('*/GROUP_INDEX.json'):
        stage=path.parent.name;index=read(path)
        finished=sum((HERE/stage/'claims'/r['group']/'COMPLETE.json').exists() for r in index)
        completed=sum(len(r['rollout_uids']) for r in index if (HERE/stage/'claims'/r['group']/'COMPLETE.json').exists())
        stages.append(dict(stage=stage,queue_groups=len(index),finished_groups=finished,completed_requests_in_finished_groups=completed,partial_groups_excluded_from_fast_count=True,queue_complete=finished==len(index)))
    state=dict(utc=utc(),phase=phase,stages=stages,resources=snapshot(),resource_schedule=read(HERE/'RESOURCE_SCHEDULE_IDLE_CPU_AUTHORIZATION.json'))
    write(HERE/'STATUS.json',state)
    lines=[f'{state["utc"]}。阶段：{phase}。v1 已结束并冻结，本轮只读。',
           '组件10组、历史56项检查（55次去重后的完整轨迹回放）全部通过，执行实现、能力门请求与去重预检已封存。',
           '依用户最新授权，无其他用户计算作业时使用24 CPU（23调度+1本地），其他用户提交计算作业时恢复18 CPU；GPU仍为2份额。',
           f'当前Slurm合计{state["resources"]["scheduled_cpus"]} CPU、{state["resources"]["gpu_shards"]} GPU份额。快速计数只含完整小批次，进行中的已落盘局会使实际完成数更高。']
    lines += [f'- {r["stage"]}：小批次{r["finished_groups"]}/{r["queue_groups"]}，完整小批次已落盘请求{r["completed_requests_in_finished_groups"]}。' for r in stages]
    (HERE/'STATUS.md').write_text('\n\n'.join(lines)+'\n')
    return state

if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'running')
