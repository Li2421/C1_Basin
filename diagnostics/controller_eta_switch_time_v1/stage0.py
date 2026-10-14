from core import *
from collections import Counter

def main():
    prereg();p=read(R4/'protocol.json');cases=[r for r in read(R4/'CASES.json') if r['split']=='loco_test' and r['eta_index'] in (10,13)];assert len(cases)==7168;raw={};index=[];rows=[]
    for c in cases:
        f=R4/'raw'/f"{c['rollout_uid']}.json";rec=read(f);assert rec['case']==c and rec['protocol_sha256']==sha(R4/'protocol.json');raw[c['rollout_uid']]=rec;index.append(dict(path=str(f),sha256=sha(f),case=c))
    for n in p['controller_order']:
        for eta in (10,13):
            rr=[raw[c['rollout_uid']] for c in cases if c['controller_name']==n and c['eta_index']==eta];counts=Counter(r['outcome'] for r in rr);success=[r['summary']['first_success_step'] for r in rr if r['success']];failures={}
            for term in ('timeout','collision','deadlock','numerical_failure'):
                events=[]
                for r in rr:
                    if r['outcome']!=term:continue
                    if term=='collision':events.append(min(v for k,v in r['summary'].items() if k.startswith('first_') and 'collision' in k and v is not None))
                    elif term=='numerical_failure':events.append(r['error']['step'])
                    else:events.append(r['episode_length'])
                failures[term]=dict(count=len(events),event_steps=events,quantiles=np.quantile(events,[0,.25,.5,.75,1]) if events else None)
            rows.append(dict(controller=n,eta=eta,total=512,outcomes={k:int(counts[k]) for k in ('success','timeout','collision','deadlock','numerical_failure')},proportions={k:counts[k]/512 for k in ('success','timeout','collision','deadlock','numerical_failure')},success_step_quantiles=np.quantile(success,[0,.25,.5,.75,1]) if success else None,success_steps=success,failure_times=failures))
    write(HERE/'STAGE0_RESULTS.json',dict(utc=utc(),status='COMPLETE DESCRIPTIVE',new_rollouts=0,rows=rows,used_seeds=list(range(8)),main_design_changed=False))
    freeze(HERE/'HISTORICAL_RAW_INDEX.json',index)
    s='# 阶段0：已有结局的一页描述摘要\n\nDEVELOPMENT；只读7168个既有test记录（7Flow×2η×64state×seed0–7），新rollout0。描述不参与修改已预注册干预。Ring原评价器没有runtime deadlock分支；timeout保持原标签，数值失败保留unknown。\n\n| controller | η | success | timeout | collision | deadlock | unknown | success median / range step |\n|---|---|---|---|---|---|---|---|\n'
    for r in rows:
        c=r['outcomes'];q=r['success_step_quantiles'];s+='| '+r['controller'].replace('_macflow','')+' | '+str(r['eta'])+' | '+' | '.join(f'{100*c[k]/512:.2f}%' for k in ('success','timeout','collision','deadlock','numerical_failure'))+' | '+(f'{q[2]:.0f} / {q[0]:.0f}–{q[-1]:.0f}' if q is not None else 'NA')+' |\n'
    s+='\nbase_u/v10_local、v9_local及v9-v10的配对重点均已列在表中；事件比例和成功步数的不同只是描述。失败发生步数见STAGE0_RESULTS.json：timeout按700步，collision按simulator首次碰撞，numerical_failure按solver error step，三者不合并。历史记录没有轨迹，不能仅靠结局步数定位η因果效应。下一步按原设计执行32例一致性门，再做时间干预。\n'
    (HERE/'STAGE0_SUMMARY.md').write_text(s);print('Stage0 descriptive summary complete; new rollout=0')
if __name__=='__main__':main()
