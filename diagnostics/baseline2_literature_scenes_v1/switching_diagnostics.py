"""Read stored traces to distinguish progress and budget fallback switches."""
from experiment import *
from closure_scope import FLOW_NAMES
from pilot_results import resolve
import numpy as np
import collections
import csv

def progress(p,g,d0,tol):
    dist=np.linalg.norm(g-p,axis=-1);values=np.empty_like(dist);moving=d0>0
    values[moving]=np.clip(1-dist[moving]/d0[moving],0,1);values[~moving]=(dist[~moving]<=tol).astype(float)
    return float(values.mean())

def main():
    rows=[];groups=collections.defaultdict(list)
    stages=[p.name[:-len('_CASES.json')] for p in HERE.glob('*_CASES.json') if not p.name.endswith('_MISSING_CASES.json') and p.name.startswith(('fixed_','pilot_fixed_','learning_eval_'))]
    for stage in stages:
        raw,_=resolve(stage)
        for r in raw:
            c=r['case']
            if c['theta'] is None:continue
            s=c['state'];cfg=c['scenario_payload']['config'];g=np.asarray(s['goals']);p0=np.asarray(s['positions']);d0=np.linalg.norm(g-p0,axis=-1);fallback=round(6*cfg['max_steps']/7);step=r['switch_step'];value=None
            if step is None:trigger='never_before_termination'
            else:
                with np.load(r['trace_path']) as z:value=progress(z['positions'][step],g,d0,cfg['goal_tolerance'])
                byprogress=value>=c['theta'];bybudget=step>=fallback;assert byprogress or bybudget,(stage,c['rollout_uid'],step,value)
                trigger='both' if byprogress and bybudget else 'progress_only' if byprogress else 'budget_fallback_only'
            ctl=c['controller_payload']['nominal'];row=dict(stage=stage,scene=FLOW_NAMES[s['scene']],scene_code=s['scene'],state=s['index'],N=s['N'],controller=f"{ctl['type']}（{ctl['controller']}）",type=ctl['type'],method=c['method'],seed=c['seed'],theta=c['theta'],initial_exact_zero_goal_distance=int((d0==0).sum()),initially_within_arrival_tolerance=int((d0<=cfg['goal_tolerance']).sum()),initial_progress=progress(p0,g,d0,cfg['goal_tolerance']),switch_step=step,progress_at_switch=value,budget_fallback_step=fallback,trigger=trigger,success=r['success'],unknown=r['numerical_failure'],rollout_uid=r['rollout_uid']);rows.append(row);groups[(stage,s['scene'],ctl['type'],s['index']==0)].append(row)
    with (HERE/'SWITCHING_DIAGNOSTICS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summaries=[]
    for (stage,scene,typ,state0),rr in sorted(groups.items()):summaries.append(dict(stage=stage,scene=scene,type=typ,state0=state0,episodes=len(rr),trigger_counts=dict(collections.Counter(r['trigger'] for r in rr)),mean_initial_progress=float(np.mean([r['initial_progress'] for r in rr]))))
    write(HERE/'SWITCHING_DIAGNOSTICS.json',summaries)
    text=['# 进度切换与预算回退诊断','','只读取既有位置、switch_step与封存阈值，未生成新rollout，未改切换定义。真正触发发生在动作执行前；最后一步到达后已终止而没有再次检查进度的局数单列为未切换。恰在预算点同时满足进度时记both，不猜测原因。','','|批次|场景|类型|state0|局数|触发计数|初始平均进度|','|---|---|---|---|---|---|---|']
    for r in summaries:text.append('|'+ '|'.join(map(str,[r['stage'],FLOW_NAMES[r['scene']],r['type'],r['state0'],r['episodes'],r['trigger_counts'],r['mean_initial_progress']]))+'|')
    text+=['','窄通道扰动最多sqrt(2)*.05=.07071米，小于原到达容差.08米，因此两台挡路机器人初始即在到达区。其扰动距离非零时，原clip(1−distance/d0)初始为0；到达停止并不把这个值自动设为1。仅精确d0=0时使用预注册的未定义处扩展。state0与扰动state的进度结构因此不同，阈值切换可能主要依赖6/7预算回退。保留原定义，不在观察结果后把“已在容差内”强制改为进度1。此表是解释性诊断，不作新的效果估计。']
    (HERE/'SWITCHING_DIAGNOSTICS.md').write_text('\n'.join(text)+'\n')
    seal('SWITCHING_DIAGNOSTICS.receipt.json',[HERE/'SWITCHING_DIAGNOSTICS.csv',HERE/'SWITCHING_DIAGNOSTICS.json',HERE/'SWITCHING_DIAGNOSTICS.md',HERE/'switching_diagnostics.py'],predictions_only=True,no_switch_rule_change=True)

if __name__=='__main__':main()
