"""Report complete research cohorts only; pilot outcomes never fill main cells."""
from experiment import *
from audit import V2
from stats_kernel import bootstrap,contrast
from collections import Counter
import numpy as np

METHODS=('safety','fixed_eta10','fixed_eta13','human_progress')
LABELS=('只加安全投影','全程固定 η10 修正','全程固定 η13 修正','人定双 η 广播进度切换')

def evidence(p):
    with (HERE/'EVIDENCE_INPUTS.jsonl').open('a') as f:f.write(canonical(dict(path=str(p),sha256=sha(p)))+'\n')
    return read(p)

def row(raw,diag=None):
    c=raw['case'];s=c['state'];ctl=c['controller_payload']['nominal'];d=diag or {};cats=d.get('local',[])
    return dict(controller=ctl['controller'],type=ctl['type_code'],N=s['N'],state=s['index'],method=c['method'],seed=c['seed'],
        success=raw['success'],upper=min(1,raw['success']+raw['numerical_failure']),unknown=raw['numerical_failure'],collision=raw['collision'],timeout=raw['timeout'],
        deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats),wall_episode=int('wall_filter_equilibrium' in cats),
        canonical_joint_episode=int('filter_equilibrium_deadlock' in d.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in d.get('actual_product_joint',[])),
        direct_stationary_episode=int(any(d.get('direct_stationary',[]))),direct_stationary_agent_fraction=sum(d.get('direct_stationary',[]))/s['N'],
        constraint_violation_steps=raw.get('constraint_violation_steps',0),sampled_speed_violation_steps=raw.get('sampled_speed_violation_steps',0),
        acceleration_bound_violation_steps=raw.get('acceleration_bound_violation_steps',0),diagnostics_available=diag is not None,rollout_uid=c['rollout_uid'])

def paired_statistics(current,old,first):
    rows={'fixed':current,'v2':old,'first':first};types={r['controller']:r['type'] for r in current};ctls=sorted(types)
    states=sorted({(r['N'],r['state']) for r in current});arms=[(stage,m) for stage in rows for m in METHODS]
    lookup={(stage,r['controller'],r['N'],r['state'],r['seed'],r['method']):r for stage,rr in rows.items() for r in rr}
    M=np.full((len(ctls),len(states),2*len(arms)),np.nan)
    for i,ctl in enumerate(ctls):
        for j,(n,s) in enumerate(states):
            for k,(stage,method) in enumerate(arms):
                rr=[lookup[(stage,ctl,n,s,seed,method)] for seed in (0,1)]
                M[i,j,2*k]=np.mean([r['success'] for r in rr]);M[i,j,2*k+1]=np.mean([r['upper'] for r in rr])
    point,draws=bootstrap(M,np.array([types[c] for c in ctls]));out={}
    for m in METHODS:
        k=2*arms.index(('fixed',m));d=dict(success=float(point[k]),success_ci95=np.quantile(draws[:,k],[.025,.975]).tolist(),success_upper=float(point[k+1]),
             versus_v2=contrast(point,draws,arms,{('fixed',m):1,('v2',m):-1}),
             versus_first=contrast(point,draws,arms,{('fixed',m):1,('first',m):-1}))
        if m!='safety':d['gain_vs_first_difference']=contrast(point,draws,arms,{('fixed',m):1,('fixed','safety'):-1,('first',m):-1,('first','safety'):1})
        rr=[r for r in current if r['method']==m]
        d['counts']={k:sum(r[k] for r in rr) for k in ('success','unknown','collision','timeout','constraint_violation_steps','sampled_speed_violation_steps','acceleration_bound_violation_steps')}
        rates={}
        for metric in ('deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode','direct_stationary_episode','direct_stationary_agent_fraction'):
            rates[metric]=float(np.mean([np.mean([np.mean([r[metric] for r in rr if r['controller']==ctl]) for ctl in ctls if types[ctl]==kind]) for kind in sorted(set(types.values()))]))
        d['rates']=rates;out[m]=d
    return dict(methods=out,bootstrap_iterations=5000,controllers=ctls,states=states)

def main():
    selection=read(HERE/'FINAL_SELECTION.json');stage=selection.get('accepted_stage');results={};text=[]
    text.append('第二阶段修复开发证据。未通过单元门或试跑门的场景不填入全量研究结果，不把未运行请求当unknown；门口原身份复用。')
    original=(V2/'DYNAMICS_STAGE.md').read_text()
    counts=read(HERE/'HISTORY_RAW_COUNTS.json')
    text+=['','| 场景 | 状态 | 完成 | 计划 | unknown | 约束违反步 | 采样速度违反步 | 加速度上限违反步 | 诊断缺失局 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for scene in ('lit_circle_five','lit_crowded_random','lit_doorway','lit_narrow_passage'):
        if scene=='lit_doorway':
            text.append('| 门口场景 | 原身份复用 v2 | 23040 | 23040 | 5 | 0 | 0 | 0 | 0 |')
            results[scene]=dict(status='EXACT_REUSE',source=str(V2/'DYNAMICS_STAGE.md'),source_sha256=sha(V2/'DYNAMICS_STAGE.md'),requests=23040)
        elif scene=='lit_circle_five' and stage:
            cases=read(HERE/f'{stage}_CASES.json');rr=[];old=[]
            for c in cases:
                raw=read(HERE/stage/'raw'/f"{c['rollout_uid']}.json");dp=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json"
                rr.append(row(raw,read(dp) if dp.exists() else None))
                old.append(row(evidence(V2/'optional_dynamics_lit_circle_five'/'raw'/f"{c['paired_v2_rollout_uid']}.json")))
            first=evidence(V2/'COMPACT_REVISED_MAIN_RESULTS_20261021.json')
            first=[r for r in first if r['scene']==scene and r['layer']=='distributed' and r['method'] in METHODS]
            stats=paired_statistics(rr,old,first);results[scene]=dict(status='complete',stage=stage,statistics=stats)
            totals={k:sum(r[k] for r in rr) for k in ('unknown','constraint_violation_steps','sampled_speed_violation_steps','acceleration_bound_violation_steps')}
            text.append(f"| 圆形围场五机随机场景 | complete | {len(rr)} | 5760 | {totals['unknown']} | {totals['constraint_violation_steps']} | {totals['sampled_speed_violation_steps']} | {totals['acceleration_bound_violation_steps']} | {sum(not r['diagnostics_available'] for r in rr)} |")
            write(HERE/'COMPACT_FIXED_DYNAMICS_RESULTS.json',rr)
        else:
            n=5760 if scene=='lit_circle_five' else 39424 if scene=='lit_crowded_random' else 22528
            reason=selection.get('reason','试跑/回退门未通过') if scene=='lit_circle_five' else '随机快照单元门未通过'
            text.append(f'| {NAMES[scene]} | N/A：{reason} | 0（全量研究） | {n} | N/A | N/A | N/A | N/A | N/A |')
            results[scene]=dict(status='GAP',reason=reason,planned_requests=n,full_cohort_completed=0)
    for scene in ('lit_circle_five','lit_crowded_random','lit_doorway','lit_narrow_passage'):
        text+=['',f'### {NAMES[scene]}','']
        if scene=='lit_doorway':
            section=original.split('### 门口场景',1)[1].split('### 窄通道场景',1)[0]
            text.append(section.strip());text.append('\n与v2第二阶段差值为0（逐条原身份复用，非新实验）；修复前后碰撞均为0/23040。')
        elif results[scene]['status']=='complete':
            ss=results[scene]['statistics']['methods']
            text+=['| 做法 | 成功率 [95% CI] | unknown 成功上界 |','|---|---|---|']
            for m,label in zip(METHODS,LABELS):
                s=ss[m];ci=s['success_ci95'];text.append(f"| {label} | {100*s['success']:.2f}% [{100*ci[0]:.2f}, {100*ci[1]:.2f}] | {100*s['success_upper']:.2f}% |")
            text+=['','| 做法 | 第二阶段−第一阶段 | 修正增益之差 | 与v2第二阶段配对差 |','|---|---|---|---|']
            def fmt(d):return f"{100*d['delta']:+.2f} pp [{100*d['ci95'][0]:.2f}, {100*d['ci95'][1]:.2f}]；unknown识别界 [{100*d['identified_bounds'][0]:.2f},{100*d['identified_bounds'][1]:.2f}]"
            for m,label in zip(METHODS,LABELS):
                s=ss[m];text.append(f"| {label} | {fmt(s['versus_first'])} | {fmt(s['gain_vs_first_difference']) if m!='safety' else 'N/A'} | {fmt(s['versus_v2'])} |")
            text+=['','| 做法 | 逐机器人4a′ | 墙前4a″ | 耦合联合参照 | 实际乘积联合锥 | 直接静止 | 直接静止agent fraction |','|---|---|---|---|---|---|---|']
            for m,label in zip(METHODS,LABELS):
                rates=ss[m]['rates'];text.append('| '+label+' | '+' | '.join(f'{100*rates[k]:.2f}%' for k in ('deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode','direct_stationary_episode','direct_stationary_agent_fraction'))+' |')
        else:text.append('全量四种做法、与第一阶段及v2第二阶段的配对差、5000次bootstrap、死锁比例均为N/A：'+results[scene]['reason']+'。未完成的试跑不替代研究终点。')
    text+=['','### 碰撞局数（v2完整队列与本轮试跑分别列出）','','| 场景/范围 | 做法 | 碰撞局数 |','|---|---|---|']
    for scene,r in counts.items():
        for m,label in zip(METHODS,LABELS):
            g=r['methods'][m];text.append(f"| {NAMES[scene]} / v2第二阶段全量 | {label} | {g['collision']}/{g['n']} |")
    for p in HERE.glob('lit_circle_five_*_PILOT_RESULTS.json'):
        r=read(p);text.append(f"| 圆形 / {r['stage']}试跑 | 只加安全投影与全程eta13 | {r['collisions']}/{r['completed']}；unknown {r['unknown']} |")
    write(HERE/'DYNAMICS_RESULTS.json',results)
    (HERE/'DYNAMICS_STAGE.md').write_text('\n'.join(text)+'\n')
if __name__=='__main__':main()
