"""Matched baseline summaries and exact-cache verification; reads completed evidence."""
import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from analyze import metric,avg,ci
BASE=Path(__file__).resolve().parent;OUT=BASE/'baselines'

def load(p):return json.load(open(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def recovery_proxy(rs):
    delays=[];caps=[];events=0
    for r in rs:
        tr=np.load(r['trace']);d=np.linalg.norm(tr['goals'][None]-tr['positions'],axis=-1);dt=r['horizon_time']/plans_horizon(r)
        for t in range(40,len(d)):
            active=(d[t-40]-d[t]<.05)&(d[t]>.08)
            if not active.any():continue
            events+=1;later=np.flatnonzero(np.all((d[t:,active]<=d[t,active]-.05)|(d[t:,active]<=.08),axis=1))
            if len(later):delay=float(later[0]*dt);delays.append(delay);caps.append(delay)
            else:caps.append(max(0.,r['horizon_time']-t*dt))
            break
    return dict(diagnostic_stall_episodes=events,diagnostic_recovered=len(delays),diagnostic_recovery_conditional_time=avg(delays),diagnostic_recovery_capped_time=avg(caps))

def plans_horizon(r):
    # All audited frozen scene families use dt=.05; recover horizon steps explicitly.
    return round(r['horizon_time']/.05)

def main():
    import sys
    sys.path.insert(0,str(BASE.parents[1]))
    from shared_rollout_db.src import rollout_db as db
    rr=list(map(json.loads,open(OUT/'candidates.jsonl')));by={r['rollout_uid']:r for r in rr};assert len(by)==len(rr)
    old={r['rollout_uid']:r for r in map(json.loads,open(BASE/'candidates.jsonl'))};labels=load(OUT/'LABELS.json');plans={}
    for p in OUT.glob('*_planned.json'):
        for c in load(p)['exact_configs']:plans[c['rollout_uid']]=c
    assert set(plans)==set(by)
    with db.connect(True)as con:
        for uid,r in by.items():
            c=plans[uid];assert c['spec']==r['spec']
            assert uid==db.uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=c['seed_key']))
            assert c['controller_uid']==db.uid('ctl',dict(scenario=c['scenario_uid'],payload=c['controller_payload']))
            for name,h in c['controller_payload']['baseline_sources'].items():assert h==sha(BASE/name)
            q=con.execute('SELECT success,collision,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(uid,)).fetchone()
            assert q and not q['conflict_quarantined']and all(q[k]==r[k]for k in ('success','collision','numerical_failure'))
            assert Path(r['trace']).exists()
    metrics={};byscene={};bycontroller={};all_data={}
    for split,filename in [('test','TEST_FIXED_LABELS.json'),('confirm','CONFIRMATORY_FIXED_LABELS.json')]:
        oldlabels=load(BASE/filename)
        data={name:[old[i]for i in ids]for name,ids in oldlabels.items()}
        data.update({name:[by[i]for i in ids]for name,ids in labels[split].items()})
        all_data[split]=data;bb=data['B3_frozen'];metrics[split]={}
        for name,rs in data.items():
            assert len(rs)==16
            for r,b in zip(sorted(rs,key=lambda r:(r['scene'],r['controller'],r['state_index'])),sorted(bb,key=lambda r:(r['scene'],r['controller'],r['state_index']))):
                assert (r['state_uid'],r['seed_key'])==(b['state_uid'],b['seed_key'])
            m=metric(rs,bb);m.update(recovery_proxy(rs));m['solver_failure_episodes']=sum(r.get('baseline',{}).get('clf_failures',0)>0 for r in rs)
            m['solver_failure_problems']=sum(r.get('baseline',{}).get('clf_failures',0)for r in rs)
            m['clf_problems']=sum(r.get('baseline',{}).get('clf_problems',0)for r in rs)
            m['local_progress_conditional_time']=avg([r.get('local_progress_time')for r in rs if r.get('local_progress_time')is not None])
            # Actual stopped/group recovery time cannot be inferred from initial-goal progress proxy.
            st=[r['baseline']for r in rs if r.get('baseline')];calls=sum(x['calls']for x in st)
            m['baseline_kernel_us']=1e6*sum(x['seconds']for x in st)/calls if calls else None
            m['roundabout_created']=sum(x['roundabout_created']for x in st);m['roundabout_invalid_attempts']=sum(x['roundabout_invalid']for x in st)
            m['clf_reproject']=sum(x['clf_reproject']for x in st);m['clf_max_reproject']=max([x['clf_max_reproject']for x in st],default=0.)
            metrics[split][name]=m
        byscene[split]={sc:{name:metric([r for r in rs if r['scene']==sc])for name,rs in data.items()}for sc in sorted({r['scene']for r in bb})}
        bycontroller[split]={ctl:{name:metric([r for r in rs if r['controller']==ctl])for name,rs in data.items()}for ctl in sorted({r['controller']for r in bb})}
    summary=dict(records=len(rr),new_rollouts=sum(r['source']=='new_rollout'for r in rr),outcomes=dict(Counter(r['outcome']for r in rr)),
       numerical_failures=sum(r['numerical_failure']for r in rr),worker_seconds=sum(r['elapsed_seconds']for r in rr),
       frozen_enhanced=load(OUT/'ENHANCED_FROZEN.json'),fixed=metrics,by_scene=byscene,by_controller=bycontroller,
       verification=dict(passed=True,exact_UIDs_and_sources=True,matched_physical_states_and_seeds=True,cache_conflicts=0,raw_sha256=sha(OUT/'candidates.jsonl')))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def f(x):return '—'if x is None else f'{x:.3f}'
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |'for r in rows])
    text='''## 补充：Merry-Go-Round、CBF–CLF 与强化右手规则

应用户追加要求，另建 `baselines/` 缓存与记录，没有改变已冻结的 dictionary 执行源码。先审计 `diagnostics/baseline_comparison_v1/{HISTORY_CHECK,LITERATURE_CHECK,REPORT_BATCH1,STOP_REPORT}.md` 和纯函数代码。旧数据来自六个学习控制器、不同初态/随机流，不能作为本轮两个原生nominal的匹配成绩；仅复用纯计算内核。旧任务的已停止协调器及其初始化器均未导入或恢复。原 RHR-B、B3、安全过滤已保存的32条对应结果直接复用。

- **RHR A**：40步内目标距离进展<.05m时，对未到达机器人加.35m/s目标右侧偏置；RHR B使用场景原点的右切向，具有额外中心信息；均再次走原filter。
- **持续RHR**：每步对未到达机器人加.1m/s目标右偏置。
- **强化RHR（本任务的开发集调参版）**：预声明14个recipe，包含A/B、窗口20/40步、强度.15/.35/.5以及40步保持，另含持续.1/.35；仅用22个开发案例选一个全局固定recipe。不是逐测试状态挑最好，也不宣称某篇论文的标准“强化版”。
- **Merry heuristic**：同40步/.05m触发，未到达机器人.5m冲突连通组，组内向当前中心右切向偏置.35；中心退化时用目标右侧。这是现有启发式，不能称为论文原实现。
- **Merry paper SI（仅环岛部分）**：复用 [Lee等Merry-Go-Round论文](https://arxiv.org/html/2503.05848v1)机制的单积分器适配代码；预测冲突、持久环岛、合并/加入、逆时针切向和径向稳定、正交及外扇区退出。半径max(.3,2.2r)，局部预测1s/1m，几何中心搜索网格75/m、最大移动10半径。保留本系统原安全filter；**未完整移植作者障碍右手/临时目标模块，不是作者unicycle软件的原样复现，也不把论文保证转移到本系统**。所有常数与几何假设见 `baseline_roundabout.py`；无法放置环岛的次数单独记录。
- **CBF–CLF p=1/100**：在原局部CBF行和径向限速下，最小化 `||u-ref||²+p δ²`，约束 `2(p-g)ᵀu≤-||p-g||²+δ, δ≥0`。参考 [Ames等CLF/CBF框架](https://arxiv.org/abs/1609.06408)。λ=1，额外Clarabel解继续经过原filter，不修改原安全行；失败回退原安全投影并单独记录。它增加了软CLF优化，属于控制方法基线，不能与“只增Basis列”的贡献混称。

'''
    checks=load(OUT/'checks.json');text+=f"真实缓存局部问题{checks['local_problems']}个：关闭CLF后与原投影最大误差{checks['CLF_disabled_max_action_difference']:.3g}；开启CLF的最大原约束残差{checks['soft_CLF_max_constraint_violation']:.3g}。RHR B触发与偏置逐位对照通过。强化版冻结配置：`{summary['frozen_enhanced']['params']}`。\n\n"
    chosen=['B3_frozen','safety','RHR_A','RHR_B','RHR_persistent','RHR_enhanced','Merry_heuristic','Merry_paper_SI','CLF_CBF_p1','CLF_CBF_p100']
    for split in ('test','confirm'):
        text+=f"### {'第一次独立集，states18/19 seed3'if split=='test'else'新确认集，states20/21 seed4'}\n\n"
        text+=table(['固定方法','成功/16','timeout/碰撞','成功条件时间s','全队列截断s','ms/step','成功率95%区间'],[
           [n,m['success'],str(m['outcomes'].get('timeout',0))+'/'+str(m['outcomes'].get('collision',0)),f(m['conditional_time']),f(m['truncated_time']),f(m['ms_per_step']),str(m['success_ci'])]for n in chosen for m in [metrics[split][n]]])+'\n\n'
        text+=table(['相对固定B3','配对成功数','方法/B3时间s','救回/损失','成功率差95%区间'],[
           [n,m['paired_success'],f(m['paired_method_time'])+'/'+f(m['paired_B3_time']),str(m['rescued'])+'/'+str(m['broken']),str(m['paired_rate_ci'])]for n in chosen[2:]for m in [metrics[split][n]]])+'\n\n'
        text+=table(['场景（各4次）']+chosen[2:],[[sc]+[v[n]['success']for n in chosen[2:]]for sc,v in byscene[split].items()])+'\n\n'
    text+='### 相同停滞诊断的局部恢复时间\n\n'
    text+=table(['队列/方法','首次停滞事件/16','其中恢复','恢复条件时间s','事件队列截断时间s'],[[split+'/'+n,m['diagnostic_stall_episodes'],m['diagnostic_recovered'],f(m['diagnostic_recovery_conditional_time']),f(m['diagnostic_recovery_capped_time'])]for split in ('test','confirm')for n in chosen for m in [metrics[split][n]]])+'\n\n'
    text+='统一诊断：首次出现未到达机器人在40步内目标距离改善<.05m；冻结该时刻群体，直到其中每个机器人都比该时刻改善≥.05m或到达。失败赋剩余horizon，无事件样本不进入事件条件均值。正常排队也可能触发，故不是严格死锁分类。\n\n'
    text+=f"补充新仿真{len(rr)}条（含开发调参与smoke）；累计worker墙钟{summary['worker_seconds']:.1f}s；闭环numerical failure={summary['numerical_failures']}。实际CLF内部solver失败按每方法summary独立记录；区间同主实验按物理scene/state块bootstrap，样本很小。局部进展时间是总目标距离下降.1m的诊断代理，已保存在逐候选结果，不能当严格死锁解除时间。\n\n"
    text+='[完整补充统计](baselines/summary.json)、[冻结定义](baselines/DESIGN.json)、[开发排名](baselines/ENHANCED_FROZEN.json)、[真实局部问题回归](baselines/checks.json)。原始逐候选数据在 `baselines/candidates.jsonl`，轨迹与官方preflight/merger记录在同目录。\n'
    text+='\n论文身份审计见 [literature_baselines.md](literature_baselines.md)。用户所说的论文“强化右手规则”可能是Chen等的adaptive RHR/动态优先级，尚未确认；本轮调参版不代表该论文。MGR部分适配两组各4次静态墙碰撞，原CBF violation为0；其余失败为timeout。原论文未完整复现。\n'
    (BASE/'baseline_report.md').write_text(text)
    print('baseline rows',len(rr),'enhanced',summary['frozen_enhanced']['params'])
    for split,mm in metrics.items():print(split,{k:(v['success'],v['solver_failure_problems'])for k,v in mm.items()})
if __name__=='__main__':main()
