"""Recompute paper comparisons from immutable saved predictions only."""
from experiment import *
from phase_reports import records
from scipy.stats import beta
import numpy as np
import csv
import collections

METHODS={
 'impc_dr':'无限时域模型预测控制与死锁解除（IMPC-DR）',
 'buffered_voronoi_right':'缓冲Voronoi单元与右手绕行点规则·误加末速度约束变体',
 'buffered_voronoi_right_corrected':'缓冲Voronoi单元与右手绕行点规则·修正实现',
 'circle_centralized':'集中式控制障碍函数控制器',
 'circle_df':'去中心化全责控制障碍函数控制器（DF）',
 'circle_dr':'去中心化互惠控制障碍函数控制器（DR）',
 'circle_ccs':'协作避碰调度策略（CCS）',
 'circle_pcca':'伪协作避碰策略与单步延迟（PCCA）',
 'circle_pcca_filtered':'伪协作避碰策略与0.2秒低通（PCCA）',
 'doorway_off':'模型预测控制与控制障碍函数·活性关闭',
 'doorway_on':'模型预测控制与控制障碍函数·活性开启',
 'intersection_off':'模型预测控制与控制障碍函数·活性关闭',
 'intersection_on':'模型预测控制与控制障碍函数·活性开启'}
TARGET_NAMES={'impc_dr':'无限时域模型预测控制与死锁解除','buffered_voronoi_right_corrected':'缓冲Voronoi单元与右手规则','circle_centralized':'集中式','circle_df':'去中心化全责','circle_dr':'去中心化互惠','circle_ccs':'协作避碰调度','circle_pcca':'伪协作避碰与单步延迟','circle_pcca_filtered':'伪协作避碰与0.2秒低通'}
FULL_NAMES=dict(NAMES,lit_intersection_secondary='文献交叉口场景')

def interval(x,n):return [0. if x==0 else float(beta.ppf(.025,x,n-x+1)),1. if x==n else float(beta.ppf(.975,x+1,n-x))]
def csvfile(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)

def main(final=False):
    groups=collections.defaultdict(list);planned=collections.Counter();sources=[]
    for stage in ('paper_primary','paper_bvc_v2','paper_secondary'):
        p=HERE/f'{stage}_CASES.json'
        if not p.exists():continue
        cases=read(p);rr=records(stage)
        if final:assert len(rr)==len(cases),(stage,len(rr),len(cases))
        for c in cases:planned[(c['state']['scene'],c['state']['N'],c['method'])]+=1
        for r in rr:
            c=r['case'];groups[(c['state']['scene'],c['state']['N'],c['method'])].append(r)
            sources.append(dict(rollout_uid=r['rollout_uid'],trace_sha256=r['trace_sha256']))
    targets=list(csv.DictReader((HERE/'PAPER_TARGETS.csv').open()));rows=[]
    for key,n in sorted(planned.items()):
        scene,N,method=key;rr=groups[key];dt=.05 if scene=='lit_circle_five' else .2 if scene=='lit_narrow_passage' else .15 if scene=='lit_crowded_random' else .6
        strict=int(50/dt) if scene=='lit_crowded_random' else None
        success=sum(r['success'] and (strict is None or r['episode_length']<=strict) for r in rr)
        unknown=sum(r['numerical_failure'] for r in rr);quiet=0;physical=0;outer_overlap=[];times=[];hmins=[];minimum=1e99;detours=0;infeasible=sum(bool(r.get('optimization_infeasible')) or r.get('infeasible_steps',0)>0 for r in rr)
        for r in rr:
            assert sha(r['trace_path'])==r['trace_sha256']
            with np.load(r['trace_path']) as z:
                p=z['positions'];p=p[:strict+1] if strict is not None else p;g=np.asarray(r['case']['state']['goals']);distance=np.linalg.norm(p-g,axis=-1)
                if scene=='lit_circle_five':outer_overlap.append(float(max(0,np.linalg.norm(p,axis=-1).max()+2-11)))
                v=z['velocities'][:len(p)] if 'velocities' in z else np.diff(p,axis=0)/dt
                physical+=bool(not r['numerical_failure'] and np.all(distance[-1]<.1))
                w=int(np.ceil(5/dt))
                if r['timeout'] and len(p)>w and np.max(np.linalg.norm(p[-w-1:]-p[-w-1],axis=-1))<.1 and np.max(np.linalg.norm(v[-w:],axis=-1))<.1 and not np.all(distance[-1]<.1):quiet+=1
                if r.get('paper_converged') and not r['numerical_failure'] and (strict is None or r['episode_length']<=strict):times.append((len(p)-1)*dt if scene in ('lit_doorway','lit_intersection_secondary') else r['episode_length']*dt)
            if r.get('h_min') is not None:hmins.append(r['h_min'])
            if r.get('minimum_agent_clearance') is not None:minimum=min(minimum,r['minimum_agent_clearance'])
            detours+=len(r.get('detour_events',[]))
        target=next((float(x['value']) for x in targets if x['scene_code']==scene and int(x['N'])==N and x['method']==TARGET_NAMES.get(method) and x['metric']=='success_percent'),None)
        ci=interval(success,n);tolerance=max(.05,(ci[1]-ci[0])/2);comparison='无直接数值目标'
        if target is not None:
            accepted=[]
            for possible in range(success,success+unknown+1):
                possible_ci=interval(possible,n);limit=max(.05,(possible_ci[1]-possible_ci[0])/2)
                accepted.append(abs(possible/n-target/100)<=limit+1e-12)
            comparison='未完成' if len(rr)<n else '一致（含unknown上下界）' if unknown and all(accepted) else '不一致（含unknown上下界）' if unknown and not any(accepted) else 'unknown导致不能判定' if unknown else '一致' if accepted[0] else '不一致'
        stucktarget=next((int(float(x['value'])) for x in targets if x['scene_code']==scene and x['method']==TARGET_NAMES.get(method) and x['metric']=='gridlocks'),None)
        if stucktarget is not None:
            outer=[interval(quiet,n)[0],interval(min(n,quiet+unknown+n-len(rr)),n)[1]]
            if not outer[0]<=stucktarget/n<=outer[1]:comparison='静止代理计数不一致；未知上下界亦未包含论文值'
            elif unknown or len(rr)<n:comparison='unknown导致不能确认一致；区间未排除论文值'
            else:comparison='静止代理计数区间包含论文值；非死锁证明'
        rows.append(dict(scene=FULL_NAMES[scene],scene_code=scene,N=N,controller=METHODS[method],method_code=method,planned=n,completed=len(rr),pending=n-len(rr),strict_horizon_success=success,official_or_implemented_arrival=sum(bool(r.get('paper_converged')) for r in rr),physical_final_distance_lt_point1=physical,collision=sum(r['collision'] for r in rr),timeout=sum(r['timeout'] for r in rr),optimization_infeasible_trials=infeasible,unknown=unknown,stationary_unreached_5s=quiet,paper_gridlock_target=stucktarget,success_lower=success/n,success_upper=(success+unknown+n-len(rr))/n,ci95_lower=ci[0],ci95_upper=ci[1],paper_success_target_percent=target,comparison_tolerance_pp=100*tolerance,comparison=comparison,min_convergence_seconds=min(times) if times else None,max_convergence_seconds=max(times) if times else None,mean_convergence_seconds=float(np.mean(times)) if times else None,h_min=min(hmins) if hmins else None,minimum_agent_clearance=minimum if minimum<1e99 else None,detour_events=detours,outer_boundary_violating_trials=sum(x>0 for x in outer_overlap),maximum_disk_outer_overlap_m=max(outer_overlap) if outer_overlap else None,evidence='DEVELOPMENT'))
    write(HERE/'PAPER_REPRODUCTION.json',rows);csvfile(HERE/'PAPER_REPRODUCTION.csv',rows)
    text=['# 论文版复现','','本表仅为论文版。数值unknown保留且不重试；timeout与有限窗口静止均不等于无限时域死锁。主要与修正分支分别保留，未据论文目标调参。','','|场景|controller|N|完成/计划|严格时限成功（按保存样本检查）|位置参照到达|碰撞|timeout|优化无解|unknown|最后5秒静止未到达|数值比较|','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:text.append('|'+ '|'.join(map(str,[r['scene'],r['controller'],r['N'],f"{r['completed']}/{r['planned']}",r['strict_horizon_success'],r['physical_final_distance_lt_point1'],r['collision'],r['timeout'],r['optimization_infeasible_trials'],r['unknown'],r['stationary_unreached_5s'],r['comparison']]))+'|')
    text+=['','窄通道场景：400dpi图8第一帧实际是1.8秒，真实t0未公开；本轮声明reset来自该帧。即使无限时域模型预测控制与死锁解除能通过，右手规则未出现要求的反复离开/返回活锁，仍须标注“论文证据未复现”。缓冲Voronoi单元的首版末速度0约束无文献依据，独立保留为变体；修正分支只删除此约束，右手启停/代价等不完整披露部分均沿用预注册。','','二维拥挤随机场景：原始700组样本不加流程版的硬墙排除。官方代码采用Clarabel，保留硬编码r_min=.335、ρ=.2、η自适应、覆盖A后再计算B的原离散实现等与论文数值/方程的差异；初始化矩阵验证表明这不是正确Tustin，见OFFICIAL_DYNAMICS_AUDIT.json。官方终止用预测轨迹cost_index<=1（预测位置.01阈值），并非流程版.08到达事件；表中另报最终位置<.1的参照数。50秒只计333步内已保存的终止，原334步轨迹不重复生成。增量序列凸规划与分布式模型预测控制未重新实现。原IMPC成功轨迹未保存每步QP状态，因此表II无解率只能给已观测失败的部分信息，不能把未观测无解等同0。','','圆形围场五机随机场景：按Clopper–Pearson二项95%区间比较静止代理与论文卡死计数，unknown存在时不宣称一致。此代理是预注册有限窗口证据，不是论文原始长期死锁证明。论文表I有负h_min；论文收敛与机器人间无重叠的任务成功分列。外圆是软约束，另外从既有轨迹报告越出外圆的次数和最大圆盘越界米数，不将软边界冒充硬墙。外圆软约束权重、最小最大违反恢复、初速度与采样密度均含声明假设。正式期刊版未取得。第二优先级协作避碰调度与伪协作避碰见补充三，主机外圆约束与低通离散是实现假设。','','门口场景与文献交叉口场景：只运行题目要求的关闭/开启各一个官方实例，不能据此估计总体100%/0%概率。保留原两次各3步方法调用与50轮顺序更新，同一模拟器在两次调用间不重置，因此每轮.6秒、共30秒；早期raw的15秒字段与150步计数有误，见TIME_METADATA_CORRECTION.json，不据其重跑或改写原记录。流程版40.4秒时限不能混入本表。活性分支实际主要调整速度，论文中的活性约束在公开代码中被注释；不改代码补成另一算法。静态他人障碍半径.02、重复障碍及交叉口坐标笔误保留。Autonomous Robots 49:12是不错的期刊；按任务写作约定，其层级低于IJRR、IEEE T-RO。','','碰撞检查覆盖：圆形围场五机随机场景对保持加速度段求解析最小配对间隙；其余论文版基于保存的位置样本，门口两次调用的MPC观测存在重置，不能把这些样本说成完整连续时间安全证书。流程版使用原native swept事件，两个版本不混用。原始论文目标完整保存于PAPER_TARGETS.csv；本表的最小/最大/平均收敛时间、h_min与无解试验数保存在CSV/JSON，未知与未实现项见GAPS.md。']
    (HERE/'PAPER_REPRODUCTION.md').write_text('\n'.join(text)+'\n')
    if final:
        from paper_target_comparison import write_comparison
        write_comparison(rows,targets,TARGET_NAMES)
        seal('PAPER_REPRODUCTION.receipt.json',[HERE/'PAPER_REPRODUCTION.json',HERE/'PAPER_REPRODUCTION.csv',HERE/'PAPER_REPRODUCTION.md',HERE/'PAPER_TARGET_COMPARISON.csv',HERE/'PAPER_TARGET_COMPARISON.json',HERE/'paper_results.py',HERE/'paper_target_comparison.py'],all_paper_branches_complete=True)

if __name__=='__main__':main('--final' in sys.argv)
