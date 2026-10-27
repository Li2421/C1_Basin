from experiment import *
import csv

def write_comparison(observed,targets,names):
    lookup={}
    for r in observed:
        method=r['method_code'];target_name=names.get(method)
        if method.endswith('_off'):target_name='模型预测控制与控制障碍函数'
        if method.endswith('_on'):target_name='论文活性方法'
        if target_name:lookup[(r['scene_code'],r['N'],target_name)]=r
    result=[]
    for t in targets:
        scene='lit_intersection_secondary' if t['scene_code']=='文献交叉口场景' else t['scene_code'];N=int(t['N']);r=lookup.get((scene,N,t['method']));metric=t['metric'];value=None;upper=None;status='未实现的论文基线';scope=''
        if r:
            assert not r['pending'];n=r['planned'];status='描述性比较';scope='已保存轨迹/已知到达试验'
            if metric=='success_percent':value=100*r['success_lower'];upper=100*r['success_upper'];status=r['comparison'];scope='严格50秒；unknown下界/上界'
            elif metric in ('infeasible_percent','infeasible_trials'):
                if r['method_code']=='impc_dr':status='完整每步QP状态未保存，不能重建';scope='保留unknown；不把缺少状态算成0无解'
                else:
                    scale=100/n if metric=='infeasible_percent' else 1;value=scale*r['optimization_infeasible_trials'];upper=scale*(r['optimization_infeasible_trials']+r['unknown']);scope='已观察优化无解；unknown可能无解的上界'
            elif metric=='gridlocks':value=r['stationary_unreached_5s'];upper=value+r['unknown'];status=r['comparison'];scope='最后5秒静止未到达代理；非无限时域死锁证明'
            elif metric=='deadlock_percent':value=100*r['stationary_unreached_5s']/n;upper=100*(r['stationary_unreached_5s']+r['unknown'])/n;status='各一个实例的定性比较，不能估计总体百分比';scope='30秒官方执行；静止未到达候选'
            else:
                field={'mean_completion_seconds':'mean_convergence_seconds','min_seconds':'min_convergence_seconds','max_seconds':'max_convergence_seconds','mean_seconds':'mean_convergence_seconds','h_min':'h_min'}.get(metric)
                value=r.get(field) if field else None;upper=None;scope='到达时间只统计已知收敛试验，含论文收敛但配对重叠者；h_min只覆盖保存的轨迹'
                if r['unknown']:status='存在unknown，以下仅为观察值'
        result.append(dict(scene=t['scene'],scene_code=scene,method=t['method'],N=N,metric=metric,paper_target=float(t['value']),observed_lower_or_point=value,observed_upper=upper,point_minus_target=None if value is None else value-float(t['value']),comparison=status,scope=scope))
    with (HERE/'PAPER_TARGET_COMPARISON.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=result[0]);w.writeheader();w.writerows(result)
    write(HERE/'PAPER_TARGET_COMPARISON.json',result)
