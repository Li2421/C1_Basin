"""Unmodified v2 analysis kernels selected by AST; no historical imports."""
import numpy as np
import collections
LABELS=('安全层导致的全局失败','墙前陷阱实例','controller级失败')

def classify(eligible,upper,counts):
    if not eligible:return '未通过能力门'
    if upper>.05:return '非全局失败实例'
    total=sum(counts.values())
    safety=counts.get('robot_filter_equilibrium_deadlock',0)+counts.get('blocked_by_arrived',0)
    if total and safety>total/2:return LABELS[0]
    if total and counts.get('wall_filter_equilibrium',0)>total/2:return LABELS[1]
    return LABELS[2]


def weighted_rates(rows,extra_metrics=()):
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['layer'],r['method'])].append(r)
    output={}
    metrics=('success','upper','collision','timeout','unknown','deadlock_episode','wall_episode','canonical_joint_episode','actual_joint_episode',*extra_metrics)
    for (layer,method),records in groups.items():
        clusters=collections.defaultdict(list)
        for r in records:clusters[(r['type'],r['controller'],r['scene'],r['N'],r['state'])].append(r)
        ctls=collections.defaultdict(list)
        for (type_code,ctl,scene,n,index),rr in clusters.items():
            if {r['seed'] for r in rr}!={0,1}:continue
            vector=np.asarray([[*[r[k] for k in metrics],r['deadlock_agents']/r['N'],r['wall_agents']/r['N']] for r in rr],float).mean(0)
            ctls[(type_code,ctl)].append(vector)
        types=collections.defaultdict(list)
        for (type_code,ctl),values in ctls.items():types[type_code].append(np.mean(values,axis=0))
        if not types:continue
        point=np.mean([np.mean(values,axis=0) for values in types.values()],axis=0)
        output[layer+'|'+method]=dict(rates=dict(zip((*metrics,'deadlock_agent_fraction','wall_agent_fraction'),point.tolist())),controllers=len(ctls),types=sorted(types),scenes=sorted({r['scene'] for r in records}),state_clusters=len(clusters),weighting='Equal type -> controller -> available (scene,N,index) -> paired seeds')
    return output


def controller_drops(rows):
    groups=collections.defaultdict(dict)
    for r in rows:groups[(r['layer'],r['scene'],r['controller'])].setdefault(r['method'],[]).append(r)
    out=[]
    for (layer,scene,ctl),methods in sorted(groups.items()):
        for method,rr in methods.items():
            for reference,label in [('fixed_eta10','catastrophe_vs_F10'),('safety','harm_vs_safety')]:
                if reference not in methods:continue
                keys={(r['N'],r['state'],r['seed']) for r in rr}
                base=[r for r in methods[reference] if (r['N'],r['state'],r['seed']) in keys]
                common={(r['N'],r['state'],r['seed']) for r in base};target=[r for r in rr if (r['N'],r['state'],r['seed']) in common]
                if not common:continue
                lo=np.mean([r['success'] for r in target]);hi=np.mean([r['upper'] for r in target]);blo=np.mean([r['success'] for r in base]);bhi=np.mean([r['upper'] for r in base])
                out.append(dict(layer=layer,scene=scene,controller=ctl,method=method,metric=label,paired_episodes=len(common),paired_scenes=sorted({r.get('original_scene',r['scene']) for r in target}),definite_drop_10pp=bool(hi-blo<=-.1+1e-12),possible_drop_10pp=bool(lo-bhi<=-.1+1e-12),delta_lower=float(lo-bhi),delta_upper=float(hi-blo)))
    return out

