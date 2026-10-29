"""Literal inherited trajectory classifier; extended only by goal blocking."""
from core import *

def longest(flags):
    best=run=0
    for x in flags:
        run=run+1 if x else 0;best=max(best,run)
    return best

def failure_type(result,scene):
    if not result.get('timeout'):return dict(category='数值异常或碰撞',outcome=result.get('outcome'),trace=result.get('trace'))
    with np.load(result['trace'])as trace:
        remaining=trace['route_remaining'];positions=trace['positions'];actions=trace['actions']
        history=json.loads(str(trace['admission_json']))if 'admission_json'in trace else []
        recorded_right_of_way=bool(history)
        dt=scene['config']['dt'];window=round(2/dt)
        moving=bool(len(remaining)>window and remaining[-window-1].sum()-remaining[-1].sum()>.05)
        gates=read(HERE/'PASSAGES.json')['by_scene'][scene['name']]['passages'];breaches=[];blocks=[]
        if not history and gates:
            # The old baseline has no admission state. Geometric two-way
            # occupancy is still measurable; exit-block attribution is not.
            history=[]
            for t,pos in enumerate(positions[:-1]):
                step=[]
                for gate in gates:
                    u=np.asarray(gate['axis']);n=np.array([-u[1],u[0]]);local=pos-gate['centre']
                    inside=(np.abs(local@u)<=gate['half_length']+scene['config']['agent_radius'])&(np.abs(local@n)<=gate['width']/2)&(remaining[t]>scene['config']['goal_tolerance'])
                    dirs=np.sign((trace['route_guides'][t]-pos)@u).astype(int)
                    step.append(dict(inside=inside.tolist(),robot_direction=dirs.tolist(),direction=0))
                history.append(step)
        for j,gate in enumerate(gates):
            mixed=[];blocked=[[]for _ in positions[0]]
            for t,step in enumerate(history):
                if j>=len(step):continue
                m=step[j];inside=np.asarray(m['inside']);dirs=np.asarray(m['robot_direction']);s=m['direction']
                mixed.append(bool(np.any(inside&(dirs==1))and np.any(inside&(dirs==-1))))
                slow=inside&(dirs==s)&(np.linalg.norm(actions[t],axis=1)<.02)if s else np.zeros(len(inside),bool)
                ex=np.asarray(gate['centre'])+s*gate['half_length']*np.asarray(gate['axis'])
                other=(dirs!=s)&(np.linalg.norm(positions[t]-ex,axis=1)<=2*scene['config']['agent_radius']+.05)
                for i in range(len(slow)):blocked[i].append(bool(slow[i]and np.any(other)))
            breaches.append(longest(mixed)*dt);blocks.append(max((longest(flags)*dt for flags in blocked),default=0))
        a=any(x>=2 for x in breaches);b=any(x>=2 for x in blocks)
        category='通行权失守'if a else'出口被堵'if b else'仍在推进的超时'if moving else'通道外拥堵或其他'
        return dict(category=category,flags=dict(right_of_way_breach=a,exit_blocked=b if recorded_right_of_way else None,progressing_timeout=moving),recorded_right_of_way=recorded_right_of_way,
                    max_bidirectional_seconds=max(breaches,default=0),max_exit_blocked_seconds=max(blocks,default=0),trace=result['trace'])
