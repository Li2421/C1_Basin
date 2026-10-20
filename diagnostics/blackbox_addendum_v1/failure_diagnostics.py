from core import *
from progress import progress

def diagnose(raw):
    with np.load(raw['trace_path']) as z:
        positions=z['positions'].copy();velocities=z['velocities'].copy();goals=z['goals'].copy()
    d0=np.linalg.norm(goals-positions[0],axis=1);pT=progress(positions[-1],goals,d0);window=positions[-101:];delta=window[:,None]-window[None,:];diameter=float(np.linalg.norm(delta,axis=-1).max());mask=np.all(np.linalg.norm(velocities[1:],axis=-1)<.05,axis=1);best=count=0
    for low in mask:count=count+1 if low else 0;best=max(best,count)
    recent_speed=float(np.linalg.norm(velocities[-100:],axis=-1).max()) if len(velocities)>1 else 0.;unknown=bool(raw.get('numerical_failure'));collision=bool(raw.get('collision'));term='unknown' if unknown else 'collision' if collision else raw['outcome'];category='碰撞或unknown' if unknown or collision else '停滞候选' if term=='timeout' and len(positions)>=101 and diameter<=.05 and recent_speed<.05 and best>=100 else '仍在移动的超时' if term=='timeout' else '非超时的其他未成功终止'
    return dict(termination=term,final_mean_progress=pT,last100_max_displacement=diameter,last100_max_speed=recent_speed,longest_all_robots_slow_steps=best,physical_steps=len(positions)-1,category=category,not_certified_deadlock=True)
