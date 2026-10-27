"""Replay saved failure states; distinguish added speed bound from original DI rows."""
import json
from pathlib import Path
import numpy as np
from formula_baselines import Wang,solve_qp,closest_segment
HERE=Path(__file__).resolve().parent
def main():
    cases=json.load(open(HERE/'paper/pilot_formula_planned.json'))['exact_configs'];byid={c['rollout_uid']:c for c in cases};results=[]
    for r in map(json.loads,open(HERE/'paper/candidates.jsonl')):
        if r['rollout_uid']not in byid or r['controller']!='wang2017_full_di_tracking' or not r['numerical_failure']:continue
        c=byid[r['rollout_uid']];cfg=c['scenario_payload']['config'];trace=np.load(r['trace']);pos=trace['positions'][-1];vel=trace['velocities'][-1];i=r['controller_stats']['solver']['failures'][-1]['agent'];p=pos[i];v=vel[i];w=Wang(None,0)
        A=[];b=[]
        for j,pj in enumerate(pos):
            if i==j:continue
            row,bi,_=w.barrier(p-pj,v-vel[j],2*cfg['agent_radius']+cfg['agent_collision_margin'],2*w.alpha,.5);A.append(row);b.append(bi)
        for wall in c['scenario_payload']['walls']:
            q,_=closest_segment(p,wall);row,bi,_=w.barrier(p-q,v,cfg['agent_radius']+cfg['wall_radius']+cfg['wall_collision_margin'],w.alpha,1.);A.append(row);b.append(bi)
        mat=np.r_[np.asarray(A),np.eye(2),-np.eye(2)];rhs=np.r_[b,[w.alpha]*4];tests={}
        for name,speed in [('original_acceleration_rows',None),('plus_common_speed',cfg['max_speed'])]:
            try:
                u,status=solve_qp(np.eye(2),np.zeros(2),mat,rhs,speed,v,cfg['dt']);tests[name]=dict(status=status,max_residual=float(np.max(mat@u-rhs)),next_speed=float(np.linalg.norm(v+cfg['dt']*u)))
            except Exception as exc:tests[name]=dict(status=str(exc))
        results.append(dict(scene=r['scene'],step=r['episode_length'],agent=i,tests=tests))
    (HERE/'paper/failure_diagnosis.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
