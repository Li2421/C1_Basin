"""Recompute all dictionary directions from old real states, no nominal inference."""
import json,itertools,time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from runner import OUT,RN,inputs,dump,init_worker,AVAILABLE_CPUS
from discovery_design import freeze
from shared_control.basis_dictionary import evaluate,add_target,RoleState,NEW,ALL
from shared_control.hard_projection import HardProjectionConfig


def analyze(c):
    tr=np.load(c['diagnostic_trace']);geo=json.loads(str(tr['first_geometry_json']));scene=inputs()[0][c['scene']];cfg=RN.LiteratureConfig(**scene['config'])
    speed=np.linalg.norm(tr['actions'],axis=-1).max(axis=-1);tail=np.arange(len(speed)//3,len(speed));t=int(tail[np.argmin(speed[tail])]);p=tr['positions'][t];g=tr['goals'];safe=tr['safe'][t];raw=tr['nominal'][t]
    fs=[]
    for x in geo[t]:
        f=RN.SwapSafety.__new__(RN.SwapSafety);f.A=np.asarray(x['A']);f.lower=np.asarray(x['lower']);f.kinds=x['labels'];f.max_speed=cfg.max_speed;f.config=HardProjectionConfig();fs.append(f)
    fields=np.stack([RN.distributed_fields(p,g,safe,i,cfg.max_speed,cfg.goal_tolerance,RN.contact_radius(cfg))for i in range(len(p))])
    cols,meta=evaluate(p,g,safe,raw,fields,fs,scene['walls'],cfg,RoleState(),t)
    B=np.column_stack([cols[k].ravel()for k in ALL]);norms=np.linalg.norm(B,axis=0);cos=(B.T@B)/np.maximum(norms[:,None]*norms[None,:],1e-30)
    angles=np.degrees(np.arccos(np.clip(cos,-1,1)));angles[(norms[:,None]*norms[None,:])<1e-18]=np.nan
    b3=B[:,:3];u,s,_=np.linalg.svd(b3,full_matrices=False);rank=int(np.sum(s>1e-8));q=u[:,:rank]
    def project(v):return np.concatenate([f.project(v[i:i+1])[0]for i,f in enumerate(fs)])
    baseline=project(add_target(safe,fields,cols,c['baseline_eta'],{}));metrics={}
    for name in NEW:
        d=cols[name].ravel();newpart=d-q@(q.T@d);records=[]
        for gain in(-1.,1.):
            try:
                output=project(add_target(safe,fields,cols,c['baseline_eta'],{name:gain}))
                records.append(dict(gain=gain,action_change=float(np.linalg.norm(output-baseline)),action_norm=float(np.linalg.norm(output))))
            except Exception as e:records.append(dict(gain=gain,numerical_unknown=str(e)))
        metrics[name]=dict(norm=float(np.linalg.norm(d)),relative_outside_B3=float(np.linalg.norm(newpart)/np.linalg.norm(d))if np.linalg.norm(d)>1e-8 else None,
            rank_gain=int(np.linalg.matrix_rank(np.c_[b3,d],tol=1e-8))-rank,projected=records)
    return dict(case=c,step=t,source_trace=c['diagnostic_trace'],selected_speed=float(speed[t]),rank_b3=rank,
        rank_dictionary=int(np.linalg.matrix_rank(B,tol=1e-8)),column_names=ALL,norms=norms,angles_degrees=angles,columns=metrics,
        meta=meta,positions=p,goals=g,safe=safe,nominal=raw,
        interpretation='New role activation at diagnostic state; numerical rank/projection response, not success or impossibility certificate.')


def main():
    start=time.monotonic();p=freeze()
    with ProcessPoolExecutor(max_workers=min(len(AVAILABLE_CPUS),len(p['cases'])),initializer=init_worker)as pool:results=list(pool.map(analyze,p['cases']))
    dump(OUT/'static_states.json',results)
    summary={}
    for name in NEW:
        rr=[x['columns'][name]for x in results]
        summary[name]=dict(states=len(rr),nonzero=sum(r['norm']>1e-8 for r in rr),rank_gain_states=sum(r['rank_gain']>0 for r in rr),
            projection_changes=sum(any(a.get('action_change',0)>1e-5 for a in r['projected'])for r in rr),
            mean_outside_B3=float(np.mean([r['relative_outside_B3']for r in rr if r['relative_outside_B3']is not None]))if any(r['relative_outside_B3']is not None for r in rr)else None,mean_norm=float(np.mean([r['norm']for r in rr])))
    dump(OUT/'static_summary.json',dict(columns=summary,elapsed=time.monotonic()-start));print(summary)

if __name__=='__main__':main()
