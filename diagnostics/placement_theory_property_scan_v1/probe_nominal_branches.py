"""Nominal Flow readout clipping and nearest-wall diagnostics at paired states."""
import copy
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'../placement_boundary_mechanism_audit_v1'))
from closed_loop_probe import key_for
from validation_stage2.run_rollouts import make_runtime

def main():
    p=json.loads((HERE/'protocol.json').read_text());rows=[];runtime={}
    for st in p['states']:
        if st['scenario'] not in runtime:runtime[st['scenario']]=make_runtime(st['scenario'])
        rt=runtime[st['scenario']]
        variants=[(0.,0)]
        if st['uid'] in p['state_probe_uids']:
            variants += [(h,s) for h in p['state_radii'] for s in (-1,1)]
        for h,sign in variants:
            state=copy.deepcopy(st);pos=np.asarray(state['physical']['positions'],float)
            if h:
                direction=np.asarray(p['state_position_directions'][st['uid']],float)
                state['physical']['positions']=(pos+h*sign*direction).tolist()
            env=rt.make_env(state);key=key_for(st['uid'],p['seed'],0,p['future_root'])
            obs,flow,_=rt.nominal(env,key)
            z=np.asarray(rt.trace_fn(obs,key))[-1].reshape(-1,2)
            raw=rt.mean+rt.scale*z
            clipped=np.clip(raw,-1,1)
            c=np.abs(raw)>1.
            radial=np.linalg.norm(clipped,axis=1)>rt.config.max_speed
            geometry=rt.safety(env).geometry
            walls=np.asarray(geometry['wall_h'])
            rank=np.argsort(walls,axis=1)
            gap=np.take_along_axis(walls,rank[:,1:2],axis=1)-np.take_along_axis(walls,rank[:,0:1],axis=1)
            rows.append(dict(uid=st['uid'],scene=st['scenario'],radius=h,sign=sign,
                component_clip_mask=c.astype(int).tolist(),radial_bound_mask=radial.astype(int).tolist(),
                nearest_wall_index=rank[:,0].tolist(),nearest_wall_gap=gap.ravel().tolist(),
                nominal_action_norm=float(np.linalg.norm(flow)),
                nominal_speed_max=float(np.linalg.norm(flow,axis=1).max())))
    byuid={}
    for r in rows:
        b=next(x for x in rows if x['uid']==r['uid'] and x['radius']==0)
        r['component_clip_switch']=r['component_clip_mask']!=b['component_clip_mask']
        r['radial_bound_switch']=r['radial_bound_mask']!=b['radial_bound_mask']
        r['nearest_wall_switch']=r['nearest_wall_index']!=b['nearest_wall_index']
    out=dict(total=len(rows),perturbed=sum(r['radius']>0 for r in rows),
        clip_switches=sum(r['component_clip_switch'] for r in rows if r['radius']>0),
        radial_switches=sum(r['radial_bound_switch'] for r in rows if r['radius']>0),
        nearest_wall_switches=sum(r['nearest_wall_switch'] for r in rows if r['radius']>0),
        baseline_clipped_states=sum(any(v for q in r['component_clip_mask'] for v in q) for r in rows if r['radius']==0),
        baseline_radial_states=sum(any(r['radial_bound_mask']) for r in rows if r['radius']==0),
        caveat='Nearest-wall argmin is diagnostic; each wall constraint and segment endpoint switch needs separate analysis.')
    (HERE/'nominal_branch_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (HERE/'nominal_branch_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
