#!/usr/bin/env python3
"""Freeze the fixed-eight multiball geometry pilot before any new rollout."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=ROOT/'diagnostics/orthoflow3_t0_multiball_basin_learning_v1'
PRIOR=ROOT/'diagnostics/orthoflow3_t0_basin_completion_v1'
T0=ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1'
GAP=ROOT/'diagnostics/orthoflow3_zero_active_gap_connectivity_v1'
BASIS=ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
AFF=np.array([.875,0,.375]); SCALE=np.array([.75,1,.75])
BASIS_SHA='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(p,rr,fields=None):
    fields=fields or list(rr[0])
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
def key(e): return np.asarray(e,np.float64).tobytes().hex()
def inside(x,eq): return bool(np.all(eq[:,:3]@x+eq[:,3] <= 2e-10))

def main():
    HERE.mkdir(parents=True,exist_ok=True);(HERE/'stage_a').mkdir(exist_ok=True);(HERE/'logs').mkdir(exist_ok=True)
    if sha(BASIS)!=BASIS_SHA: raise RuntimeError('OrthoFlow3 hash mismatch')
    states=json.load(open(PRIOR/'frozen_8state_manifest.json'))['attempted_states']
    if len(states)!=8 or not all(x['true_t0'] for x in states): raise RuntimeError('fixed t0 cohort mismatch')
    dump(HERE/'fixed8_manifest.json',{'states':states,'source':str(PRIOR/'frozen_8state_manifest.json'),
         'source_sha256':sha(PRIOR/'frozen_8state_manifest.json'),'state_count':8,'source_groups':len({x['source_group'] for x in states})})
    # Exact normalized hull.
    lo=np.array([.5,-.5,0.]);hi=np.array([1.25,.5,.75]);verts=[np.zeros(3)]
    for a in (lo[0],hi[0]):
      for b in (lo[1],hi[1]):
       for c in (lo[2],hi[2]): verts.append(np.array([a,b,c]))
    tv=(np.asarray(verts)-AFF)/SCALE; hull=ConvexHull(tv);eq=np.asarray(hull.equations)
    # Exclude every exact eta queried in the prior fixed-eight scientific cache globally,
    # so the common construction/coverage coordinates are identical for all states.
    prior_keys=set(); prior_rows=0
    for st in states:
        sid=st['state_id']
        for p in [T0/'anchor_runs'/sid/'raw/pilot_rollouts.jsonl',PRIOR/'raw'/sid/'q64_rollouts.jsonl']:
            if not p.exists(): continue
            for line in p.read_text().splitlines():
                if line.strip():
                    r=json.loads(line);prior_keys.add(key(r['eta']));prior_rows+=1
    eng=qmc.Sobol(3,scramble=False); accepted=[];source_index=0
    low=np.array([0.,-.5,0.]);high=np.array([1.25,.5,.75])
    while len(accepted)<96:
        for u in eng.random(256):
            eta=low+(high-low)*u;te=(eta-AFF)/SCALE;k=key(eta)
            if inside(te,eq) and k not in prior_keys:
                accepted.append({'global_index':len(accepted),'purpose':'construction' if len(accepted)<64 else 'independent_coverage',
                    'sobol_source_index':source_index,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],
                    't1':te[0],'t2':te[1],'t3':te[2]})
                prior_keys.add(k)
                if len(accepted)==96: break
            source_index+=1
    write(HERE/'outward_sobol_cloud.csv',accepted)
    # Copy exact frozen directions and domain records.
    directions=list(csv.DictReader(open(T0/'frozen_ray_directions.csv')))
    if len(directions)!=18: raise RuntimeError('direction count')
    write(HERE/'frozen_ray_directions.csv',directions)
    dump(HERE/'geometry_constants.json',{'normalization':{'affine':AFF.tolist(),'scale':SCALE.tolist()},
         'E_bridge':'conv({0} union ACTIVE_BOX)','normalized_volume':float(hull.volume),'halfspaces':eq.tolist(),
         'kappa':.85,'ray_radii':[.025,.05,.10,.20,.35,.50],'K_max':4,'new_component_max':3,
         'redundancy_min_outside_distance':.10,'retained_gamma':.80,'future_root_seed':2026092811})
    protocol='''# OrthoFlow3 true-t0 multiball basin-to-learning v1\n\nStage A is frozen before outcomes. It uses the exact eight true-t0 states and their accepted original balls. One common unscrambled Sobol rejection sequence in E_bridge supplies 64 construction points followed by 32 disjoint coverage points; coordinates queried anywhere in the prior fixed-eight cache are excluded globally. Outer candidates are ranked by descending normalized distance outside the current union, with Sobol index tie-break. Only B63 centers at least 0.10 outside the union are attempted. At most three new components are built with the accepted 18-ray, fixed-grid, three-bisection, eight-limiting-direction, kappa=0.85 sphere protocol and four independent mandatory B63 interior checks (.25/.60/.90/.97). Failed components are recorded, never repaired.\n\nThe frozen union receives 16 independent interior screening points and two near-boundary 64-seed checks per component. A disjoint 32-point common cloud is screened; every 8/8 point is promoted to 64 for the empirical robust-coverage statistic. Union volume uses one common deterministic 2^18-point Sobol set. The six preregistered geometry gates and >75% retained-set degeneracy rule are applied exactly. No network may be trained unless all gates pass.\n'''
    (HERE/'protocol.md').write_text(protocol)
    # Expected cost: three attempted components/state. Range reflects early ray failures and candidate rejection.
    expected={'outward_screen':8*64*8,'center_promotions':8*3*56,
      'component_ray_and_validation_low':8*3*1500,'component_ray_and_validation_high':8*3*2100,
      'union_validation':8*(16*8+8*56),'coverage_if_75pct_screen_success':8*(32*8+24*56)}
    lo_cont=expected['outward_screen']+expected['center_promotions']+expected['component_ray_and_validation_low']+expected['union_validation']+expected['coverage_if_75pct_screen_success']
    hi_cont=expected['outward_screen']+expected['center_promotions']+expected['component_ray_and_validation_high']+expected['union_validation']+8*(32*64)
    dump(HERE/'stage_a_cost_preflight.json',{'expected_components':expected,'projected_new_continuations':[lo_cont,hi_cont],
         'empirical_steps_per_continuation':473.3,'projected_physical_steps':[int(lo_cont*473.3),int(hi_cont*473.3)],
         'projected_critical_path_hours_6_shards':[2.0,4.0],'server_checked_idle_at':'2026-09-28T01:37:20+08:00',
         'authorized_shards':6,'decision':'PROCEED_FIXED8_GEOMETRY','training_authorized':False})
    dump(HERE/'cache_reuse_audit.json',{'prior_exact_rows_scanned':prior_rows,'prior_unique_eta_coordinates_excluded_globally':len(prior_keys)-96,
         'new_rollouts_during_preflight':0,'exact_match_key':'state_id,float64 eta bytes,future_index,future_root=2026092811'})
    print(json.dumps({'states':[x['state_id'] for x in states],'prior_rows':prior_rows,'cloud_sha256':sha(HERE/'outward_sobol_cloud.csv'),
      'projected':[lo_cont,hi_cont]},indent=2))
if __name__=='__main__': main()
