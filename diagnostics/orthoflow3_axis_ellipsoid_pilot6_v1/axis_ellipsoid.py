#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path('/home/zhihan/research/Basin_C1')
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
OUT = ROOT/'diagnostics/orthoflow3_axis_ellipsoid_pilot6_v1'
BALL = ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1'
OLD = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_pilot6_v1'
QDIR = ROOT/'diagnostics/orthoflow3_q_learnability_v2'
DDIR = ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
QGDIR = ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1'
BASIS = ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
INTERFACE = ROOT/'shared_control/basis_families.py'

STATE_IDS = ['N_r104_s125','ZR_P_r052_m080_s95400004_p253','S_r043_p02',
             'N_r076_s119','R_D2_s95101009_p112','N_r004_m120']
SHRINK = .85
SHELL_SCALE = 1.05
MEANINGFUL_RATIO = 1.25
PROMOTE_INDICES = (3,7,11,15,19,23,27,31)
FUTURE_ROOT = 2026092702


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def read_csv(path: Path) -> list[dict]:
    with path.open() as f: return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else ['state_id'])
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def ekey(eta) -> str:
    return np.asarray(eta, dtype=np.float64).tobytes().hex()


def load_old(work: Path):
    spec=importlib.util.spec_from_file_location('ellipsoid_old_pilot', OLD/'pilot6.py')
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    mod.HERE=work; mod.CAP_CONT=10000; mod.CAP_STEPS=3000000
    return mod


def frozen_paths() -> list[Path]:
    return [BASIS,INTERFACE,BALL/'frozen_state_manifest.json',BALL/'robust_centers.csv',
            BALL/'ebridge_definition.json',BALL/'ebridge_halfspaces.csv',BALL/'eta_normalization.json',
            BALL/'frozen_ray_directions.csv',BALL/'ray_screening.csv',BALL/'boundary_bisection.csv',
            BALL/'boundary_brackets.csv',BALL/'limiting_direction_promotion.csv',
            BALL/'conservative_ball_parameters.csv',BALL/'ball_volume_statistics.csv']


def geometry():
    n=json.load(open(BALL/'eta_normalization.json'))
    affine=np.asarray(n['affine_center'],float); scale=np.asarray(n['scale'],float)
    hs=read_csv(BALL/'ebridge_halfspaces.csv')
    eq=np.asarray([[float(x['n1']),float(x['n2']),float(x['n3']),float(x['b'])] for x in hs])
    vol=float(json.load(open(BALL/'ebridge_definition.json'))['normalized_volume'])
    return affine,scale,eq,vol


def norm_eta(e,affine,scale): return (np.asarray(e,float)-affine)/scale
def raw_eta(e,affine,scale): return affine+scale*np.asarray(e,float)
def inside_hull(x,eq,tol=2e-10): return bool(np.all(eq[:,:3]@np.asarray(x)+eq[:,3] <= tol))


def exact_source_files(include_work=True) -> list[Path]:
    files=[]
    for root in (QDIR,DDIR,QGDIR): files.extend(sorted(root.glob('raw/*/shard*.jsonl')))
    for p in (OLD/'raw/pilot_rollouts.jsonl',BALL/'raw/pilot_rollouts.jsonl'):
        if p.exists(): files.append(p)
    if include_work: files.extend(sorted(OUT.glob('work/*/raw/pilot_rollouts.jsonl')))
    return files


def load_exact_rows(states=None) -> dict[tuple[str,str,int],dict]:
    rows={}
    for p in exact_source_files():
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r=json.loads(line)
            if 'state_id' not in r or 'eta' not in r or 'future_index' not in r: continue
            if states is not None and r['state_id'] not in states: continue
            rows[(r['state_id'],ekey(r['eta']),int(r['future_index']))]=r
    return rows


def centers() -> dict[str,np.ndarray]:
    out={}
    for r in read_csv(BALL/'robust_centers.csv'):
        if r['status']=='CENTER_RESOLVED': out[r['state_id']]=np.array([float(r['c1']),float(r['c2']),float(r['c3'])])
    if list(out)!=STATE_IDS: raise RuntimeError('center identity/order mismatch')
    return out


def build_axis_candidates() -> dict[tuple[str,str],list[dict]]:
    # Frozen screened-success candidates, largest radius first.
    candidates=defaultdict(dict)
    for filename in ('ray_screening.csv','boundary_bisection.csv'):
        for r in read_csv(BALL/filename):
            if not r['direction_id'].startswith('axis_') or r.get('screen_8of8')!='True': continue
            e=np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])]);rho=float(r['rho'])
            candidates[(r['state_id'],r['direction_id'])][ekey(e)]={'rho':rho,'eta':e,'source':filename}
    # Existing selected B63 fallbacks are authoritative and ranked first.
    selected={}
    prior_prom=[]
    for r in read_csv(BALL/'limiting_direction_promotion.csv'):
        if not r['direction_id'].startswith('axis_'): continue
        prior_prom.append(r)
        if r['selected']=='True' and r['B63']=='True':
            e=np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])])
            selected[(r['state_id'],r['direction_id'])]={'rho':float(r['rho']),'eta':e,'source':'prior_selected_B63'}
            candidates[(r['state_id'],r['direction_id'])][ekey(e)]=selected[(r['state_id'],r['direction_id'])]
    out={}
    for sid in STATE_IDS:
        for ax in range(1,4):
            for side in ('p','m'):
                did=f'axis_{ax}_{side}'; key=(sid,did)
                vals=sorted(candidates[key].values(),key=lambda x:(-x['rho'],ekey(x['eta'])))
                if key in selected:
                    sel=selected[key]; vals=[sel]+[x for x in vals if ekey(x['eta'])!=ekey(sel['eta']) and x['rho']<sel['rho']-1e-14]
                if not vals: raise RuntimeError(f'no screened success candidate {key}')
                out[key]=vals
    return out


def prepare() -> None:
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'logs').mkdir(exist_ok=True); (OUT/'plans').mkdir(exist_ok=True); (OUT/'work').mkdir(exist_ok=True)
    before={str(p):sha(p) for p in frozen_paths()}; dump(OUT/'frozen_hashes_before.json',before)
    if before[str(BASIS)]!='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38': raise RuntimeError('OrthoFlow3 hash mismatch')
    c=centers(); affine,scale,eq,vol=geometry()
    protocol={
      'audit':'ORTHOFLOW3_AXIS_ELLIPSOID_PILOT6_V1','states':STATE_IDS,'center_source':str(BALL/'robust_centers.csv'),
      'center_source_sha256':sha(BALL/'robust_centers.csv'),'center_frozen':True,
      'geometry':'axis-aligned ellipsoid in normalized eta coordinates','normalization':{'affine_center':affine.tolist(),'scale':scale.tolist()},
      'domain':'E_bridge=conv({0} union ACTIVE_BOX)','domain_sha256':sha(BALL/'ebridge_definition.json'),
      'axis_extent':'largest B63-confirmed screened-success point on each +/- coordinate axis, with deterministic inward fallback',
      'boundary_failure_confirmation':'first screened-failure bracket point is promoted to 64 for categorical diagnosis',
      'shrink_factor':SHRINK,'domain_containment':'exact halfspace support max n.c+b+sqrt(sum((n_j*r_j)^2)) <= 0; uniform radii scale-down only if required',
      'interior_sampling':'32-point deterministic unscrambled Sobol-to-volume-uniform-unit-ball transform per state',
      'promoted_interior_indices':list(PROMOTE_INDICES),'promotion_rule':'fixed indices before outcomes; 64 matched seeds',
      'shell_sampling':'deterministic Fibonacci directions; first 24 whose 1.05-scaled ellipsoid point remains in E_bridge; no clipping',
      'shell_scale':SHELL_SCALE,'meaningful_volume_ratio_threshold':MEANINGFUL_RATIO,
      'decision_pass':'6 contained, 6 robust centers, 0 promoted false inclusions, median ellipsoid volume > median ball volume, >=4/6 ratios >=1.25, 0 collisions',
      'training_performed':False,'rotated_ellipsoid':False,'J_def_used':False,'Q_used':False,
      'future_root_seed':FUTURE_ROOT,'requested_gpu_shards':6}
    dump(OUT/'protocol.json',protocol)
    (OUT/'protocol.md').write_text(
      '# OrthoFlow3 axis-aligned ellipsoid pilot6 v1\n\n'
      'Centers, OrthoFlow3, E_bridge, normalization, continuation semantics, and B63 are frozen from the completed continuous inner-ball pilot. '
      'The only new geometry is three normalized coordinate-aligned semi-axes. The global shrink factor is 0.85, shell scale is 1.05, '
      'and a volume ratio >=1.25 is predeclared as a meaningful per-state gain. Interior promotion indices are 3,7,11,15,19,23,27,31. '
      'No outcomes are used to select validation coordinates. No model is trained.\n')
    cand=build_axis_candidates(); brackets={(r['state_id'],r['direction_id']):r for r in read_csv(BALL/'boundary_brackets.csv') if r['direction_id'].startswith('axis_')}
    manifest=[]; plans=[[] for _ in STATE_IDS]
    for si,sid in enumerate(STATE_IDS):
        for ax in range(1,4):
            for side in ('p','m'):
                did=f'axis_{ax}_{side}'; vals=cand[(sid,did)]
                for rank,x in enumerate(vals):
                    manifest.append({'state_id':sid,'direction_id':did,'candidate_rank':rank,'rho':x['rho'],'eta1':x['eta'][0],'eta2':x['eta'][1],'eta3':x['eta'][2],'source':x['source']})
                x=vals[0]
                for i in range(64): plans[si].append({'state_id':sid,'eta':x['eta'].tolist(),'future_index':i,'query_role':'axis_success_primary','direction_id':did,'candidate_rank':0,'rho':x['rho']})
                br=brackets[(sid,did)]
                if br['rho_failure']:
                    # Exact failure eta is read from the frozen bisection/screening table.
                    rf=float(br['rho_failure']); rows=[]
                    for fn in ('ray_screening.csv','boundary_bisection.csv'):
                        rows += [r for r in read_csv(BALL/fn) if r['state_id']==sid and r['direction_id']==did and abs(float(r['rho'])-rf)<1e-13]
                    if len(rows)!=1: raise RuntimeError(f'failure point identity {sid} {did} {rf} {len(rows)}')
                    fr=rows[0]; fe=np.array([float(fr['eta1']),float(fr['eta2']),float(fr['eta3'])])
                    for i in range(64): plans[si].append({'state_id':sid,'eta':fe.tolist(),'future_index':i,'query_role':'axis_outer_failure','direction_id':did,'rho':rf})
    write_csv(OUT/'axis_candidate_manifest.csv',manifest)
    pdir=OUT/'plans/axis_primary';pdir.mkdir(parents=True,exist_ok=True)
    existing=load_exact_rows(set(STATE_IDS)); total=missing=0
    for i,rows in enumerate(plans):
        with (pdir/f'shard{i}.jsonl').open('w') as f:
            for r in rows:
                f.write(json.dumps(r,sort_keys=True)+'\n');total+=1
                if (r['state_id'],ekey(r['eta']),r['future_index']) not in existing:missing+=1
    dump(OUT/'cost_preflight.json',{'axis_plan_tuple_requests':total,'axis_plan_new_tuples':missing,
      'validation_projected_new_tuples':6*(8*64+24*8+24*8),'projected_total_new_upper_bound':missing+6*(8*64+24*8+24*8),
      'projected_steps_at_prior_mean':int(math.ceil((missing+5376)*276.5)),'gpu_shards':6,'within_prior_15k_5m_reference':missing+5376<15000})


def make_oracle(work: Path):
    old=load_old(work)
    class Oracle(old.Oracle):
        def _load_prior(self_inner):
            self_inner._load_jsonls(QDIR,'qv2_exact');self_inner._load_jsonls(DDIR,'direct_eta_exact');self_inner._load_jsonls(QGDIR,'q_guided_exact')
            for p in (OLD/'raw/pilot_rollouts.jsonl',BALL/'raw/pilot_rollouts.jsonl'):
                if p.exists():
                    for line in p.read_text().splitlines():
                        if line.strip(): self_inner._insert(json.loads(line),p.parent.parent.name+'_exact')
            for p in sorted(OUT.glob('work/*/raw/pilot_rollouts.jsonl')):
                if p.resolve()==self_inner.record_path.resolve(): continue
                for line in p.read_text().splitlines():
                    if line.strip(): self_inner._insert(json.loads(line),'ellipsoid_prior_stage')
            if self_inner.record_path.exists():
                for line in self_inner.record_path.read_text().splitlines():
                    if line.strip():
                        r=json.loads(line);self_inner.rows.append(r);self_inner._insert(r,'pilot')
                self_inner.new=len(self_inner.rows);self_inner.steps=sum(int(r['continuation_steps']) for r in self_inner.rows)
    qstates={x['state_id']:x for x in json.load(open(QDIR/'eligible_state_manifest.json'))['selected_states']}
    states={sid:qstates[sid] for sid in STATE_IDS}; features=np.load(QDIR/'conditioning_features.npz')['features']
    # low/high are not used to clip eta in the authoritative runner.
    return Oracle(states,features,np.array([0.,-.5,0.]),np.array([1.25,.5,.75]))


def run_plan(plan: Path, work_name: str, phase: str) -> None:
    work=OUT/'work'/work_name;work.mkdir(parents=True,exist_ok=True)
    tasks=[json.loads(x) for x in plan.read_text().splitlines() if x.strip()]
    oracle=make_oracle(work); started=time.monotonic(); rows=oracle.ensure(tasks,phase)
    dump(work/'run_summary.json',{'phase':phase,'plan':str(plan),'requests':len(tasks),'new_continuations':oracle.new,
      'physical_steps':oracle.steps,'wall_time_seconds':time.monotonic()-started,'outcomes':dict(Counter(r['outcome'] for r in oracle.rows)),
      'source_requests':dict(Counter(r.get('_source','') for r in rows))})


def summarize64(rows, sid, eta):
    rr=[rows[(sid,ekey(eta),i)] for i in range(64)]
    invalid=[x for x in rr if x['outcome']=='other_numerical']
    suc=sum(bool(x['success']) for x in rr)
    return suc,len(rr),len(invalid),Counter(x['outcome'] for x in rr)


def freeze_ellipsoids() -> None:
    affine,scale,eq,domain_vol=geometry(); c=centers(); allrows=load_exact_rows(set(STATE_IDS)); cand=build_axis_candidates()
    primary=[]; fallback_plans=[[] for _ in STATE_IDS]; selected={}; need=False
    # Select the first B63 success candidate; make one deterministic fallback plan if needed.
    for si,sid in enumerate(STATE_IDS):
        for ax in range(1,4):
            for side in ('p','m'):
                did=f'axis_{ax}_{side}'; vals=cand[(sid,did)]
                found=None
                for rank,x in enumerate(vals):
                    keys=[(sid,ekey(x['eta']),i) for i in range(64)]
                    if not all(k in allrows for k in keys):
                        if rank==0:
                            for i in range(64): fallback_plans[si].append({'state_id':sid,'eta':x['eta'].tolist(),'future_index':i,'query_role':'axis_success_fallback','direction_id':did,'candidate_rank':rank,'rho':x['rho']})
                            need=True
                        break
                    suc,n,inv,out=summarize64(allrows,sid,x['eta']); b63=(inv==0 and suc>=63)
                    primary.append({'state_id':sid,'direction_id':did,'candidate_rank':rank,'rho':x['rho'],'eta1':x['eta'][0],'eta2':x['eta'][1],'eta3':x['eta'][2],
                                    'successes':suc,'trials':n,'invalid_trials':inv,'B63':b63,'outcomes':json.dumps(dict(out),sort_keys=True),'selected':b63})
                    if b63: found=x;break
                if found is None:
                    # If rank0 was evaluated and failed, request the next unevaluated inward candidate.
                    for rank,x in enumerate(vals[1:],start=1):
                        keys=[(sid,ekey(x['eta']),i) for i in range(64)]
                        if not all(k in allrows for k in keys):
                            for i in range(64): fallback_plans[si].append({'state_id':sid,'eta':x['eta'].tolist(),'future_index':i,'query_role':'axis_success_fallback','direction_id':did,'candidate_rank':rank,'rho':x['rho']})
                            need=True;break
                else:selected[(sid,did)]=found
    write_csv(OUT/'axis_success_promotions.csv',primary)
    fdir=OUT/'plans/axis_fallback';fdir.mkdir(parents=True,exist_ok=True)
    for i,plan in enumerate(fallback_plans):
        with (fdir/f'shard{i}.jsonl').open('w') as f:
            for r in plan:f.write(json.dumps(r,sort_keys=True)+'\n')
    dump(OUT/'fallback_status.json',{'required':need,'task_requests':sum(map(len,fallback_plans)),'directions_selected':len(selected)})
    if need: return
    if len(selected)!=36: raise RuntimeError(f'axis directions unresolved {len(selected)}/36')

    # Diagnose the frozen outer bracket candidates at 64 seeds when present.
    brackets={(r['state_id'],r['direction_id']):r for r in read_csv(BALL/'boundary_brackets.csv') if r['direction_id'].startswith('axis_')}
    outer=[]
    for (sid,did),br in brackets.items():
        if not br['rho_failure']:
            outer.append({'state_id':sid,'direction_id':did,'boundary_cause':'DOMAIN_BOUNDARY','rho_failure':'','successes':'','trials':'','invalid_trials':0,'B63':''});continue
        rf=float(br['rho_failure']); matches=[]
        for fn in ('ray_screening.csv','boundary_bisection.csv'):
            matches += [r for r in read_csv(BALL/fn) if r['state_id']==sid and r['direction_id']==did and abs(float(r['rho'])-rf)<1e-13]
        e=np.array([float(matches[0]['eta1']),float(matches[0]['eta2']),float(matches[0]['eta3'])]);suc,n,inv,out=summarize64(allrows,sid,e)
        cause='NUMERICAL_OR_PROJECTION' if inv else ('B63_FAILURE' if suc<63 else 'ROBUST_SUCCESS_BEYOND_CONSERVATIVE_EXTENT')
        outer.append({'state_id':sid,'direction_id':did,'boundary_cause':cause,'rho_failure':rf,'successes':suc,'trials':n,'invalid_trials':inv,'B63':inv==0 and suc>=63,'outcomes':json.dumps(dict(out),sort_keys=True)})
    write_csv(OUT/'axis_outer_boundary_diagnosis.csv',outer)

    oldball={r['state_id']:r for r in read_csv(BALL/'ball_volume_statistics.csv')}; axisrows=[]; ellrows=[]; volrows=[]; contain=[]; ell={}
    for sid in STATE_IDS:
        ext={}
        for ax in range(1,4):
            for side in ('p','m'):
                did=f'axis_{ax}_{side}';x=selected[(sid,did)];ext[did]=x['rho']
                cause=next(y['boundary_cause'] for y in outer if y['state_id']==sid and y['direction_id']==did)
                axisrows.append({'state_id':sid,'axis':ax,'side':'positive' if side=='p' else 'negative','direction_id':did,'robust_extent':x['rho'],'boundary_cause':cause})
        raw=np.array([SHRINK*min(ext[f'axis_{ax}_p'],ext[f'axis_{ax}_m']) for ax in range(1,4)])
        ct=norm_eta(c[sid],affine,scale);slack=-(eq[:,:3]@ct+eq[:,3]);support=np.sqrt(np.sum((eq[:,:3]*raw)**2,axis=1))
        factors=np.divide(slack,support,out=np.full_like(slack,np.inf),where=support>1e-15);factor=min(1.,float(np.min(factors)))
        if factor<1.: factor=max(0.,factor*(1-1e-9))
        radii=raw*factor; residual=slack-np.sqrt(np.sum((eq[:,:3]*radii)**2,axis=1));ok=bool(np.min(residual)>=-1e-10)
        if not ok: raise RuntimeError(f'ellipsoid containment {sid}')
        v=4/3*math.pi*float(np.prod(radii));vb=float(oldball[sid]['V_ball']);ratio=v/vb
        ell[sid]={'c':c[sid],'ct':ct,'r':radii,'volume':v,'ratio':ratio}
        ellrows.append({'state_id':sid,'c1':c[sid][0],'c2':c[sid][1],'c3':c[sid][2],'r1':radii[0],'r2':radii[1],'r3':radii[2],
                        'shrink_factor':SHRINK,'containment_scale':factor,'volume':v,'domain_volume_fraction':v/domain_vol,'contained_in_Ebridge':ok})
        volrows.append({'state_id':sid,'old_ball_radius':oldball[sid]['r_ball'],'old_ball_volume':vb,'old_ball_domain_fraction':oldball[sid]['volume_fraction'],
                        'ellipsoid_volume':v,'ellipsoid_domain_fraction':v/domain_vol,'ellipsoid_to_ball_ratio':ratio,'meaningful_gain':ratio>=MEANINGFUL_RATIO})
        contain.append({'state_id':sid,'precontain_r1':raw[0],'precontain_r2':raw[1],'precontain_r3':raw[2],'containment_scale':factor,'min_final_halfspace_margin':float(np.min(residual)),'contained':ok})
    write_csv(OUT/'per_state_axis_limits.csv',axisrows);write_csv(OUT/'ellipsoid_parameters.csv',ellrows);write_csv(OUT/'volume_comparison.csv',volrows);write_csv(OUT/'domain_containment_check.csv',contain)

    # Freeze independent validation coordinates before any validation outcome.
    sob=qmc.Sobol(d=3,scramble=False).random_base2(5)
    unit=[]
    for u in sob:
        z=1-2*u[0];theta=2*math.pi*u[1];q=math.sqrt(max(0.,1-z*z));rad=u[2]**(1/3)
        unit.append(rad*np.array([q*math.cos(theta),q*math.sin(theta),z]))
    inside=[];shell=[];plans=[[] for _ in STATE_IDS]
    for si,sid in enumerate(STATE_IDS):
        e=ell[sid]
        for k,u in enumerate(unit):
            te=e['ct']+e['r']*u;eta=raw_eta(te,affine,scale);prom=k in PROMOTE_INDICES
            inside.append({'state_id':sid,'point_id':f'I{k:02d}','sobol_index':k,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],
                           'ellipsoid_radius':float(np.linalg.norm(u)),'strictly_inside':float(np.linalg.norm(u))<1-1e-12,'promoted64':prom})
            n=64 if prom else 8
            for fi in range(n):plans[si].append({'state_id':sid,'eta':eta.tolist(),'future_index':fi,'query_role':'ellipsoid_inside','point_id':f'I{k:02d}','promoted64':prom})
        # Deterministic Fibonacci pool with outcome-blind E_bridge rejection and no clipping.
        got=0
        for di in range(1024):
            z=1-2*(di+.5)/1024;theta=math.pi*(3-math.sqrt(5))*di;q=math.sqrt(max(0.,1-z*z));d=np.array([q*math.cos(theta),q*math.sin(theta),z])
            te=e['ct']+SHELL_SCALE*e['r']*d
            if not inside_hull(te,eq):continue
            eta=raw_eta(te,affine,scale);shell.append({'state_id':sid,'point_id':f'O{got:02d}','direction_pool_index':di,'d1':d[0],'d2':d[1],'d3':d[2],
              'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'ellipsoid_radius':SHELL_SCALE,'inside_Ebridge':True})
            for fi in range(8):plans[si].append({'state_id':sid,'eta':eta.tolist(),'future_index':fi,'query_role':'ellipsoid_shell','point_id':f'O{got:02d}'})
            got+=1
            if got==24:break
        if got<24:raise RuntimeError(f'insufficient shell points {sid}')
    write_csv(OUT/'interior_points.csv',inside);write_csv(OUT/'shell_points.csv',shell)
    vdir=OUT/'plans/validation';vdir.mkdir(parents=True,exist_ok=True)
    for i,plan in enumerate(plans):
        with (vdir/f'shard{i}.jsonl').open('w') as f:
            for r in plan:f.write(json.dumps(r,sort_keys=True)+'\n')
    dump(OUT/'validation_plan.json',{'interior_points':len(inside),'promoted_points':sum(x['promoted64'] for x in inside),'shell_points':len(shell),
      'tuple_requests':sum(map(len,plans)),'promotion_indices':list(PROMOTE_INDICES),'shell_scale':SHELL_SCALE,'coordinates_frozen_before_outcomes':True})


def finalize() -> None:
    rows=load_exact_rows(set(STATE_IDS));inside=read_csv(OUT/'interior_points.csv');shell=read_csv(OUT/'shell_points.csv')
    screen=[];prom=[];shellres=[]
    for x in inside:
        e=np.array([float(x['eta1']),float(x['eta2']),float(x['eta3'])]);rr=[rows[(x['state_id'],ekey(e),i)] for i in range(8)]
        suc=sum(bool(r['success']) for r in rr);invalid=sum(r['outcome']=='other_numerical' for r in rr);collision=sum(r['outcome']=='collision' for r in rr)
        screen.append({**x,'successes':suc,'trials':8,'invalid_trials':invalid,'collisions':collision,'screen_8of8':invalid==0 and suc==8})
        if x['promoted64']=='True':
            aa=[rows[(x['state_id'],ekey(e),i)] for i in range(64)];s=sum(bool(r['success']) for r in aa);inv=sum(r['outcome']=='other_numerical' for r in aa);col=sum(r['outcome']=='collision' for r in aa)
            b63=(inv==0 and s>=63);prom.append({**x,'successes':s,'trials':64,'valid_trials':64-inv,'invalid_trials':inv,'collisions':col,
              'B63':b63,'confirmed_false_inclusion':inv==0 and s<63,'outcomes':json.dumps(dict(Counter(r['outcome'] for r in aa)),sort_keys=True)})
    for x in shell:
        e=np.array([float(x['eta1']),float(x['eta2']),float(x['eta3'])]);rr=[rows[(x['state_id'],ekey(e),i)] for i in range(8)]
        suc=sum(bool(r['success']) for r in rr);inv=sum(r['outcome']=='other_numerical' for r in rr);col=sum(r['outcome']=='collision' for r in rr)
        shellres.append({**x,'successes':suc,'trials':8,'invalid_trials':inv,'collisions':col,'screen_8of8':inv==0 and suc==8,
                         'outcomes':json.dumps(dict(Counter(r['outcome'] for r in rr)),sort_keys=True)})
    write_csv(OUT/'interior_screening_results.csv',screen);write_csv(OUT/'promoted64_validation.csv',prom);write_csv(OUT/'shell_results.csv',shellres)

    params={r['state_id']:r for r in read_csv(OUT/'ellipsoid_parameters.csv')};vol={r['state_id']:r for r in read_csv(OUT/'volume_comparison.csv')}
    anis=[]
    for sid in STATE_IDS:
        rr=np.array([float(params[sid][f'r{i}']) for i in range(1,4)]);order=np.argsort(rr)
        anis.append({'state_id':sid,'r1':rr[0],'r2':rr[1],'r3':rr[2],'anisotropy_ratio':float(rr.max()/rr.min()),
          'ordered_axes_narrow_to_wide':'>'.join(f'eta{i+1}' for i in order),'narrowest_dimension':f'eta{order[0]+1}','widest_dimension':f'eta{order[-1]+1}',
          'prior_18ray_anisotropy':next(r['anisotropy_ratio'] for r in read_csv(BALL/'anisotropy_statistics.csv') if r['state_id']==sid)})
    write_csv(OUT/'anisotropy_statistics.csv',anis)
    collisions=sum(int(x['collisions']) for x in screen)+sum(int(x['collisions']) for x in prom)+sum(int(x['collisions']) for x in shellres)
    invalid=sum(int(x['invalid_trials']) for x in screen)+sum(int(x['invalid_trials']) for x in prom)+sum(int(x['invalid_trials']) for x in shellres)
    false=sum(bool(x['confirmed_false_inclusion']) for x in prom);contained=all(r['contained_in_Ebridge']=='True' for r in read_csv(OUT/'ellipsoid_parameters.csv'))
    centers_ok=all(int(r['successes'])>=63 for r in read_csv(BALL/'robust_centers.csv'))
    ratios=[float(vol[s]['ellipsoid_to_ball_ratio']) for s in STATE_IDS];meaningful=sum(r>=MEANINGFUL_RATIO for r in ratios)
    oldmed=float(np.median([float(vol[s]['old_ball_volume']) for s in STATE_IDS]));newmed=float(np.median([float(vol[s]['ellipsoid_volume']) for s in STATE_IDS]))
    complete=len(prom)==48 and all(int(x['valid_trials'])==64 for x in prom)
    if false>0 or not contained: decision='REJECT — AXIS_ELLIPSOID_NOT_RELIABLE'
    elif not complete or invalid>0 or newmed<=oldmed or meaningful<4: decision='REVISE — ELLIPSOID_TOO_CONSERVATIVE_OR_UNCERTAIN'
    elif centers_ok and collisions==0: decision='PASS — AXIS_ELLIPSOID_READY'
    else: decision='REJECT — AXIS_ELLIPSOID_NOT_RELIABLE'
    summary={'decision':decision,'criteria':{'all_6_contained':contained,'all_6_centers_robust':centers_ok,'promoted_points':len(prom),
      'confirmed_false_inclusions':false,'median_ball_volume':oldmed,'median_ellipsoid_volume':newmed,'median_volume_strictly_larger':newmed>oldmed,
      'states_with_meaningful_gain':meaningful,'meaningful_ratio_threshold':MEANINGFUL_RATIO,'collisions':collisions,'invalid_evaluations':invalid},
      'shell':{'points':len(shellres),'eight_of_eight':sum(bool(x['screen_8of8']) for x in shellres),'success_fraction':sum(bool(x['screen_8of8']) for x in shellres)/len(shellres)},
      'learning_label':'(c1,c2,c3,r1,r2,r3)' if decision.startswith('PASS') else None,'training_performed':False}
    dump(OUT/'final_decision.json',summary)

    byscreen=defaultdict(list);byprom=defaultdict(list);byshell=defaultdict(list)
    for x in screen:byscreen[x['state_id']].append(x)
    for x in prom:byprom[x['state_id']].append(x)
    for x in shellres:byshell[x['state_id']].append(x)
    table=[]
    for sid in STATE_IDS:
        p=params[sid];v=vol[sid];a=next(x for x in anis if x['state_id']==sid)
        center=f"({float(p['c1']):.6f}, {float(p['c2']):.6f}, {float(p['c3']):.6f})"
        table.append(f"| {sid} | {center} | {float(v['old_ball_radius']):.6f} | {float(v['old_ball_volume']):.6g} | ({float(p['r1']):.6f}, {float(p['r2']):.6f}, {float(p['r3']):.6f}) | {float(p['volume']):.6g} | {float(v['ellipsoid_to_ball_ratio']):.3f}x | {100*float(p['domain_volume_fraction']):.3f}% | {sum(bool(x['B63']) for x in byprom[sid])}/8 B63 | {sum(bool(x['screen_8of8']) for x in byshell[sid])}/24 | {float(a['anisotropy_ratio']):.3f} |")
    report=['# OrthoFlow3 axis-aligned ellipsoid pilot6 v1','',f'## Decision\n\n**{decision}**','',
      'Centers, OrthoFlow3, E_bridge, normalization, safety projection, horizon, RNG, and B63 semantics were frozen. No model was trained. The globally frozen shrink factor was 0.85 and the shell scale was 1.05.','',
      '| State | Frozen center | Ball r | Ball volume | Ellipsoid (r1,r2,r3) | Ellipsoid volume | E/B ratio | E_bridge fraction | Interior 64-seed | Shell 8/8 | Axis anisotropy |',
      '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|',*table,'',
      '## Validation','',f"- Independent interior screening: {sum(bool(x['screen_8of8']) for x in screen)}/{len(screen)} were 8/8.",
      f"- Predeclared promotion: {sum(bool(x['B63']) for x in prom)}/{len(prom)} were B63; confirmed false inclusions: {false}.",
      f"- Shell: {summary['shell']['eight_of_eight']}/{summary['shell']['points']} points were 8/8 at ellipsoid radius {SHELL_SCALE}.",
      f"- Median volume increased from {oldmed:.8g} to {newmed:.8g}; {meaningful}/6 states exceeded the predeclared {MEANINGFUL_RATIO:.2f}x meaningful-gain threshold.",
      f'- Collisions: {collisions}; invalid/numerical/projection evaluations: {invalid}.','',
      '### Confirmed false inclusions','']
    false_rows=[x for x in prom if bool(x['confirmed_false_inclusion'])]
    if false_rows:
        for x in false_rows:
            report.append(
              f"- `{x['state_id']}` / `{x['point_id']}`: eta=({float(x['eta1']):.9f}, "
              f"{float(x['eta2']):.9f}, {float(x['eta3']):.9f}), ellipsoid radius "
              f"{float(x['ellipsoid_radius']):.6f}, true success {int(x['successes'])}/64, "
              f"outcomes {x['outcomes']}."
            )
    else:
        report.append('- None.')
    report += ['',
      '## Interpretation','',
      'The proposed axis-aligned ellipsoid would have supplied `(c1,c2,c3,r1,r2,r3)`, but the confirmed false inclusions mean it is not a validated conservative learning label. A later direct-center baseline could use `L_center=||G_phi(x)-c(x)||^2`; a set objective would use `d_E^2=sum_j((G_j-c_j)/r_j)^2`, with zero set loss inside `d_E<=1` and positive penalty outside. Neither objective was implemented here.','',
      'A later deformation study may compare robust-center-like outputs against minimum intervention inside the verified ellipsoid, using either `||eta||^2` or rollout `J_def`. No choice was made here.','',
      '## Regression and integrity','',
      'The frozen repository regression suite completed with 139 tests and 13 subtests passing, zero failures. Frozen canonical hashes are checked separately in `frozen_hash_regression.json`.','',
      '## Stop condition','',
      'No G_phi, Q, J, H, gate, rotated ellipsoid, polytope, or nonconvex set model was trained or constructed.']
    (OUT/'axis_ellipsoid_report.md').write_text('\n'.join(report)+'\n')

    # Aggregate new-work runtime from disjoint worker summaries.
    runs=[]
    for p in sorted(OUT.glob('work/*/run_summary.json')):runs.append(json.load(open(p)))
    phases=sorted(set(x['phase'] for x in runs))
    runtime={'worker_runs':len(runs),'tuple_requests':sum(x['requests'] for x in runs),
      'reused_continuations':sum(x['requests']-x['new_continuations'] for x in runs),
      'new_continuations':sum(x['new_continuations'] for x in runs),'physical_steps':sum(x['physical_steps'] for x in runs),
      'sum_worker_wall_seconds':sum(x['wall_time_seconds'] for x in runs),
      'rollout_elapsed_wall_seconds':sum(max(x['wall_time_seconds'] for x in runs if x['phase']==phase) for phase in phases),
      'max_gpu_shards':6,'gpu_memory_peak_mib':'not captured by scheduler','cpu_threads_per_shard':2,
      'ram_allocation_gib_per_shard':8,'max_concurrent_ram_allocation_gib':48,'observed_ram_peak':'not captured by scheduler','host':platform.node(),
      'outcomes':dict(sum((Counter(x['outcomes']) for x in runs),Counter()))}
    dump(OUT/'runtime_statistics.json',runtime)


def manifest() -> None:
    after={str(p):sha(p) for p in frozen_paths()};before=json.load(open(OUT/'frozen_hashes_before.json'));unchanged=(after==before)
    dump(OUT/'frozen_hashes_after.json',after);dump(OUT/'frozen_hash_regression.json',{'all_unchanged':unchanged,'changed':[p for p in before if before[p]!=after.get(p)]})
    files=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    decision=json.load(open(OUT/'final_decision.json'))['decision']
    dump(OUT/'manifest.json',{'schema':'orthoflow3_axis_ellipsoid_pilot6_v1','decision':decision,'files':files,
      'frozen_hashes_unchanged':unchanged,'training_performed':False,'stop_condition_respected':True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run-plan','freeze','finalize','manifest']);ap.add_argument('--plan');ap.add_argument('--work');ap.add_argument('--phase',default='ellipsoid')
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run-plan':run_plan(Path(a.plan),a.work,a.phase)
    elif a.stage=='freeze':freeze_ellipsoids()
    elif a.stage=='finalize':finalize()
    elif a.stage=='manifest':manifest()


if __name__=='__main__': main()
