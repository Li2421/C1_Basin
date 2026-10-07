#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc

ROOT = Path('/home/zhihan/research/Basin_C1')
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
HERE = ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1'
OLD = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_pilot6_v1'
REF = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_v1'
GAP = ROOT/'diagnostics/orthoflow3_zero_active_gap_connectivity_v1'
QDIR = ROOT/'diagnostics/orthoflow3_q_learnability_v2'
DDIR = ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
QGDIR = ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1'
BASIS = ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
STATE_IDS = ['N_r104_s125','ZR_P_r052_m080_s95400004_p253','S_r043_p02',
             'N_r076_s119','R_D2_s95101009_p112','N_r004_m120']
KAPPA = .85
RAY_RADII = (.025,.05,.10,.20,.35,.50)
CAP_CONT = 15000
CAP_STEPS = 5000000


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(p: Path, x) -> None:
    p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')


def csv_write(p: Path, rows: list[dict], fields: list[str]) -> None:
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def ekey(e) -> str:
    return np.asarray(e,dtype=np.float64).tobytes().hex()


def load_old_module():
    spec=importlib.util.spec_from_file_location('old_ball_pilot',OLD/'pilot6.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    mod.HERE=HERE;mod.CAP_CONT=CAP_CONT;mod.CAP_STEPS=CAP_STEPS
    return mod


def geometry():
    norm=json.load(open(REF/'eta_normalization.json'))
    affine=np.asarray(norm['affine_center'],float);scale=np.asarray(norm['scale_high_minus_low'],float)
    lo=np.asarray(norm['active_eta_box_low'],float);hi=np.asarray(norm['active_eta_box_high'],float)
    vertices=[np.zeros(3)]
    for x in (lo[0],hi[0]):
        for y in (lo[1],hi[1]):
            for z in (lo[2],hi[2]): vertices.append(np.array([x,y,z],float))
    vertices=np.asarray(vertices);tv=(vertices-affine)/scale
    hull=ConvexHull(tv)
    equations=np.asarray(hull.equations,float) # unit outward normal, offset, <=0 inside
    return norm,affine,scale,lo,hi,vertices,tv,hull,equations


def inside_hull(x,eq,tol=2e-10):
    return bool(np.all(eq[:,:3]@np.asarray(x)+eq[:,3] <= tol))


def domain_clearance(x,eq):
    vals=-(eq[:,:3]@np.asarray(x)+eq[:,3])/np.linalg.norm(eq[:,:3],axis=1)
    return float(np.min(vals))


def ray_limit(x,d,eq):
    vals=[]
    for row in eq:
        den=float(row[:3]@d)
        if den>1e-13: vals.append(float(-(row[:3]@x+row[3])/den))
    return max(0.,min(vals)) if vals else math.inf


def sobol_bridge(eq,affine,scale,n=64):
    # Bounding box is exact in physical coordinates: [0,1.25]x[-.5,.5]x[0,.75].
    low=np.array([0.,-.5,0.]);high=np.array([1.25,.5,.75])
    engine=qmc.Sobol(d=3,scramble=False);accepted=[];source_index=0
    while len(accepted)<n:
        u=engine.random(256)
        for v in u:
            eta=low+(high-low)*v;te=(eta-affine)/scale
            if inside_hull(te,eq):
                accepted.append({'candidate_index':len(accepted),'sobol_source_index':source_index,
                                 'u1':v[0],'u2':v[1],'u3':v[2],
                                 'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],
                                 't1':te[0],'t2':te[1],'t3':te[2]})
                if len(accepted)==n: break
            source_index+=1
    return accepted


def source_files():
    out=[]
    for root in (QDIR,DDIR,QGDIR): out.extend(sorted(root.glob('raw/*/shard*.jsonl')))
    if (OLD/'raw/pilot_rollouts.jsonl').exists(): out.append(OLD/'raw/pilot_rollouts.jsonl')
    return out


def compatible_rows(states):
    rows={}
    scanned=0
    for p in source_files():
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r=json.loads(line);scanned+=1
            if r.get('state_id') not in states or 'future_index' not in r or 'eta' not in r: continue
            rows[(r['state_id'],ekey(r['eta']),int(r['future_index']))]=r
    return rows,scanned


def prepare():
    HERE.mkdir(parents=True,exist_ok=True);(HERE/'raw').mkdir(exist_ok=True);(HERE/'logs').mkdir(exist_ok=True)
    prior=json.load(open(OLD/'pilot_state_manifest.json'))['states']
    if [x['state_id'] for x in prior] != STATE_IDS: raise RuntimeError('frozen six-state identity mismatch')
    dump(HERE/'frozen_state_manifest.json',{'schema':'orthoflow3_continuous_inner_ball_pilot6_v1_states',
         'controlled_rerun_of':str(OLD/'pilot_state_manifest.json'),'source_sha256':sha(OLD/'pilot_state_manifest.json'),'states':prior})
    if sha(BASIS)!='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38': raise RuntimeError('basis hash')
    dump(HERE/'orthoflow3_manifest.json',{'implementation':str(BASIS),'sha256':sha(BASIS),
         'basis_interface':str(ROOT/'shared_control/basis_families.py'),'basis_interface_sha256':sha(ROOT/'shared_control/basis_families.py'),
         'controller_modified':False,'projection_modified':False})
    norm,affine,scale,lo,hi,vertices,tv,hull,eq=geometry()
    archived=json.load(open(GAP/'bridge_domain_definition.json'))
    if archived['definition']!='conv({0} union ACTIVE_BOX) = { alpha*x : alpha in [0,1], x in ACTIVE_BOX }': raise RuntimeError('archived bridge definition')
    ebridge={'definition':archived['definition'],'archived_source':str(GAP/'bridge_domain_definition.json'),
             'archived_sha256':sha(GAP/'bridge_domain_definition.json'),'implementation_validity_source':str(GAP/'implementation_domain_audit.md'),
             'implementation_validity_sha256':sha(GAP/'implementation_domain_audit.md'),
             'physical_vertices':vertices.tolist(),'normalized_vertices':tv.tolist(),
             'halfspace_convention':'n dot eta_tilde + b <= 0','normalized_volume':float(hull.volume),
             'normalization':{'affine_center':affine.tolist(),'scale':scale.tolist()},
             'hull_vertex_indices':[int(x) for x in hull.vertices]}
    dump(HERE/'ebridge_definition.json',ebridge)
    hs=[]
    for i,row in enumerate(eq): hs.append({'halfspace_index':i,'n1':row[0],'n2':row[1],'n3':row[2],'b':row[3],'normal_norm':np.linalg.norm(row[:3])})
    csv_write(HERE/'ebridge_halfspaces.csv',hs,list(hs[0]))
    dump(HERE/'eta_normalization.json',{'rule':'eta_tilde=(eta-affine_center)/scale','affine_center':affine.tolist(),'scale':scale.tolist(),
         'archived_source':str(REF/'eta_normalization.json'),'archived_sha256':sha(REF/'eta_normalization.json')})
    cloud=sobol_bridge(eq,affine,scale,64);csv_write(HERE/'ebridge_sobol_sequence.csv',cloud,list(cloud[0]))
    directions=list(csv.DictReader(open(REF/'ray_directions.csv')))
    if len(directions)!=18: raise RuntimeError('ray direction count')
    csv_write(HERE/'frozen_ray_directions.csv',directions,list(directions[0]))
    rows,scanned=compatible_rows(set(STATE_IDS))
    grouped=defaultdict(list)
    for (sid,ek,i),r in rows.items(): grouped[(sid,ek)].append(r)
    b63=[];nonb63=[]
    for (sid,ek),rr in grouped.items():
        ii={int(x['future_index']):x for x in rr}
        if all(i in ii for i in range(64)):
            suc=sum(bool(ii[i]['success']) for i in range(64));rec={'state_id':sid,'eta':ii[0]['eta'],'successes':suc}
            (b63 if suc>=63 else nonb63).append(rec)
    cache={'exact_compatibility':'Q-v2 h/current-Flow conditioning, future_root=2026092702, future_index, fixed eta, same horizon/projections/hash',
           'compatible_source_files':[str(x) for x in source_files()],'records_scanned':scanned,'compatible_exact_tuples':len(rows),
           'known_B63_state_eta':b63,'known_nonB63_state_eta':nonb63,
           'skipped_sources':{
             'zero_active_gap_connectivity_v1':'migration matched-flow seed semantics differ from Q-v2 h-conditioned future_root',
             'representation/local/bilateral/min-def':'continuation RNG or h-conditioning identity not exact',
             'old_pilot_quarantined_queries':'now implementation-valid in E_bridge and included because tuple semantics are exact'},
           'new_rollouts_launched_during_cache_audit':0}
    dump(HERE/'cache_reuse_audit.json',cache)
    protocol=f'''# OrthoFlow3 continuous-domain inner-ball pilot6 v1

The state population is the exact six-state controlled rerun specified by the protocol. Geometry uses the archived normalization `eta_tilde=(eta-[0.875,0,0.375])/[0.75,1,0.75]` and the exact convex hull of zero plus all eight ACTIVE-box vertices. All candidate/ray/inside/shell coordinates are frozen algorithmically before their respective outcomes.

- Common cloud: one deterministic unscrambled Sobol rejection sequence over the E_bridge bounding box; first 32 accepted points, with the next 32 as the sole predeclared extension.
- Center: maximize `min(d_fail,d_domain)` over 8/8 or exact B63 candidates; no J/Q/eta-norm criterion.
- Rays: the archived 18 directions, radii {RAY_RADII}, 0.95 domain-boundary probe, and at most three fixed bisections.
- Robustness: B63 is >=63/64; 8/8 is screening only. Eight smallest directions are promoted, with inward fallback and at most ten directions.
- Ball: `r_ball=.85*min(r_success,r_domain)` and must be wholly contained in E_bridge.
- Validation: 12 independent Sobol-to-ball points, three predeclared robust promotions at approximately .25/.60/.90 radius, all screening failures promoted, plus eight independent 1.15-radius shell points.
- No neural model is trained; learned Q and J_def are not used for construction.
'''
    (HERE/'protocol.md').write_text(protocol)
    # Expected adaptive cost uses the prior pilot's observed 276.5 steps/continuation.
    projected={'common_stage32':6*33*8,'possible_extension_three_states':3*32*8,
      'center_promotions_estimate':3*56,'ray_screen_and_bisection_estimate':6*18*7*8,
      'limiting_direction_promotions_estimate':6*8*56,'inside_screen_and_mandatory_promotions':6*12*8+6*3*56,
      'outside_shell':6*8*8}
    projected_total=sum(projected.values());projected_steps=int(math.ceil(projected_total*276.5))
    dump(HERE/'cost_preflight.json',{'components':projected,'projected_new_continuations_before_cache_reuse':projected_total,
      'projected_physical_steps':projected_steps,'continuation_cap':CAP_CONT,'physical_step_cap':CAP_STEPS,
      'within_caps':projected_total<=CAP_CONT and projected_steps<=CAP_STEPS,
      'policy':'staged adaptive protocol; runtime hard stops remain 15000 continuations and 5M steps'})
    print(json.dumps({'states':STATE_IDS,'normalized_volume':hull.volume,'halfspaces':len(eq),'sobol64_sha256':sha(HERE/'ebridge_sobol_sequence.csv'),
      'compatible_tuples':len(rows),'known_B63_pairs':len(b63),'known_nonB63_pairs':len(nonb63),'projected':json.load(open(HERE/'cost_preflight.json'))},indent=2))


class ContinuousOracle:
    def __init__(self, old, states,features,lo,hi):
        class Oracle(old.Oracle):
            def _load_prior(self_inner):
                self_inner._load_jsonls(QDIR,'qv2_exact');self_inner._load_jsonls(DDIR,'direct_eta_exact');self_inner._load_jsonls(QGDIR,'q_guided_exact')
                p=OLD/'raw/pilot_rollouts.jsonl'
                if p.exists():
                    for line in p.read_text().splitlines():
                        if line.strip(): self_inner._insert(json.loads(line),'prior_pilot_exact')
                if self_inner.record_path.exists():
                    for line in self_inner.record_path.read_text().splitlines():
                        if line.strip():
                            r=json.loads(line);self_inner.rows.append(r);self_inner._insert(r,'pilot')
                    self_inner.new=len(self_inner.rows);self_inner.steps=sum(int(r['continuation_steps']) for r in self_inner.rows)
        self.obj=Oracle(states,features,lo,hi)


def run():
    if not (HERE/'cost_preflight.json').exists(): raise RuntimeError('run prepare first')
    pre=json.load(open(HERE/'cost_preflight.json'))
    if not pre['within_caps']: raise RuntimeError('preflight exceeds caps')
    old=load_old_module()
    selection=json.load(open(HERE/'frozen_state_manifest.json'))['states']
    qstates={x['state_id']:x for x in json.load(open(QDIR/'eligible_state_manifest.json'))['selected_states']}
    states={sid:qstates[sid] for sid in STATE_IDS}
    norm,affine,scale,lo,hi,vertices,tv,hull,eq=geometry()
    features=np.load(QDIR/'conditioning_features.npz')['features']
    oracle=ContinuousOracle(old,states,features,lo,hi).obj
    started=time.monotonic()
    cloud=list(csv.DictReader(open(HERE/'ebridge_sobol_sequence.csv')))
    directions=list(csv.DictReader(open(HERE/'frozen_ray_directions.csv')))
    cache=json.load(open(HERE/'cache_reuse_audit.json'))
    known_b63=defaultdict(list);known_non=defaultdict(list)
    for r in cache['known_B63_state_eta']:known_b63[r['state_id']].append(r)
    for r in cache['known_nonB63_state_eta']:known_non[r['state_id']].append(r)
    taskset=old.taskset;rows_summary=old.rows_summary;norm_eta=old.norm_eta;raw_eta=old.raw_eta

    common=[];extension=[];screen_by=defaultdict(list)
    def screen_candidates(sid,entries,phase):
        tasks=[]
        for idx,r,stage in entries:
            eta=np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])])
            tasks.extend(taskset(sid,eta,8,candidate_index=idx,candidate_stage=stage))
        rows=oracle.ensure(tasks,f'common_{phase}');result=[]
        for j,(idx,r,stage) in enumerate(entries):
            eta=np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])]);out=rows[j*8:(j+1)*8]
            suc,n=rows_summary(out);rec={'state_id':sid,'candidate_index':idx,'stage':stage,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],
              'successes':suc,'trials':n,'screen_8of8':suc==8,'deadlock':sum(x['outcome']=='safe_deadlock' for x in out),
              'timeout':sum(x['outcome']=='timeout' for x in out),'collision':sum(x['outcome']=='collision' for x in out),'numerical':sum(x['outcome']=='other_numerical' for x in out)}
            screen_by[sid].append(rec);result.append(rec)
        return result
    for sid in STATE_IDS:
        z={'eta1':0.,'eta2':0.,'eta3':0.};entries=[(-1,z,'zero')]+[(i,cloud[i],'stage32') for i in range(32)]
        common.extend(screen_candidates(sid,entries,'stage32'))
    csv_write(HERE/'common_cloud_results.csv',common,list(common[0]))

    center_scores=[];center_prom=[];centers={};center_meta={}
    def candidate_ranking(sid,stage_limit,failed_exact):
        queried=[x for x in screen_by[sid] if x['candidate_index']==-1 or 0<=int(x['candidate_index'])<stage_limit]
        failures=[np.array([x['eta1'],x['eta2'],x['eta3']]) for x in queried if not x['screen_8of8']]
        failures += [np.asarray(x['eta'],float) for x in known_non[sid]]
        candidates=[];seen=set()
        for x in queried:
            if x['screen_8of8']:
                e=np.array([x['eta1'],x['eta2'],x['eta3']]);candidates.append((int(x['candidate_index']),f'cloud_{x["candidate_index"]}',e,False))
        for j,x in enumerate(sorted(known_b63[sid],key=lambda z:ekey(z['eta']))):
            e=np.asarray(x['eta'],float);candidates.append((100000+j,f'archived_B63_{j}',e,True))
        ranked=[]
        for idx,cid,e,is_b63 in candidates:
            k=ekey(e)
            if k in seen or k in failed_exact:continue
            seen.add(k);te=norm_eta(e,affine,scale);dd=domain_clearance(te,eq)
            df=min((np.linalg.norm(te-norm_eta(f,affine,scale)) for f in failures),default=float('inf'))
            score=min(dd,df);ranked.append(( -score,idx,cid,e,is_b63,score,df,dd,len(failures)==0))
        ranked.sort(key=lambda x:(x[0],x[1],x[2]));return ranked
    for sid in STATE_IDS:
        failed=set();accepted=None;stage_limit=32;stage_name='stage32'
        while True:
            ranked=candidate_ranking(sid,stage_limit,failed)
            for rank,item in enumerate(ranked):
                _,idx,cid,e,archived,score,df,dd,unobs=item
                center_scores.append({'state_id':sid,'stage':stage_name,'rank':rank,'candidate_id':cid,'candidate_index':idx,
                  'eta1':e[0],'eta2':e[1],'eta3':e[2],'d_fail':df,'d_domain':dd,'margin_score':score,
                  'failure_distance_unobserved':unobs,'archived_B63':archived})
                out=oracle.ensure(taskset(sid,e,64,center_candidate=cid,center_rank=rank),f'center_B63_{stage_name}')
                suc,n=rows_summary(out);b63=suc>=63
                center_prom.append({'state_id':sid,'stage':stage_name,'rank':rank,'candidate_id':cid,'eta1':e[0],'eta2':e[1],'eta3':e[2],
                  'successes':suc,'trials':n,'B63':b63,'selected':b63})
                if b63:
                    accepted=(e,score,df,dd,suc,cid);break
                failed.add(ekey(e))
            if accepted is not None:break
            if stage_limit==32:
                extension.extend(screen_candidates(sid,[(i,cloud[i],'extension64') for i in range(32,64)],'extension64'))
                stage_limit=64;stage_name='extension64';continue
            break
        if accepted:
            e,score,df,dd,suc,cid=accepted;centers[sid]=e;center_meta[sid]={'margin_score':score,'d_fail':df,'r_domain':dd,'Q64':suc/64,'successes':suc,'candidate_id':cid}
    csv_write(HERE/'common_cloud_extension.csv',extension,list(common[0]))
    csv_write(HERE/'center_scores.csv',center_scores,list(center_scores[0]))
    csv_write(HERE/'center_promotion.csv',center_prom,list(center_prom[0]))
    robust=[]
    for sid in STATE_IDS:
        if sid in centers:
            e=centers[sid];m=center_meta[sid];robust.append({'state_id':sid,'status':'CENTER_RESOLVED','c1':e[0],'c2':e[1],'c3':e[2],**m})
        else:robust.append({'state_id':sid,'status':'CENTER_UNRESOLVED','c1':'','c2':'','c3':'','margin_score':'','d_fail':'','r_domain':'','Q64':'','successes':'','candidate_id':''})
    csv_write(HERE/'robust_centers.csv',robust,list(robust[0]))

    limits=[];rayrows=[];brackets=[];bisrows=[];details={}
    for sid,c in centers.items():
        ct=norm_eta(c,affine,scale);state_detail=[]
        for dr in directions:
            did=dr['direction_id'];d=np.array([float(dr['d1']),float(dr['d2']),float(dr['d3'])]);rd=ray_limit(ct,d,eq)
            limits.append({'state_id':sid,'direction_id':did,'rho_domain':rd})
            rhos=[r for r in RAY_RADII if r<rd-1e-12];near=.95*rd
            if rd>1e-12 and all(abs(near-r)>1e-10 for r in rhos):rhos.append(near)
            center_item={'rho':0.,'eta':c,'successes':center_meta[sid]['successes'],'screen':True,'stage':'center'}
            state_detail.append({'direction_id':did,'d':d,'rho_domain':rd,'grid':sorted(set(rhos)),'next':0,
              'tested':[center_item],'last':center_item,'failure':None})
        # Evaluate one outward radius per still-active direction in each batch.
        while True:
            entries=[];tasks=[]
            for item in state_detail:
                if item['failure'] is not None or item['next']>=len(item['grid']):continue
                rho=item['grid'][item['next']];e=raw_eta(ct+rho*item['d'],affine,scale);entries.append((item,rho,e))
                tasks.extend(taskset(sid,e,8,direction_id=item['direction_id'],rho=rho))
            if not entries:break
            rows=oracle.ensure(tasks,f'ray_screen_round_{max(x[0]["next"] for x in entries)}')
            for j,(item,rho,e) in enumerate(entries):
                out=rows[j*8:(j+1)*8];suc,n=rows_summary(out);x={'rho':rho,'eta':e,'successes':suc,'screen':suc==8,'stage':'fixed_or_domain'}
                item['tested'].append(x);item['next']+=1
                rayrows.append({'state_id':sid,'direction_id':item['direction_id'],'rho':rho,'eta1':e[0],'eta2':e[1],'eta3':e[2],'successes':suc,'trials':n,'screen_8of8':suc==8,'stage':'fixed_or_domain'})
                if suc==8:item['last']=x
                else:item['failure']=x
        failing=[x for x in state_detail if x['failure'] is not None]
        for item in failing:item['lowr']=item['last']['rho'];item['highr']=item['failure']['rho']
        for bi in range(1,4):
            entries=[];tasks=[]
            for item in failing:
                rho=(item['lowr']+item['highr'])/2;e=raw_eta(ct+rho*item['d'],affine,scale);entries.append((item,rho,e))
                tasks.extend(taskset(sid,e,8,direction_id=item['direction_id'],rho=rho,bisection_round=bi))
            if not entries:break
            rows=oracle.ensure(tasks,f'ray_bisect_round_{bi}')
            for j,(item,rho,e) in enumerate(entries):
                out=rows[j*8:(j+1)*8];suc,n=rows_summary(out);x={'rho':rho,'eta':e,'successes':suc,'screen':suc==8,'stage':f'bisection_{bi}'}
                item['tested'].append(x);bisrows.append({'state_id':sid,'direction_id':item['direction_id'],'round':bi,'rho':rho,'eta1':e[0],'eta2':e[1],'eta3':e[2],'successes':suc,'trials':n,'screen_8of8':suc==8})
                if suc==8:item['lowr']=rho;item['last']=x
                else:item['highr']=rho;item['failure']=x
        for item in state_detail:
            if item['failure'] is None:brackets.append({'state_id':sid,'direction_id':item['direction_id'],'rho_success':item['last']['rho'],'rho_failure':'','success_to_domain_boundary':True})
            else:brackets.append({'state_id':sid,'direction_id':item['direction_id'],'rho_success':item['lowr'],'rho_failure':item['highr'],'success_to_domain_boundary':False})
            item['estimate']=item['last']['rho']
        details[sid]=state_detail
    csv_write(HERE/'domain_ray_limits.csv',limits,['state_id','direction_id','rho_domain'])
    csv_write(HERE/'ray_screening.csv',rayrows,list(rayrows[0]) if rayrows else ['state_id'])
    csv_write(HERE/'boundary_brackets.csv',brackets,list(brackets[0]) if brackets else ['state_id'])
    csv_write(HERE/'boundary_bisection.csv',bisrows,list(bisrows[0]) if bisrows else ['state_id'])

    prom=[];balls={};ballrows=[];contain=[]
    for sid,c in centers.items():
        ds=sorted(details[sid],key=lambda x:(x['estimate'],x['direction_id']));selected=ds[:8];confirmed=[]
        for rank,item in enumerate(selected):
            good=sorted([x for x in item['tested'] if x['screen']],key=lambda x:x['rho'],reverse=True);picked=None
            for gi,x in enumerate(good):
                out=oracle.ensure(taskset(sid,x['eta'],64,direction_id=item['direction_id'],rho=x['rho'],limiting_rank=rank),f'limit_B63_{item["direction_id"]}')
                suc,n=rows_summary(out);b63=suc>=63
                prom.append({'state_id':sid,'direction_id':item['direction_id'],'limiting_rank':rank,'fallback_rank':gi,'rho':x['rho'],
                  'eta1':x['eta'][0],'eta2':x['eta'][1],'eta3':x['eta'][2],'successes':suc,'trials':n,'B63':b63,'selected':b63})
                if b63:picked=x;break
            if picked is not None:confirmed.append((item['direction_id'],picked['rho']))
        if len(confirmed)==8:
            rs=min(x[1] for x in confirmed);rd=center_meta[sid]['r_domain'];rr=min(rs,rd);rb=KAPPA*rr
            ct=norm_eta(c,affine,scale);slacks=-(eq[:,:3]@ct+eq[:,3])-rb*np.linalg.norm(eq[:,:3],axis=1);ok=bool(np.min(slacks)>=-1e-10)
            contain.append({'state_id':sid,'min_halfspace_slack':float(np.min(slacks)),'contained_in_Ebridge':ok})
            if not ok:raise RuntimeError(f'domain containment failure {sid}')
            balls[sid]={'c':c,'ct':ct,'r_success':rs,'r_domain':rd,'r_raw':rr,'r_ball':rb}
            ballrows.append({'state_id':sid,'status':'BALL_RESOLVED','c1':c[0],'c2':c[1],'c3':c[2],'r_success':rs,'r_domain':rd,'r_raw':rr,'r_ball':rb,'kappa':KAPPA})
        else:
            ballrows.append({'state_id':sid,'status':'BALL_UNRESOLVED','c1':c[0],'c2':c[1],'c3':c[2],'r_success':'','r_domain':center_meta[sid]['r_domain'],'r_raw':'','r_ball':'','kappa':KAPPA})
    csv_write(HERE/'limiting_direction_promotion.csv',prom,list(prom[0]) if prom else ['state_id'])
    csv_write(HERE/'conservative_ball_parameters.csv',ballrows,list(ballrows[0]))
    csv_write(HERE/'domain_containment_check.csv',contain,list(contain[0]) if contain else ['state_id'])

    # Independent points are frozen completely before any inside outcome.
    u=qmc.Sobol(d=3,scramble=False).random_base2(5)[1:25:2][:12]
    unit=[]
    for x in u:
        z=1-2*x[0];theta=2*math.pi*x[1];rr=math.sqrt(max(0.,1-z*z));rad=x[2]**(1/3);unit.append(rad*np.array([rr*math.cos(theta),rr*math.sin(theta),z]))
    insidepts=[];mandatory={};
    for sid,b in balls.items():
        rel=np.array([np.linalg.norm(x) for x in unit]);chosen=[]
        for target in (.25,.60,.90):
            available=[i for i in range(12) if i not in chosen];k=min(available,key=lambda i:(abs(rel[i]-target),i));chosen.append(k)
        mandatory[sid]=set(chosen)
        for i,x in enumerate(unit):
            e=raw_eta(b['ct']+b['r_ball']*x,affine,scale);insidepts.append({'state_id':sid,'point_id':f'I{i:02d}','eta1':e[0],'eta2':e[1],'eta3':e[2],
              'relative_radius':rel[i],'normalized_radius':rel[i]*b['r_ball'],'mandatory_promotion':i in mandatory[sid]})
    csv_write(HERE/'inside_ball_points.csv',insidepts,list(insidepts[0]) if insidepts else ['state_id'])
    inscreen=[];tasks=[]
    for x in insidepts:
        e=np.array([x['eta1'],x['eta2'],x['eta3']]);tasks.extend(taskset(x['state_id'],e,8,point_id=x['point_id']))
    rows=oracle.ensure(tasks,'inside_screen_all') if tasks else []
    for j,x in enumerate(insidepts):
        out=rows[j*8:(j+1)*8];suc,n=rows_summary(out);inscreen.append({**x,'successes':suc,'trials':n,'screen_8of8':suc==8})
    csv_write(HERE/'inside_ball_screening.csv',inscreen,list(inscreen[0]) if inscreen else ['state_id'])
    inprom=[]
    for x in inscreen:
        if not (x['mandatory_promotion'] or not x['screen_8of8']):continue
        e=np.array([x['eta1'],x['eta2'],x['eta3']]);out=oracle.ensure(taskset(x['state_id'],e,64,point_id=x['point_id']),f'inside_B63_{x["point_id"]}')
        suc,n=rows_summary(out);inprom.append({**x,'successes64':suc,'trials64':n,'B63':suc>=63,'false_inclusion':suc<63})
    csv_write(HERE/'inside_ball_promoted64.csv',inprom,list(inprom[0]) if inprom else ['state_id'])
    fi={'inside_points_screened':len(inscreen),'screen_8of8':sum(x['screen_8of8'] for x in inscreen),'screen_failures':sum(not x['screen_8of8'] for x in inscreen),
        'promoted_to_64':len(inprom),'promoted_B63':sum(x['B63'] for x in inprom),'robust_false_inclusions':sum(x['false_inclusion'] for x in inprom),
        'robust_false_inclusion_rate':sum(x['false_inclusion'] for x in inprom)/len(inprom) if inprom else None}
    dump(HERE/'false_inclusion_summary.json',fi)

    # Frozen 64-direction Fibonacci pool; take first eight shell-valid directions per state.
    shell_dirs=[]
    golden=math.pi*(3-math.sqrt(5))
    for i in range(64):
        z=1-2*(i+.5)/64;rad=math.sqrt(1-z*z);shell_dirs.append(np.array([rad*math.cos(i*golden),rad*math.sin(i*golden),z]))
    shellpts=[]
    for sid,b in balls.items():
        k=0
        for di,d in enumerate(shell_dirs):
            e=raw_eta(b['ct']+1.15*b['r_ball']*d,affine,scale)
            if inside_hull(norm_eta(e,affine,scale),eq):
                shellpts.append({'state_id':sid,'point_id':f'O{k:02d}','direction_pool_index':di,'eta1':e[0],'eta2':e[1],'eta3':e[2],'normalized_radius':1.15*b['r_ball']});k+=1
                if k==8:break
        if k<8:raise RuntimeError(f'insufficient shell directions {sid}')
    csv_write(HERE/'outside_shell_points.csv',shellpts,list(shellpts[0]) if shellpts else ['state_id'])
    shellres=[];tasks=[]
    for x in shellpts:
        e=np.array([x['eta1'],x['eta2'],x['eta3']]);tasks.extend(taskset(x['state_id'],e,8,point_id=x['point_id']))
    rows=oracle.ensure(tasks,'outside_shell_all') if tasks else []
    for j,x in enumerate(shellpts):
        out=rows[j*8:(j+1)*8];suc,n=rows_summary(out);shellres.append({**x,'successes':suc,'trials':n,'screen_8of8':suc==8})
    csv_write(HERE/'outside_shell_results.csv',shellres,list(shellres[0]) if shellres else ['state_id'])

    vols=[];anis=[];zero_geo=[];active_geo=[]
    for sid,b in balls.items():
        vb=4/3*math.pi*b['r_ball']**3;vols.append({'state_id':sid,'r_ball':b['r_ball'],'V_ball':vb,'V_Ebridge':hull.volume,'volume_fraction':vb/hull.volume})
        rs=np.array([x['estimate'] for x in details[sid]]);ratio=float(np.max(rs)/max(np.min(rs),1e-12));cv=float(np.std(rs)/max(np.mean(rs),1e-12));cat='mild' if ratio<=1.5 else ('moderate' if ratio<=2.5 else 'strong')
        anis.append({'state_id':sid,'min_radius':np.min(rs),'max_radius':np.max(rs),'anisotropy_ratio':ratio,'coefficient_variation':cv,'category':cat})
        zero=np.zeros(3);zt=norm_eta(zero,affine,scale);dist=float(np.linalg.norm(b['ct']-zt));inside=dist<=b['r_ball']+1e-12
        rec={'state_id':sid,'normalized_center_norm':float(np.linalg.norm(b['ct'])),'normalized_center_distance_from_zero':dist,'zero_inside_ball':inside,
             'signed_zero_to_ball_distance':dist-b['r_ball'],'r_ball':b['r_ball']}
        zknown=next((x for x in known_b63[sid] if np.linalg.norm(np.asarray(x['eta']))<1e-12),None)
        (zero_geo if zknown is not None else active_geo).append(rec)
    csv_write(HERE/'ball_volume_statistics.csv',vols,list(vols[0]) if vols else ['state_id'])
    csv_write(HERE/'anisotropy_statistics.csv',anis,list(anis[0]) if anis else ['state_id'])
    csv_write(HERE/'zero_state_geometry.csv',zero_geo,list(zero_geo[0]) if zero_geo else ['state_id'])
    csv_write(HERE/'active_state_geometry.csv',active_geo,list(active_geo[0]) if active_geo else ['state_id'])

    exploit=[]
    for r in csv.DictReader(open(QGDIR/'critic_exploitation_audit.csv')):
        if r.get('critic_exploitation_candidate')!='True':continue
        sid=r['state_id']
        if sid not in STATE_IDS: exploit.append({'state_id':sid,'Qhat':r['Qhat'],'true_Q64':r['true_Q64'],'status':'NOT_APPLICABLE','inside_ball':'NOT_APPLICABLE'})
        elif sid not in balls: exploit.append({'state_id':sid,'Qhat':r['Qhat'],'true_Q64':r['true_Q64'],'status':'STATE_MATCH_BALL_UNRESOLVED','inside_ball':'NOT_APPLICABLE'})
        else:
            e=np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])]);inside=np.linalg.norm(norm_eta(e,affine,scale)-balls[sid]['ct'])<=balls[sid]['r_ball']+1e-12
            exploit.append({'state_id':sid,'Qhat':r['Qhat'],'true_Q64':r['true_Q64'],'status':'EXACT_STATE_MATCH','inside_ball':inside})
    csv_write(HERE/'q_exploitation_case_check.csv',exploit,['state_id','Qhat','true_Q64','status','inside_ball'])

    # Preliminary metrics and neutral classification; final report is audited after execution.
    radii=[b['r_ball'] for b in balls.values()];false=fi['robust_false_inclusions'];strong=sum(x['category']=='strong' for x in anis)
    if len(centers)<6 or len(balls)<6:classification='CONTINUOUS_INNER_BALL_UNDERRESOLVED'
    elif false>=2:classification='CONTINUOUS_INNER_GEOMETRY_FAILS'
    elif strong>=4 and np.median([x['volume_fraction'] for x in vols])<1e-3:classification='CONTINUOUS_SPHERE_TOO_RESTRICTIVE'
    elif false==0 and sum(x['screen_8of8'] for x in shellres)>=.75*len(shellres):classification='CONTINUOUS_INNER_BALL_PROMISING_BUT_CONSERVATIVE'
    elif false==0:classification='CONTINUOUS_INNER_BALL_STRONGLY_SUPPORTED'
    else:classification='CONTINUOUS_INNER_BALL_PROMISING_BUT_CONSERVATIVE'
    decision={'classification':classification,'centers_resolved':len(centers),'balls_resolved':len(balls),'false_inclusion':fi,
      'anisotropy_counts':dict(Counter(x['category'] for x in anis)),'learning_experiment_started':False,
      'next_step_pending_manual_scientific_review':True}
    dump(HERE/'final_decision.json',decision)
    runtime={'reused_exact_tuples_used':len(oracle.used_source_keys)+len(oracle.used_prior_pilot_keys),'new_continuations':oracle.new,
      'physical_steps':oracle.steps,'wall_time_seconds':time.monotonic()-started,'max_gpu_shards':1,'gpu_memory_mib_observed':None,
      'cpu_threads':8,'ram_allocation_gib':24,'host':platform.node(),'outcomes':dict(Counter(x['outcome'] for x in oracle.rows)),
      'budget_compliant':oracle.new<=CAP_CONT and oracle.steps<=CAP_STEPS}
    dump(HERE/'runtime_statistics.json',runtime)
    print(json.dumps({'decision':decision,'runtime':runtime,'radii':radii,'volumes':vols,'anisotropy':anis,'shell_8of8':sum(x['screen_8of8'] for x in shellres)},indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
    prepare()
    if not a.prepare_only:run()


if __name__=='__main__':main()
