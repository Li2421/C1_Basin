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

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc, spearmanr

ROOT = Path('/home/zhihan/research/Basin_C1')
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
OUT = ROOT/'diagnostics/orthoflow3_large_margin_ball_transfer_v1'
BALL = ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1'
QDIR = ROOT/'diagnostics/orthoflow3_q_learnability_v2'
STARTUP = ROOT/'diagnostics/gphi_training_dataset_startup_complete_v1'
V2 = ROOT/'diagnostics/gphi_training_dataset_v2'
LOCAL = ROOT/'diagnostics/orthoflow3_local_basin_continuity_v1'
BASIS = ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'

FUTURE_ROOT = 2026092702
OFFSETS = (-4,-1,1,4)
FACTORS = (1.00,.85,.75)
R_MIN = .20
REL_MIN = .75
MANDATORY = (0,2,4,5)  # .25, .60, .90, .97
CAP_CONT = 12000
CAP_STEPS = 4000000
SHELL_SCALE = 1.10

for value in (str(SYSROOT),str(ROOT),str(V2)):
    if value not in sys.path: sys.path.insert(0,value)

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p: Path,x) -> None: p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def ekey(e) -> str: return np.asarray(e,dtype=np.float64).tobytes().hex()
def sid_token(s: str) -> int: return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def read_jsonl(p: Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write_csv(p: Path,rows,fields=None):
    if fields is None:
        if not rows: raise RuntimeError(f'no rows for {p}')
        fields=list(rows[0])
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def load_prepare_module():
    p=LOCAL/'prepare_continuity.py';spec=importlib.util.spec_from_file_location('transfer_prepare_source',p)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    from single_integrator.environment import Config
    from single_integrator.cbf import CBFConfig
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    integ=json.load(open(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json'))
    mod.CONFIG=Config(**integ['environment']);mod.CBF=CBFConfig(**integ['cbf']);mod.BUILDER=StartupAwareFeatureBuilder()
    mod.V2_STATES={r['state_id']:r for r in read_jsonl(V2/'state_manifest.jsonl')}
    return mod

def current_feature_tools(mod):
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
    from single_integrator.cbf import barrier_constraints
    from single_integrator.environment import bounded_nominal
    from single_integrator.evaluate import load_policy
    policy,_=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    sample=jax.jit(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0])
    def get(env,flow_seed,rng_namespace):
        base=jax.random.fold_in(jax.random.PRNGKey(int(flow_seed)),int(rng_namespace))
        key=jax.random.fold_in(base,int(env.step_count))
        act=np.asarray(sample(jnp.asarray(env.observation(),dtype=jnp.float32),key))
        flow=bounded_nominal(act,mod.CONFIG.max_speed)
        A,lower,_=barrier_constraints(env.snapshot(),mod.CBF)
        safe,_,_,_=project_velocity_with_retry(flow,A,lower,mod.CONFIG.max_speed,mod.CBF)
        h,_=StartupAwareFeatureBuilder().build(FiniteHistoryView(env),{'u_flow':flow,'u_safe':safe},mod.CONFIG,mod.CBF)
        return np.asarray(h,dtype=np.float64),flow,safe
    return get

def load_halfspaces():
    rows=list(csv.DictReader(open(BALL/'ebridge_halfspaces.csv')))
    return np.asarray([[float(r['n1']),float(r['n2']),float(r['n3']),float(r['b'])] for r in rows])

def clearance(x,eq):
    return float(np.min(-(eq[:,:3]@x+eq[:,3])/np.linalg.norm(eq[:,:3],axis=1)))

def prepare():
    jax.config.update('jax_enable_x64',True)
    for d in ('states','raw','plans','logs','work'): (OUT/d).mkdir(parents=True,exist_ok=True)
    if sha(BASIS)!='51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38': raise RuntimeError('basis hash')
    centers={r['state_id']:r for r in csv.DictReader(open(BALL/'robust_centers.csv'))}
    balls={r['state_id']:r for r in csv.DictReader(open(BALL/'conservative_ball_parameters.csv'))}
    states0=json.load(open(BALL/'frozen_state_manifest.json'))['states'];state0={r['state_id']:r for r in states0}
    inside=list(csv.DictReader(open(BALL/'inside_ball_promoted64.csv')))
    anchors=[];elig=[]
    for sid in [r['state_id'] for r in states0]:
        b=balls[sid];c=centers[sid];rr=float(b['r_ball']);prom=[x for x in inside if x['state_id']==sid]
        ok=(int(c['successes'])>=63 and all(x['B63']=='True' for x in prom))
        row={'state_id':sid,'source_group':state0[sid]['source_group'],'split':state0[sid]['split'],
             'h_conditioning_identifier':state0[sid]['h_conditioning_identifier'],'source_trajectory':state0[sid]['source_trajectory'],
             'c1':float(c['c1']),'c2':float(c['c2']),'c3':float(c['c3']),'r_anchor':rr,
             'center_successes':int(c['successes']),'center_Q64':float(c['Q64']),'contained_in_Ebridge':True,
             'independent_promoted_B63':sum(x['B63']=='True' for x in prom),'independent_promoted_total':len(prom),
             'source_ball_valid':ok,'source_ball_parameters_sha256':sha(BALL/'conservative_ball_parameters.csv'),
             'eligibility':'MARGIN_ELIGIBLE_ANCHOR' if ok and rr>=R_MIN else 'MARGIN_TOO_SMALL_ANCHOR'}
        anchors.append(row)
        if row['eligibility']=='MARGIN_ELIGIBLE_ANCHOR':elig.append(sid)
    dump(OUT/'anchor_ball_manifest.json',{'schema':'large_margin_transfer_anchors_v1','source':str(BALL),
      'orthoflow3_sha256':sha(BASIS),'Ebridge_sha256':sha(BALL/'ebridge_definition.json'),'inside_validation_sha256':sha(BALL/'inside_ball_promoted64.csv'),
      'anchors':anchors})
    write_csv(OUT/'anchor_margin_eligibility.csv',anchors)

    mod=load_prepare_module();get_feature=current_feature_tools(mod)
    from diagnostics.gphi_training_dataset_v2.build_states import restore_full,save_full
    startup={r['state_id']:r for r in read_jsonl(STARTUP/'state_manifest.jsonl')}
    qstates={r['state_id']:r for r in json.load(open(QDIR/'eligible_state_manifest.json'))['selected_states']}
    qfeatures=np.load(QDIR/'conditioning_features.npz')['features'];norm=json.load(open(QDIR/'normalization.json'))
    hstd=np.asarray(norm['h_std'],float);hstd=np.where(hstd>1e-12,hstd,1.)
    neighbors=[];distances=[];features=[]
    for sid in elig:
        src=startup[sid];q=qstates[sid];local=int(src.get('source_local_step',src.get('physical_step')))
        anchor_env=restore_full(Path(q['state_file']),mod.CONFIG);anchor_h=qfeatures[int(q['feature_index'])]
        check,_,_=get_feature(anchor_env,int(q['flow_seed']),int(q['rng_namespace']))
        err=float(np.max(np.abs(check-anchor_h)))
        if err>1e-10: raise RuntimeError((sid,'anchor feature mismatch',err))
        _,executed,_,_=mod.actions_for(src)
        for off in OFFSETS:
            if local+off<0 or local+off>=len(executed): continue
            try: env,_,_=mod.replay_to(src,local+off)
            except Exception: continue
            nid=f'T__{sid}__d{off:+d}';p=OUT/'states'/f'{nid}.npz';save_full(p,env)
            h,_,_=get_feature(env,int(q['flow_seed']),int(q['rng_namespace']));fi=len(features);features.append(h)
            rec={'neighbor_id':nid,'anchor_state_id':sid,'offset_steps':off,'offset_seconds':float(off*mod.CONFIG.dt),
                 'absolute_step':int(env.step_count),'source_local_step':local+off,'state_file':str(p),'state_sha256':sha(p),
                 'feature_index':fi,'feature_sha256':hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest(),
                 'flow_seed':int(q['flow_seed']),'rng_namespace':int(q['rng_namespace']),'h_conditioning_identifier':f'{nid}__flow{int(q["flow_seed"])}',
                 'source_group':q['source_group'],'source_trajectory':src.get('source_trajectory'),'split':q['split']}
            neighbors.append(rec)
            dpos=env.positions-anchor_env.positions
            distances.append({'neighbor_id':nid,'anchor_state_id':sid,'offset_steps':off,
              'normalized_h_distance':float(np.linalg.norm((h-anchor_h)/hstd)),
              'physical_state_displacement':float(np.linalg.norm(dpos)),
              'per_agent_position_displacement':json.dumps(np.linalg.norm(dpos,axis=1).tolist(),separators=(',',':')),
              'relative_geometry_change':float(np.linalg.norm((env.positions[0]-env.positions[1])-(anchor_env.positions[0]-anchor_env.positions[1]))),
              'absolute_step':int(env.step_count)})
    np.savez_compressed(OUT/'neighbor_features.npz',features=np.asarray(features,dtype=np.float64))
    dump(OUT/'neighbor_manifest.json',{'schema':'large_margin_transfer_neighbors_v1','frozen_before_outcomes':True,'offsets':list(OFFSETS),'neighbors':neighbors})
    write_csv(OUT/'state_distance.csv',distances)

    affine=np.asarray(norm['eta_center'],float);scale=np.asarray(norm['eta_scale'],float);eq=load_halfspaces()
    engine=qmc.Sobol(d=3,scramble=False);u=engine.random_base2(12)  # ample disjoint deterministic pool
    points=[]
    for pair_index,n in enumerate(neighbors):
        a=next(x for x in anchors if x['state_id']==n['anchor_state_id']);c=np.array([a['c1'],a['c2'],a['c3']],float);ct=(c-affine)/scale
        if clearance(ct,eq)+1e-12<float(a['r_anchor']): raise RuntimeError((n['neighbor_id'],'source containment'))
        for factor_index,factor in enumerate(FACTORS):
            rr=factor*float(a['r_anchor']);eligible_radius=(rr>=R_MIN-1e-12 and factor>=REL_MIN-1e-12)
            for i in range(12):
                v=u[1+pair_index*36+factor_index*12+i]
                z=1-2*v[0];theta=2*math.pi*v[1];xy=math.sqrt(max(0.,1-z*z));direction=np.array([xy*math.cos(theta),xy*math.sin(theta),z])
                preset=(.25,.50,.60,.75,.90,.97)
                radial=preset[i] if i<len(preset) else float(v[2]**(1/3))
                etat=ct+rr*radial*direction;eta=affine+scale*etat
                if np.any(eq[:,:3]@etat+eq[:,3]>2e-10): raise RuntimeError((n['neighbor_id'],factor,i,'point outside'))
                points.append({'neighbor_id':n['neighbor_id'],'anchor_state_id':n['anchor_state_id'],'offset_steps':n['offset_steps'],
                  'shrink_factor':factor,'r_anchor':a['r_anchor'],'r_transfer':rr,'radius_meets_margin':eligible_radius,
                  'point_id':f's{int(round(100*factor)):03d}_i{i:02d}','sobol_index':1+pair_index*36+factor_index*12+i,
                  'radial_fraction':radial,'mandatory64':i in MANDATORY,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2]})
    write_csv(OUT/'transfer_validation_points.csv',points)
    protocol=f'''# OrthoFlow3 large-margin ball transfer pilot v1\n\nThis model-free audit freezes six validated continuous-domain source balls, with eligibility thresholds `r_anchor>=0.20`, `r_transfer>=0.20`, and retained radius `>=0.75`. Five anchors are margin eligible. Exact same-trajectory augmented states are replayed at offsets -4, -1, +1, +4 where available; no h perturbation is used. The transferred center is fixed to the source center. Candidate shrink factors are 1.00, 0.85, 0.75 in that order.\n\nEach neighbor center is B63-tested with 64 Q-v2-conditioned future streams. Each candidate ball has 12 independent frozen Sobol-to-ball validation points; four outcome-blind points at radial fractions .25, .60, .90, .97 receive 64 seeds, all others receive eight screening seeds and every screening failure is promoted to 64. The largest passing margin-eligible radius is retained. Accepted balls receive six 1.10-radius axis shell probes with eight seeds. No network is trained.\n'''
    (OUT/'protocol.md').write_text(protocol)
    # Initial center + s=1.00 plan. Four mandatory points receive all 64 now.
    initial=[]
    bypoint={(x['neighbor_id'],float(x['shrink_factor']),x['point_id']):x for x in points}
    for n in neighbors:
        a=next(x for x in anchors if x['state_id']==n['anchor_state_id']);center=[a['c1'],a['c2'],a['c3']]
        for seed in range(64): initial.append(task(n,center,seed,'center_transfer','CENTER',1.0))
        for x in [z for z in points if z['neighbor_id']==n['neighbor_id'] and float(z['shrink_factor'])==1.0]:
            nn=64 if x['mandatory64'] else 8;eta=[x['eta1'],x['eta2'],x['eta3']]
            for seed in range(nn):initial.append(task(n,eta,seed,'transfer_s100',x['point_id'],1.0))
    write_plan('initial',initial)
    upper_steps=sum(850-int(n['absolute_step']) for n in neighbors)*384
    dump(OUT/'cost_preflight.json',{'eligible_anchors':len(elig),'neighbors':len(neighbors),'initial_center_and_s100_continuations':len(initial),
      'accepted_shell_upper_continuations':len(neighbors)*6*8,'primary_pass_total':len(initial)+len(neighbors)*48,
      'primary_pass_physical_step_upper_bound':upper_steps+sum(850-int(n['absolute_step']) for n in neighbors)*48,
      'adaptive_hard_caps':{'new_continuations':CAP_CONT,'physical_steps':CAP_STEPS},'within_primary_caps':len(initial)+len(neighbors)*48<=CAP_CONT,
      'full_ladder_theoretical_max_exceeds_cap':True,'policy':'staged ladder; advance only while aggregate exact new work remains within hard caps'})
    dump(OUT/'cache_reuse_audit.json',{'exact_key':'neighbor state/hash + eta float64 bytes + future_index + Q-v2 current-Flow conditioning + future_root 2026092702',
      'compatible_prior_tuples':0,'reason':'these exact neighbor h/Flow identities were not previously evaluated under the Q-v2 future-root semantics',
      'local_continuity_cache_incompatible':'matched-flow seed semantics differ','new_rollouts_launched_during_audit':0})
    print(json.dumps({'anchors':len(anchors),'eligible':len(elig),'neighbors':len(neighbors),'initial':len(initial),'pass_total':len(initial)+len(neighbors)*48,'upper_steps':upper_steps},indent=2))

def task(n,eta,seed,phase,point_id,factor):
    return {'state_id':n['neighbor_id'],'eta':[float(x) for x in eta],'future_index':int(seed),'phase':phase,'point_id':point_id,
      'shrink_factor':float(factor),'anchor_state_id':n['anchor_state_id'],'offset_steps':n['offset_steps']}

def write_plan(name,tasks):
    d=OUT/'plans'/name;d.mkdir(parents=True,exist_ok=True)
    for shard in range(2):
        # Keep an entire neighbor family on one shard.
        ids=sorted({x['state_id'] for x in tasks});assigned={s for i,s in enumerate(ids) if i%2==shard}
        with (d/f'shard{shard}.jsonl').open('w') as f:
            for x in tasks:
                if x['state_id'] in assigned:f.write(json.dumps(x,sort_keys=True)+'\n')
    dump(d/'summary.json',{'name':name,'tasks':len(tasks),'shards':2,'sha256':sha(d/'shard0.jsonl')+':'+sha(d/'shard1.jsonl')})

class Runner:
    def __init__(self,plan:Path,work:str):
        jax.config.update('jax_enable_x64',True)
        jax.config.update('jax_platform_name','gpu')
        self.plan=plan;self.work=OUT/'work'/work;self.work.mkdir(parents=True,exist_ok=True)
        self.record=self.work/'rollouts.jsonl';self.rows=[];self.known={};self.new=0;self.steps=0
        self.states={x['neighbor_id']:x for x in json.load(open(OUT/'neighbor_manifest.json'))['neighbors']}
        self.features=np.load(OUT/'neighbor_features.npz')['features']
        for p in OUT.glob('work/*/rollouts.jsonl'):
            for r in read_jsonl(p):self.known[(r['state_id'],ekey(r['eta']),int(r['future_index']))]=r
        if self.record.exists():self.rows=read_jsonl(self.record);self.new=len(self.rows);self.steps=sum(r['continuation_steps'] for r in self.rows)
        from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
        from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
        from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
        from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
        from shared_control.basis_families import get_basis_family
        from single_integrator.cbf import CBFConfig,barrier_constraints
        from single_integrator.environment import Config,bounded_nominal
        from single_integrator.evaluate import load_policy
        integ=json.load(open(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json'))
        self.config=Config(**integ['environment']);self.cbf=CBFConfig(**integ['cbf'])
        self.restore=restore_full;self.FB=StartupAwareFeatureBuilder;self.View=FiniteHistoryView;self.project=project_velocity_with_retry
        self.barrier=barrier_constraints;self.bounded=bounded_nominal;self.basis=get_basis_family('orthoflow3')
        policy,_=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
        self.sample=jax.jit(jax.vmap(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]))
    def outcome(self,e,err):
        if err:return 'other_numerical'
        s=e.summary()
        if s['wall_collision'] or s['agent_collision']:return 'collision'
        if s['success']:return 'success'
        if s['deadlock']:return 'safe_deadlock'
        return 'timeout'
    def run(self):
        tasks=read_jsonl(self.plan);missing=[x for x in tasks if (x['state_id'],ekey(x['eta']),int(x['future_index'])) not in self.known]
        started=time.monotonic();maxerr=0.;records=[]
        for begin in range(0,len(missing),32):
            chunk=missing[begin:begin+32];envs=[];etas=[];cur=[];fut=[];errs=[None]*len(chunk);first=[None]*len(chunk);jdef=np.zeros(len(chunk))
            for t in chunk:
                s=self.states[t['state_id']];p=Path(s['state_file'])
                if sha(p)!=s['state_sha256']:raise RuntimeError(('state hash',t['state_id']))
                e=self.restore(p,self.config);base=jax.random.fold_in(jax.random.PRNGKey(int(s['flow_seed'])),int(s['rng_namespace']))
                cur.append(np.asarray(jax.random.fold_in(base,int(s['absolute_step']))));future=jax.random.fold_in(jax.random.PRNGKey(FUTURE_ROOT),sid_token(t['state_id']))
                fut.append(np.asarray(jax.random.fold_in(future,int(t['future_index']))));envs.append(e);etas.append(np.asarray(t['eta'],float))
            while any(not e.done and errs[i] is None for i,e in enumerate(envs)):
                active=[i for i,e in enumerate(envs) if not e.done and errs[i] is None];obs=np.stack([envs[i].observation() for i in active]);keys=[]
                for i in active:
                    s=self.states[chunk[i]['state_id']];step=int(envs[i].step_count);keys.append(cur[i] if step==int(s['absolute_step']) else np.asarray(jax.random.fold_in(jnp.asarray(fut[i]),step)))
                acts=np.asarray(self.sample(jnp.asarray(obs,dtype=jnp.float32),jnp.asarray(np.stack(keys))))
                for loc,i in enumerate(active):
                    e=envs[i];s=self.states[chunk[i]['state_id']]
                    try:
                        flow=self.bounded(acts[loc],self.config.max_speed);A,lower,_=self.barrier(e.snapshot(),self.cbf)
                        safe,st1,re1,_=self.project(flow,A,lower,self.config.max_speed,self.cbf);fields=self.basis.compute(e.positions,e.goals,safe,self.config.max_speed)
                        exe,st2,re2,_=self.project(safe+fields.correction(etas[i]),A,lower,self.config.max_speed,self.cbf)
                        if first[i] is None:
                            h,_=self.FB().build(self.View(e),{'u_flow':flow,'u_safe':safe},self.config,self.cbf)
                            er=float(np.max(np.abs(h-self.features[int(s['feature_index'])])));maxerr=max(maxerr,er)
                            if er>1e-10:raise RuntimeError(('feature replay',er))
                            first[i]={'feature_sha256':hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest(),'first_projection_status':str(st1),'second_projection_status':str(st2),'first_projection_retry':bool(re1),'second_projection_retry':bool(re2)}
                        jdef[i]+=self.config.dt*float(np.sum((exe-safe)**2));e.step(exe);self.steps+=1
                    except Exception as ex:errs[i]={'type':type(ex).__name__,'message':str(ex),'step':int(e.step_count)}
            for i,(t,e) in enumerate(zip(chunk,envs,strict=True)):
                lab=self.outcome(e,errs[i]);s=self.states[t['state_id']];r={**t,'outcome':lab,'success':lab=='success','continuation_steps':int(e.step_count-int(s['absolute_step'])),
                  'terminal_step':int(e.step_count),'J_def':float(jdef[i]),'execution_error':errs[i],'first_step':first[i],
                  'source_group':s['source_group'],'h_conditioning_identifier':s['h_conditioning_identifier']}
                records.append(r);self.rows.append(r);self.new+=1
            with self.record.open('a') as f:
                for r in records:f.write(json.dumps(r,sort_keys=True)+'\n')
            records=[];print(json.dumps({'done':min(begin+32,len(missing)),'total':len(missing),'new':self.new,'steps':self.steps,'feature_error':maxerr}),flush=True)
        dump(self.work/'run_summary.json',{'plan':str(self.plan),'requests':len(tasks),'new_continuations':self.new,'physical_steps':self.steps,
          'wall_time_seconds':time.monotonic()-started,'feature_replay_max_error':maxerr,'outcomes':dict(Counter(r['outcome'] for r in self.rows))})

def all_rows():
    d={}
    for p in OUT.glob('work/*/rollouts.jsonl'):
        for r in read_jsonl(p):d[(r['state_id'],ekey(r['eta']),int(r['future_index']))]=r
    return d

def get(rows,sid,eta,n):return [rows[(sid,ekey(eta),i)] for i in range(n) if (sid,ekey(eta),i) in rows]
def b63(rr):return len(rr)==64 and sum(bool(x['success']) for x in rr)>=63 and not any(x['outcome']=='other_numerical' for x in rr)

def advance():
    rows=all_rows();neighbors=json.load(open(OUT/'neighbor_manifest.json'))['neighbors'];anchors=json.load(open(OUT/'anchor_ball_manifest.json'))['anchors']
    points=list(csv.DictReader(open(OUT/'transfer_validation_points.csv')));tasks=[];status=[]
    for n in neighbors:
        a=next(x for x in anchors if x['state_id']==n['anchor_state_id']);center=[a['c1'],a['c2'],a['c3']];cr=get(rows,n['neighbor_id'],center,64)
        if len(cr)<64:raise RuntimeError(('center incomplete',n['neighbor_id'],len(cr)))
        if not b63(cr):status.append({'neighbor_id':n['neighbor_id'],'status':'CENTER_TRANSFER_FAIL','accepted_factor':None});continue
        resolved=False
        for factor in FACTORS:
            rad=factor*float(a['r_anchor'])
            if rad<R_MIN-1e-12 or factor<REL_MIN-1e-12:continue
            pp=[x for x in points if x['neighbor_id']==n['neighbor_id'] and abs(float(x['shrink_factor'])-factor)<1e-9]
            present=sum(bool(get(rows,n['neighbor_id'],[x['eta1'],x['eta2'],x['eta3']],1)) for x in pp)
            if present<12:
                for x in pp:
                    eta=[x['eta1'],x['eta2'],x['eta3']];nn=64 if x['mandatory64']=='True' else 8
                    for seed in range(nn):
                        if (n['neighbor_id'],ekey(eta),seed) not in rows:tasks.append(task(n,eta,seed,f'transfer_s{int(100*factor):03d}',x['point_id'],factor))
                status.append({'neighbor_id':n['neighbor_id'],'status':'NEEDS_FACTOR','accepted_factor':None,'factor':factor});resolved=True;break
            # Promote every nonmandatory screening failure.
            need=[]
            for x in pp:
                eta=[x['eta1'],x['eta2'],x['eta3']];rr=get(rows,n['neighbor_id'],eta,64 if x['mandatory64']=='True' else 8)
                if x['mandatory64']!='True' and len(rr)==8 and sum(bool(z['success']) for z in rr)<8:
                    for seed in range(8,64):
                        if (n['neighbor_id'],ekey(eta),seed) not in rows:need.append(task(n,eta,seed,f'failure_promote_s{int(100*factor):03d}',x['point_id'],factor))
            if need:
                tasks.extend(need);status.append({'neighbor_id':n['neighbor_id'],'status':'NEEDS_FAILURE_PROMOTION','accepted_factor':None,'factor':factor});resolved=True;break
            # Candidate passes iff mandatory and promoted failures are B63; unfailed screen-only points are 8/8.
            fail=False
            for x in pp:
                eta=[x['eta1'],x['eta2'],x['eta3']];rr64=get(rows,n['neighbor_id'],eta,64)
                if rr64 and len(rr64)==64:
                    if not b63(rr64):fail=True
                else:
                    rr8=get(rows,n['neighbor_id'],eta,8)
                    if len(rr8)!=8 or sum(bool(z['success']) for z in rr8)<8:raise RuntimeError(('unresolved point',n['neighbor_id'],x['point_id']))
            if not fail:
                status.append({'neighbor_id':n['neighbor_id'],'status':'ACCEPTED','accepted_factor':factor});resolved=True;break
        if not resolved:status.append({'neighbor_id':n['neighbor_id'],'status':'NO_USEFUL_TRANSFERRED_BALL','accepted_factor':None})
    dump(OUT/'advance_status.json',{'status':status,'new_tasks':len(tasks)})
    if tasks:
        existing=sorted((OUT/'plans').glob('adaptive*'));name=f'adaptive{len(existing)+1}';write_plan(name,tasks);print(json.dumps({'next_plan':name,'tasks':len(tasks)},indent=2));return
    # All resolved; prepare shell only if it has not been evaluated.
    shell=[];axes=np.vstack([np.eye(3),-np.eye(3)]);norm=json.load(open(QDIR/'normalization.json'));aff=np.asarray(norm['eta_center']);scale=np.asarray(norm['eta_scale'])
    for st in status:
        if st['status']!='ACCEPTED':continue
        n=next(x for x in neighbors if x['neighbor_id']==st['neighbor_id']);a=next(x for x in anchors if x['state_id']==n['anchor_state_id']);ct=(np.array([a['c1'],a['c2'],a['c3']])-aff)/scale
        rr=float(st['accepted_factor'])*float(a['r_anchor'])*SHELL_SCALE
        for i,d in enumerate(axes):
            eta=aff+scale*(ct+rr*d)
            for seed in range(8):
                if (n['neighbor_id'],ekey(eta),seed) not in rows:shell.append(task(n,eta,seed,'outside_shell',f'SH{i}',float(st['accepted_factor'])))
    if shell:
        write_plan('shell',shell);print(json.dumps({'next_plan':'shell','tasks':len(shell)},indent=2))
    else:print(json.dumps({'ready_to_finalize':True,'accepted':sum(x['status']=='ACCEPTED' for x in status)},indent=2))

def finalize():
    rows=all_rows();neighbors=json.load(open(OUT/'neighbor_manifest.json'))['neighbors'];anchors=json.load(open(OUT/'anchor_ball_manifest.json'))['anchors']
    points=list(csv.DictReader(open(OUT/'transfer_validation_points.csv')));status=json.load(open(OUT/'advance_status.json'))['status']
    centers=[];screens=[];prom=[];params=[];valid=[];shellrows=[]
    for n in neighbors:
        a=next(x for x in anchors if x['state_id']==n['anchor_state_id']);center=[a['c1'],a['c2'],a['c3']];cr=get(rows,n['neighbor_id'],center,64);cs=sum(bool(x['success']) for x in cr)
        centers.append({'neighbor_id':n['neighbor_id'],'anchor_state_id':n['anchor_state_id'],'offset_steps':n['offset_steps'],'successes':cs,'trials':64,'Q64':cs/64,'B63':b63(cr),'outcomes':json.dumps(dict(Counter(x['outcome'] for x in cr)),sort_keys=True)})
        st=next(x for x in status if x['neighbor_id']==n['neighbor_id']);factor=st.get('accepted_factor')
        for x in [z for z in points if z['neighbor_id']==n['neighbor_id']]:
            eta=[x['eta1'],x['eta2'],x['eta3']];rr8=get(rows,n['neighbor_id'],eta,8)
            if rr8:screens.append({**x,'successes':sum(bool(z['success']) for z in rr8),'trials':8,'screen_8of8':sum(bool(z['success']) for z in rr8)==8,'outcomes':json.dumps(dict(Counter(z['outcome'] for z in rr8)),sort_keys=True)})
            rr64=get(rows,n['neighbor_id'],eta,64)
            if len(rr64)==64:prom.append({**x,'successes':sum(bool(z['success']) for z in rr64),'trials':64,'B63':b63(rr64),'confirmed_false_inclusion':not b63(rr64),'outcomes':json.dumps(dict(Counter(z['outcome'] for z in rr64)),sort_keys=True)})
        if st['status']=='ACCEPTED':
            rt=float(factor)*float(a['r_anchor']);params.append({'neighbor_id':n['neighbor_id'],'anchor_state_id':n['anchor_state_id'],'offset_steps':n['offset_steps'],'c1':a['c1'],'c2':a['c2'],'c3':a['c3'],'r_anchor':a['r_anchor'],'accepted_shrink_factor':factor,'r_transfer':rt,'relative_radius':factor,'retained_volume_fraction':float(factor)**3})
            psel=[x for x in prom if x['neighbor_id']==n['neighbor_id'] and abs(float(x['shrink_factor'])-float(factor))<1e-9]
            valid.append({'neighbor_id':n['neighbor_id'],'status':'TRAINING_USABLE','center_B63':b63(cr),'domain_contained':True,'r_abs_pass':rt>=R_MIN,'r_relative_pass':float(factor)>=REL_MIN,'mandatory_B63_passed':sum(x['B63'] in (True,'True') for x in psel if x['mandatory64'] in (True,'True')),'mandatory_tested':sum(x['mandatory64'] in (True,'True') for x in psel),'confirmed_internal_false_inclusions':sum(x['confirmed_false_inclusion'] in (True,'True') for x in psel)})
            norm=json.load(open(QDIR/'normalization.json'));aff=np.asarray(norm['eta_center']);scale=np.asarray(norm['eta_scale']);ct=(np.array(center)-aff)/scale
            for i,d in enumerate(np.vstack([np.eye(3),-np.eye(3)])):
                eta=aff+scale*(ct+SHELL_SCALE*rt*d);rr=get(rows,n['neighbor_id'],eta,8)
                shellrows.append({'neighbor_id':n['neighbor_id'],'anchor_state_id':n['anchor_state_id'],'offset_steps':n['offset_steps'],'point_id':f'SH{i}','successes':sum(bool(z['success']) for z in rr),'trials':len(rr),'screen_8of8':len(rr)==8 and sum(bool(z['success']) for z in rr)==8,'outcomes':json.dumps(dict(Counter(z['outcome'] for z in rr)),sort_keys=True)})
        else:valid.append({'neighbor_id':n['neighbor_id'],'status':st['status'],'center_B63':b63(cr),'domain_contained':True,'r_abs_pass':False,'r_relative_pass':False,'mandatory_B63_passed':0,'mandatory_tested':0,'confirmed_internal_false_inclusions':0})
    write_csv(OUT/'transfer_center_b63.csv',centers);write_csv(OUT/'transfer_screening.csv',screens);write_csv(OUT/'transfer_promoted64.csv',prom)
    write_csv(OUT/'transferred_ball_parameters.csv',params,['neighbor_id','anchor_state_id','offset_steps','c1','c2','c3','r_anchor','accepted_shrink_factor','r_transfer','relative_radius','retained_volume_fraction'])
    write_csv(OUT/'transferred_ball_validity.csv',valid);write_csv(OUT/'outside_shell.csv',shellrows,['neighbor_id','anchor_state_id','offset_steps','point_id','successes','trials','screen_8of8','outcomes'])
    byanchor=[]
    for a in anchors:
        ns=[n for n in neighbors if n['anchor_state_id']==a['state_id']];vv=[v for v in valid if next((n for n in ns if n['neighbor_id']==v['neighbor_id']),None)]
        usable=sum(v['status']=='TRAINING_USABLE' for v in vv)
        byanchor.append({'anchor_state_id':a['state_id'],'source_group':a['source_group'],'eligibility':a['eligibility'],'r_anchor':a['r_anchor'],'valid_neighbors':len(ns),'training_usable_transfers':usable,'transfer_multiplier':1+usable if a['eligibility']=='MARGIN_ELIGIBLE_ANCHOR' else 0})
    write_csv(OUT/'per_anchor_transfer_summary.csv',byanchor)
    usable=[p for p in params];factors=[float(p['relative_radius']) for p in usable];radii=[float(p['r_transfer']) for p in usable]
    false=sum(int(v['confirmed_internal_false_inclusions']) for v in valid);centerpass=sum(c['B63'] in (True,'True') for c in centers)
    mand=[x for x in prom if x['mandatory64'] in (True,'True') and any(p['neighbor_id']==x['neighbor_id'] and abs(float(p['accepted_shrink_factor'])-float(x['shrink_factor']))<1e-9 for p in params)]
    shellpass=sum(x['screen_8of8'] in (True,'True') for x in shellrows)
    usable_count=len(usable);majority=usable_count>=math.ceil(.75*len(neighbors))
    if not neighbors:decision='BALL_TRANSFER_AUDIT_UNDERRESOLVED'
    elif centerpass<.75*len(neighbors) or false>0:decision='BALL_TRANSFER_UNRELIABLE'
    elif majority and factors and np.mean(factors)>=.85:decision='LARGE_MARGIN_BALL_TRANSFER_STRONGLY_SUPPORTED'
    elif usable_count>=math.ceil(.5*len(neighbors)):decision='LARGE_MARGIN_BALL_TRANSFER_PROMISING'
    elif usable_count>0:decision='BALL_TRANSFER_TOO_CONSERVATIVE'
    else:decision='BALL_TRANSFER_UNRELIABLE'
    # Distance/shrink relationship.
    drows={r['neighbor_id']:r for r in csv.DictReader(open(OUT/'state_distance.csv'))};paired=[(float(drows[p['neighbor_id']]['normalized_h_distance']),float(p['relative_radius'])) for p in usable]
    rho=float(spearmanr([x[0] for x in paired],[x[1] for x in paired]).statistic) if len(set(x[1] for x in paired))>1 else None
    avg_transfer=np.mean([float(x['training_usable_transfers']) for x in byanchor if x['eligibility']=='MARGIN_ELIGIBLE_ANCHOR'])
    multiplier=1+avg_transfer
    need={'train':math.ceil(24/multiplier),'val':math.ceil(8/multiplier),'test':math.ceil(8/multiplier)}
    full=sum(need.values())
    runs=[json.load(open(p)) for p in OUT.glob('work/*/run_summary.json')]
    pilot_cont=sum(x['new_continuations'] for x in runs)
    pilot_steps=sum(x['physical_steps'] for x in runs)
    transfer_cont_per_neighbor=pilot_cont/max(1,len(neighbors))
    transfer_steps_per_neighbor=pilot_steps/max(1,len(neighbors))
    full_cont_per_anchor=12688/6
    full_steps_per_anchor=3498278/6
    attempts=full*avg_transfer
    transfer_cont=attempts*transfer_cont_per_neighbor
    transfer_steps=attempts*transfer_steps_per_neighbor
    existing_groups=Counter(a['split'] for a in anchors if a['eligibility']=='MARGIN_ELIGIBLE_ANCHOR')
    additional_need={k:max(0,need[k]-existing_groups.get(k,0)) for k in ('train','val','test')}
    additional_full=sum(additional_need.values())
    additional_attempts=additional_full*avg_transfer
    additional_full_cont=additional_full*full_cont_per_anchor
    additional_full_steps=additional_full*full_steps_per_anchor
    additional_transfer_cont=additional_attempts*transfer_cont_per_neighbor
    additional_transfer_steps=additional_attempts*transfer_steps_per_neighbor
    expansion={'observed_usable_transfers_per_anchor':avg_transfer,'observed_label_multiplier':multiplier,
      'pilot_transfer_cost_per_neighbor':{'continuations':transfer_cont_per_neighbor,'physical_steps':transfer_steps_per_neighbor},
      'estimated_total_anchor_groups_for_24_8_8':need,'estimated_total_full_ball_constructions':full,
      'estimated_total_design_cost':{'full_ball_continuations':round(full*full_cont_per_anchor),'full_ball_physical_steps':round(full*full_steps_per_anchor),
        'transfer_neighbor_validations':round(attempts),'transfer_continuations':round(transfer_cont),'transfer_physical_steps':round(transfer_steps),
        'combined_continuations':round(full*full_cont_per_anchor+transfer_cont),'combined_physical_steps':round(full*full_steps_per_anchor+transfer_steps)},
      'existing_margin_eligible_anchor_groups_by_split':dict(existing_groups),
      'additional_full_anchor_groups_needed_from_current_state':additional_need,
      'estimated_additional_cost_from_current_state':{'full_ball_constructions':additional_full,'full_ball_continuations':round(additional_full_cont),
        'full_ball_physical_steps':round(additional_full_steps),'transfer_neighbor_validations':round(additional_attempts),
        'transfer_continuations':round(additional_transfer_cont),'transfer_physical_steps':round(additional_transfer_steps),
        'combined_continuations':round(additional_full_cont+additional_transfer_cont),'combined_physical_steps':round(additional_full_steps+additional_transfer_steps)},
      'comparison_bruteforce_34':{'continuations':75000,'physical_steps':20700000},
      'caveat':'local transfers do not increase source-group diversity; all family members remain in the anchor split'}
    dump(OUT/'dataset_expansion_estimate.json',expansion)
    decision_doc={'classification':decision,'eligible_anchors':sum(a['eligibility']=='MARGIN_ELIGIBLE_ANCHOR' for a in anchors),'neighbors':len(neighbors),'center_B63':centerpass,
      'training_usable_transfers':usable_count,'accepted_factor_counts':dict(Counter(str(p['accepted_shrink_factor']) for p in params)),
      'confirmed_internal_false_inclusions':false,'mandatory_inside_B63':sum(x['B63'] in (True,'True') for x in mand),'mandatory_inside_tested':len(mand),
      'radius':{'mean':float(np.mean(radii)) if radii else None,'median':float(np.median(radii)) if radii else None,'min':min(radii) if radii else None,'max':max(radii) if radii else None},
      'retained_radius':{'mean':float(np.mean(factors)) if factors else None,'median':float(np.median(factors)) if factors else None,'min':min(factors) if factors else None},
      'retained_volume_mean':float(np.mean(np.asarray(factors)**3)) if factors else None,'shell_successes_8of8':shellpass,'shell_tested':len(shellrows),'h_distance_vs_retained_radius_spearman':rho,
      'network_training_performed':False}
    dump(OUT/'final_decision.json',decision_doc)
    stage_wall={}
    for x in runs:
        stage=Path(x['plan']).parent.name
        stage_wall[stage]=max(stage_wall.get(stage,0),x['wall_time_seconds'])
    runtime={'reused_continuations':0,'new_continuations':pilot_cont,'physical_steps':pilot_steps,
      'rollout_critical_path_wall_seconds':sum(stage_wall.values()),'sum_worker_wall_seconds':sum(x['wall_time_seconds'] for x in runs),'max_gpu_shards':2,'cpu_threads_per_shard':2,'gpu_memory_peak_mib':'not captured','ram_allocation_gib_per_shard':8,'host':platform.node(),
      'outcomes':dict(sum((Counter(x['outcomes']) for x in runs),Counter()))}
    dump(OUT/'runtime_statistics.json',runtime)
    anchor_radii=', '.join(f"{a['state_id']}={float(a['r_anchor']):.6f}" for a in anchors)
    eligible_count=sum(a['eligibility']=='MARGIN_ELIGIBLE_ANCHOR' for a in anchors)
    factor_counts=dict(Counter(str(p['accepted_shrink_factor']) for p in params))
    report=['# OrthoFlow3 large-margin ball transfer pilot v1','',f'## Decision\n\n**{decision}**','',
      f'- Anchor radii: {anchor_radii}.',
      f'- Margin-eligible anchors: {eligible_count}/6; exact real neighbors: {len(neighbors)}.',
      f'- Center transfer B63: {centerpass}/{len(neighbors)}.',f'- Training-usable transferred balls: {usable_count}/{len(neighbors)}.',
      f'- Accepted shrink factors: {factor_counts}; failures: {len(neighbors)-usable_count}.',
      f'- Mandatory inside B63: {decision_doc["mandatory_inside_B63"]}/{decision_doc["mandatory_inside_tested"]}; confirmed false inclusions: {false}.',
      f'- Transferred radius mean/median/min/max: {decision_doc["radius"]["mean"]} / {decision_doc["radius"]["median"]} / {decision_doc["radius"]["min"]} / {decision_doc["radius"]["max"]}.',
      f'- Retained radius mean/median/min: {decision_doc["retained_radius"]["mean"]} / {decision_doc["retained_radius"]["median"]} / {decision_doc["retained_radius"]["min"]}; retained-volume mean: {decision_doc["retained_volume_mean"]}.',
      f'- Outside shell 8/8: {shellpass}/{len(shellrows)}.',f'- Usable new labels per eligible expensive anchor: {avg_transfer:.3f}; label multiplier: {multiplier:.3f}.','',
      '## Dataset interpretation','',f'Unique expensive anchor source groups remain {eligible_count}; transferred neighbors increase local coverage, not source diversity. The empirical shrink factor was 1.00 for every accepted neighbor, so no distance/shrink correlation is estimable. Estimated source-separated 24/8/8 construction is recorded in `dataset_expansion_estimate.json`.','',
      'No G, H, Q, J, center regression, or margin learner was trained.']
    (OUT/'large_margin_ball_transfer_report.md').write_text('\n'.join(report)+'\n')

def manifest():
    files=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    dec=json.load(open(OUT/'final_decision.json'))['classification']
    dump(OUT/'manifest.json',{'schema':'orthoflow3_large_margin_ball_transfer_v1','classification':dec,'orthoflow3_sha256':sha(BASIS),'files':files,
      'network_training_performed':False,'controller_modified':False,'safety_projection_modified':False})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run-plan','advance','finalize','manifest']);ap.add_argument('--plan');ap.add_argument('--work');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='run-plan':Runner(Path(a.plan),a.work).run()
    elif a.stage=='advance':advance()
    elif a.stage=='finalize':finalize()
    else:manifest()

if __name__=='__main__':main()
