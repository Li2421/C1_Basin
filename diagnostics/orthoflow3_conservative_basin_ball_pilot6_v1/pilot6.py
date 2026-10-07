#!/usr/bin/env python3
"""Frozen six-state conservative OrthoFlow3 basin-ball pilot.

This intentionally uses the exact Q-v2 h/current-Flow conditioning and
future-root semantics.  It is a sequential oracle protocol: all adaptivity is
the predeclared ray/center procedure, never an outcome-selected state or
direction choice.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

ROOT = Path('/home/zhihan/research/Basin_C1')
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
REF = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_v1'
QDIR = ROOT/'diagnostics/orthoflow3_q_learnability_v2'
DDIR = ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
QGDIR = ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1'
HERE = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_pilot6_v1'
FUTURE_ROOT = 2026092702
CAP_CONT = 12000
CAP_STEPS = 3500000
RAY_RADII = (0.05, 0.10, 0.20, 0.35, 0.50)
KAPPA = 0.85

for value in (str(SYSROOT), str(ROOT), str(ROOT/'diagnostics/gphi_training_dataset_v2')):
    if value not in sys.path:
        sys.path.insert(0, value)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def eta_key(eta: np.ndarray | list[float]) -> str:
    return np.asarray(eta, dtype=np.float64).tobytes().hex()


def sid_token(state_id: str) -> int:
    return int(hashlib.sha256(state_id.encode()).hexdigest()[:8], 16)


def norm_eta(eta: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (np.asarray(eta, dtype=np.float64)-center)/scale


def raw_eta(value: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return center+scale*np.asarray(value, dtype=np.float64)


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


class Oracle:
    def __init__(self, states: dict[str,dict], features: np.ndarray, low: np.ndarray, high: np.ndarray):
        self.states=states; self.features=features; self.low=low; self.high=high
        self.rows: list[dict]=[]; self.known: dict[tuple[str,str,int],dict]={}; self.new=0; self.steps=0
        self.used_source_keys: set[tuple[str,str,int]]=set(); self.used_prior_pilot_keys: set[tuple[str,str,int]]=set()
        self.raw=HERE/'raw'; self.raw.mkdir(parents=True, exist_ok=True)
        self.record_path=self.raw/'pilot_rollouts.jsonl'
        self.started=time.monotonic()
        self._load_prior()
        jax.config.update('jax_enable_x64', True)
        jax.config.update('jax_platform_name', 'gpu')
        from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
        from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
        from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
        from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
        from shared_control.basis_families import get_basis_family
        from single_integrator.cbf import CBFConfig, barrier_constraints
        from single_integrator.environment import Config, bounded_nominal
        from single_integrator.evaluate import load_policy
        self.restore_full=restore_full; self.FeatureBuilder=StartupAwareFeatureBuilder; self.FiniteHistoryView=FiniteHistoryView
        self.project=project_velocity_with_retry; self.barrier=barrier_constraints; self.bounded=bounded_nominal
        self.config=Config(**json.loads((ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json').read_text())['environment'])
        self.cbf=CBFConfig(**json.loads((ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json').read_text())['cbf'])
        self.policy, provenance=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
        self.sample=jax.jit(jax.vmap(lambda obs,key:self.policy.sample_actions(obs[None], seed=key)[0]))
        self.basis=get_basis_family('orthoflow3')

    def _insert(self, row: dict, source: str) -> None:
        if row.get('state_id') not in self.states or 'eta' not in row or 'future_index' not in row:
            return
        k=(row['state_id'],eta_key(row['eta']),int(row['future_index']))
        current=self.known.get(k)
        item=dict(row); item['_source']=source
        if current is None or (current['_source']!='pilot' and source=='pilot'):
            self.known[k]=item

    def _load_jsonls(self, root: Path, source: str) -> None:
        for p in root.glob('raw/*/shard*.jsonl'):
            for line in p.read_text().splitlines():
                if line.strip(): self._insert(json.loads(line), source)

    def _load_prior(self) -> None:
        # The previous preflight admitted only these Q-v2-root-compatible records.
        self._load_jsonls(QDIR, 'qv2_exact')
        self._load_jsonls(DDIR, 'direct_eta_exact')
        self._load_jsonls(QGDIR, 'q_guided_exact')
        if self.record_path.exists():
            for line in self.record_path.read_text().splitlines():
                if line.strip():
                    r=json.loads(line); self.rows.append(r); self._insert(r,'pilot')
            self.new=len(self.rows); self.steps=sum(int(r['continuation_steps']) for r in self.rows)

    @staticmethod
    def outcome(env, error):
        if error is not None: return 'other_numerical'
        summary=env.summary()
        if summary['wall_collision'] or summary['agent_collision']: return 'collision'
        if summary['success']: return 'success'
        if summary['deadlock']: return 'safe_deadlock'
        return 'timeout'

    def ensure(self, tasks: list[dict], phase: str) -> list[dict]:
        # Global persistent cache preflight. This is read-only and never changes
        # physical rollout semantics. Ambiguous/incompatible records are ignored.
        try:
            from shared_rollout_db.src.integration import reuse_toy_tasks
            for row in reuse_toy_tasks(tasks,self.states,FUTURE_ROOT):
                self._insert(row,'global_db_exact')
        except Exception as exc:
            print(json.dumps({'global_cache_preflight':'unavailable','reason':str(exc)}),flush=True)
        missing=[]
        for task in tasks:
            k=(task['state_id'],eta_key(task['eta']),int(task['future_index']))
            if k not in self.known: missing.append(task)
        if self.new+len(missing)>CAP_CONT:
            raise RuntimeError(f'continuation cap: current={self.new}, request={len(missing)}, cap={CAP_CONT}, phase={phase}')
        if missing: self._run(missing,phase)
        result=[]
        for task in tasks:
            k=(task['state_id'],eta_key(task['eta']),int(task['future_index']))
            row=self.known[k]; result.append(row)
            if row['_source']=='pilot': self.used_prior_pilot_keys.add(k)
            else: self.used_source_keys.add(k)
        return result

    def _run(self, tasks: list[dict], phase: str) -> None:
        records=[]; max_h_error=0.0
        for begin in range(0,len(tasks),32):
            chunk=tasks[begin:begin+32]; envs=[]; etas=[]; current_keys=[]; future_keys=[]; errors=[None]*len(chunk); first=[None]*len(chunk)
            for task in chunk:
                state=self.states[task['state_id']]; p=Path(state['state_file'])
                if sha(p)!=state['state_sha256']: raise RuntimeError(('state hash',task['state_id']))
                env=self.restore_full(p,self.config)
                if env.done or int(env.step_count)!=int(state['absolute_step']): raise RuntimeError(('bad restore',task['state_id']))
                base=jax.random.fold_in(jax.random.PRNGKey(int(state['flow_seed'])),int(state['rng_namespace']))
                current_keys.append(np.asarray(jax.random.fold_in(base,int(state['absolute_step']))))
                future=jax.random.fold_in(jax.random.PRNGKey(FUTURE_ROOT),sid_token(task['state_id']))
                future_keys.append(np.asarray(jax.random.fold_in(future,int(task['future_index']))))
                envs.append(env); etas.append(np.asarray(task['eta'],dtype=np.float64))
            jdef=np.zeros(len(chunk),dtype=np.float64)
            while any(not e.done and errors[i] is None for i,e in enumerate(envs)):
                active=[i for i,e in enumerate(envs) if not e.done and errors[i] is None]
                obs=np.stack([np.asarray(envs[i].observation(),dtype=np.float32) for i in active])
                keys=[]
                for i in active:
                    state=self.states[chunk[i]['state_id']]; step=int(envs[i].step_count)
                    keys.append(current_keys[i] if step==int(state['absolute_step']) else np.asarray(jax.random.fold_in(jnp.asarray(future_keys[i]),step)))
                actions=np.asarray(self.sample(jnp.asarray(obs),jnp.asarray(np.stack(keys))))
                for local,i in enumerate(active):
                    env=envs[i]; state=self.states[chunk[i]['state_id']]
                    try:
                        flow=self.bounded(actions[local],self.config.max_speed)
                        A,lower,_=self.barrier(env.snapshot(),self.cbf)
                        safe,status1,retry1,_=self.project(flow,A,lower,self.config.max_speed,self.cbf)
                        fields=self.basis.compute(env.positions,env.goals,safe,self.config.max_speed)
                        correction=fields.correction(etas[i])
                        executed,status2,retry2,_=self.project(safe+correction,A,lower,self.config.max_speed,self.cbf)
                        if first[i] is None:
                            feature,_=self.FeatureBuilder().build(self.FiniteHistoryView(env),{'u_flow':flow,'u_safe':safe},self.config,self.cbf)
                            err=float(np.max(np.abs(feature-self.features[int(state['feature_index'])]))); max_h_error=max(max_h_error,err)
                            if err>1e-10: raise RuntimeError(('feature replay',chunk[i]['state_id'],err))
                            first[i]={'feature_sha256':hashlib.sha256(np.asarray(feature,dtype=np.float64).tobytes()).hexdigest(),
                                      'first_projection_status':str(status1),'second_projection_status':str(status2),
                                      'first_projection_retry':bool(retry1),'second_projection_retry':bool(retry2)}
                        jdef[i]+=self.config.dt*float(np.sum((executed-safe)**2)); env.step(executed); self.steps+=1
                        if self.steps>CAP_STEPS: raise RuntimeError(f'physical step cap exceeded: {self.steps}>{CAP_STEPS}')
                    except Exception as exc:
                        errors[i]={'type':type(exc).__name__,'message':str(exc),'step':int(env.step_count)}
            for i,(task,env) in enumerate(zip(chunk,envs,strict=True)):
                label=self.outcome(env,errors[i]); state=self.states[task['state_id']]
                rec={**task,'phase':phase,'source_group':state['source_group'],'h_conditioning_identifier':state['h_conditioning_identifier'],
                     'outcome':label,'success':label=='success','continuation_steps':int(env.step_count-int(state['absolute_step'])),
                     'terminal_step':int(env.step_count),'J_def':float(jdef[i]),'execution_error':errors[i],'first_step':first[i]}
                self.rows.append(rec); records.append(rec); self._insert(rec,'pilot'); self.new+=1
            with self.record_path.open('a') as f:
                for rec in records:
                    f.write(json.dumps(rec,sort_keys=True)+'\n')
            try:
                from shared_rollout_db.src.integration import journal_toy
                journal_toy(records,HERE,FUTURE_ROOT)
            except Exception as exc:
                print(json.dumps({'global_cache_journal':'failed','reason':str(exc)}),flush=True)
            records=[]
            print(json.dumps({'phase':phase,'done':min(begin+32,len(tasks)),'phase_total':len(tasks),'new':self.new,'steps':self.steps,'feature_error':max_h_error}),flush=True)

    def summary(self, state_id: str, eta: np.ndarray, n: int) -> dict:
        tasks=[{'state_id':state_id,'eta':np.asarray(eta,dtype=np.float64).tolist(),'future_index':i} for i in range(n)]
        rows=self.ensure(tasks,'summary_reuse')
        return {'successes':sum(bool(r['success']) for r in rows),'trials':n,'rows':rows,
                'outcomes':dict(Counter(r['outcome'] for r in rows))}


def taskset(state_id: str, eta: np.ndarray, n: int, **extra) -> list[dict]:
    return [{'state_id':state_id,'eta':np.asarray(eta,dtype=np.float64).tolist(),'future_index':i,**extra} for i in range(n)]


def in_active(eta: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    return bool(np.all(eta>=low-1e-12) and np.all(eta<=high+1e-12))


def rho_limit(c: np.ndarray, d: np.ndarray) -> float:
    values=[]
    for x,v in zip(c,d,strict=True):
        if v>1e-14: values.append((0.5-x)/v)
        elif v<-1e-14: values.append((-0.5-x)/v)
    good=[x for x in values if x>=0]
    return min(good) if good else 0.0


def rows_summary(rows: list[dict]) -> tuple[int,int]:
    return sum(bool(x['success']) for x in rows),len(rows)


def main() -> None:
    HERE.mkdir(parents=True,exist_ok=True)
    prior=json.loads((REF/'state_manifest.json').read_text()); selection=prior['selected_states'][:6]
    qstates={x['state_id']:x for x in json.loads((QDIR/'eligible_state_manifest.json').read_text())['selected_states']}
    states={x['state_id']:qstates[x['state_id']] for x in selection}
    if list(states)!=[x['state_id'] for x in selection]: raise RuntimeError('pilot prefix mismatch')
    norm=json.loads((REF/'eta_normalization.json').read_text()); affine=np.asarray(norm['affine_center'],dtype=np.float64); scale=np.asarray(norm['scale_high_minus_low'],dtype=np.float64)
    low=np.asarray(norm['active_eta_box_low'],dtype=np.float64); high=np.asarray(norm['active_eta_box_high'],dtype=np.float64)
    directions=list(csv.DictReader(open(REF/'ray_directions.csv')))
    if len(directions)!=18: raise RuntimeError('frozen direction count')
    cloud=list(csv.DictReader(open(QDIR/'eta_probe_cloud.csv')))
    features=np.load(QDIR/'conditioning_features.npz')['features']
    reference={'parent_protocol':str(REF/'protocol.md'),'parent_state_manifest_sha256':sha(REF/'state_manifest.json'),
               'parent_manifest_sha256':sha(REF/'manifest.json'),'parent_cost_sha256':sha(REF/'cost_preflight.json'),
               'first_six_state_ids':[x['state_id'] for x in selection],'future_root_seed':FUTURE_ROOT,
               'orthoflow3_sha256':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),
               'kappa':KAPPA,'ray_radii':RAY_RADII,'inside_sampling':'unscrambled Sobol d=3 first 12; z=1-2u0, theta=2pi*u1, radius=u2^(1/3)',
               'outside_sampling':'unscrambled Sobol d=2 first 8; z=1-2u0, theta=2pi*u1, requested radius=1.15*r_ball'}
    dump(HERE/'protocol_reference.json',reference)
    dump(HERE/'pilot_state_manifest.json',{'schema':'orthoflow3_conservative_basin_ball_pilot6_state_manifest','frozen_prefix_of_parent':True,'parent_state_manifest':str(REF/'state_manifest.json'),'states':selection})
    oracle=Oracle(states,features,low,high)
    # 1. Exact common-cloud completion.
    common_rows=[]; common_by_state={}
    for st in selection:
        sid=st['state_id']; vals=[]
        for index,p in enumerate(cloud):
            eta=np.array([float(p['eta1']),float(p['eta2']),float(p['eta3'])])
            result=oracle.ensure(taskset(sid,eta,8,probe_id=p['probe_id'],cloud_index=index),'common_cloud')
            suc,n=rows_summary(result); rec={'state_id':sid,'cloud_index':index,'probe_id':p['probe_id'],'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'successes':suc,'trials':n,'screen_8of8':suc==8,'source_reused_trials':sum(r['_source']!='pilot' for r in result),'new_trials':sum(r['_source']=='pilot' for r in result)}
            common_rows.append(rec); vals.append(rec)
        common_by_state[sid]=vals
    csv_write(HERE/'common_cloud_completion.csv',common_rows,list(common_rows[0]))
    # 2. Frozen maximin center selection and B63 promotion.
    center_rows=[]; center_results=[]; centers={}
    domain_ctr=np.zeros(3)
    for st in selection:
        sid=st['state_id']; vals=common_by_state[sid]; good=[x for x in vals if x['screen_8of8']]
        failures=[x for x in vals if x['successes']<7]
        ranked=[]
        for x in good:
            e=np.array([x['eta1'],x['eta2'],x['eta3']]); te=norm_eta(e,affine,scale)
            margin=min((np.linalg.norm(te-norm_eta(np.array([f['eta1'],f['eta2'],f['eta3']]),affine,scale)) for f in failures),default=float('nan'))
            if failures: key=(-margin,x['cloud_index'])
            else: key=(float(np.linalg.norm(te-domain_ctr)),0 if x['probe_id']=='zero' else 1,x['cloud_index'])
            ranked.append((key,x,margin))
        ranked.sort(key=lambda x:x[0])
        accepted=None
        for rank,(key,x,margin) in enumerate(ranked):
            eta=np.array([x['eta1'],x['eta2'],x['eta3']]); result=oracle.ensure(taskset(sid,eta,64,probe_id=x['probe_id'],cloud_index=x['cloud_index'],center_rank=rank),'center_b63')
            suc,n=rows_summary(result); b63=suc>=63
            center_rows.append({'state_id':sid,'rank':rank,'probe_id':x['probe_id'],'cloud_index':x['cloud_index'],'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'maximin_failure_distance':margin,'successes':suc,'trials':n,'B63':b63,'selected':b63 and accepted is None,'status':'B63' if b63 else 'NOT_B63'})
            if b63:
                accepted=(eta,x,margin,{'successes':suc,'trials':n,'outcomes':dict(Counter(r['outcome'] for r in result))}); break
        if accepted is None:
            center_results.append({'state_id':sid,'status':'CENTER_UNRESOLVED','eta1':'','eta2':'','eta3':'','successes':'','trials':'','B63':False,'outcomes':''})
        else:
            eta,x,margin,s=accepted; centers[sid]=eta
            center_results.append({'state_id':sid,'status':'CENTER_RESOLVED','eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'successes':s['successes'],'trials':s['trials'],'B63':True,'outcomes':json.dumps(s['outcomes'],sort_keys=True)})
    csv_write(HERE/'center_selection.csv',center_rows,['state_id','rank','probe_id','cloud_index','eta1','eta2','eta3','maximin_failure_distance','successes','trials','B63','selected','status'])
    csv_write(HERE/'robust_center_results.csv',center_results,['state_id','status','eta1','eta2','eta3','successes','trials','B63','outcomes'])
    # 3. Frozen staged rays and six limiting B63 confirmations.
    ray_rows=[]; brackets=[]; robust_radii=[]; ball_rows=[]; balls={}; ray_detail={}
    for st in selection:
        sid=st['state_id']
        if sid not in centers: continue
        # eta=0 is explicitly legal, but it is an isolated special point outside
        # the active affine box.  A positive-radius Euclidean ball about it would
        # contain unauthorized eta values, so the frozen active-box ray protocol
        # has no continuous 3-D domain from which to form an inscribed ball.
        if not in_active(centers[sid], low, high):
            ball_rows.append({'state_id':sid,'status':'BALL_DEGENERATE_ZERO_SPECIAL','c1':centers[sid][0],'c2':centers[sid][1],'c3':centers[sid][2],'r_raw':0.0,'r_ball':0.0,'kappa':KAPPA})
            continue
        ctilde=norm_eta(centers[sid],affine,scale); details=[]
        for dr in directions:
            did=dr['direction_id']; d=np.array([float(dr['d1']),float(dr['d2']),float(dr['d3'])]); rmax=rho_limit(ctilde,d)
            tested=[{'rho':0.0,'successes':64,'trials':64,'screen_success':True,'eta':centers[sid]}]
            last_success=tested[0]; failure=None
            candidates=[r for r in RAY_RADII if r<=rmax+1e-12]
            if rmax>0 and (not candidates or rmax>candidates[-1]+1e-12): candidates.append(rmax)
            for rho in candidates:
                eta=raw_eta(ctilde+rho*d,affine,scale)
                if not in_active(eta,low,high): continue
                result=oracle.ensure(taskset(sid,eta,8,direction_id=did,rho=float(rho)),'ray_screen')
                suc,n=rows_summary(result); item={'rho':float(rho),'successes':suc,'trials':n,'screen_success':suc==8,'eta':eta}; tested.append(item)
                ray_rows.append({'state_id':sid,'direction_id':did,'rho':rho,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'successes':suc,'trials':n,'screen_success':suc==8,'stage':'coarse_or_domain'})
                if suc==8: last_success=item; continue
                failure=item; break
            if failure is not None:
                lo=last_success['rho']; hi=failure['rho']
                for bi in range(3):
                    mid=(lo+hi)/2; eta=raw_eta(ctilde+mid*d,affine,scale)
                    result=oracle.ensure(taskset(sid,eta,8,direction_id=did,rho=float(mid),bisection_round=bi+1),'ray_bisection')
                    suc,n=rows_summary(result); item={'rho':float(mid),'successes':suc,'trials':n,'screen_success':suc==8,'eta':eta}; tested.append(item)
                    ray_rows.append({'state_id':sid,'direction_id':did,'rho':mid,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'successes':suc,'trials':n,'screen_success':suc==8,'stage':f'bisection_{bi+1}'})
                    if suc==8: lo=mid; last_success=item
                    else: hi=mid; failure=item
                brackets.append({'state_id':sid,'direction_id':did,'rho_success':lo,'rho_failure':hi,'domain_limited':False})
            else:
                brackets.append({'state_id':sid,'direction_id':did,'rho_success':last_success['rho'],'rho_failure':'','domain_limited':True})
            details.append({'direction_id':did,'d':d,'rmax':rmax,'tested':tested,'estimate':last_success['rho'],'last_success':last_success,'failure':failure})
        ray_detail[sid]=details
        limiting=sorted(details,key=lambda x:(x['estimate'],x['direction_id']))[:6]
        confirmed=[]
        for rank,item in enumerate(limiting):
            candidates=sorted([x for x in item['tested'] if x['screen_success']],key=lambda x:x['rho'],reverse=True)
            selected=None
            for candidate in candidates:
                result=oracle.ensure(taskset(sid,candidate['eta'],64,direction_id=item['direction_id'],rho=float(candidate['rho']),limiting_rank=rank),'boundary_b63')
                suc,n=rows_summary(result); b63=suc>=63
                robust_radii.append({'state_id':sid,'direction_id':item['direction_id'],'limiting_rank':rank,'rho':candidate['rho'],'eta1':candidate['eta'][0],'eta2':candidate['eta'][1],'eta3':candidate['eta'][2],'successes':suc,'trials':n,'B63':b63,'selected':b63 and selected is None})
                if b63: selected=candidate; break
            if selected is not None: confirmed.append(selected['rho'])
        if len(confirmed)==6:
            rawr=float(min(confirmed)); ballr=KAPPA*rawr; balls[sid]={'c':centers[sid],'r_raw':rawr,'r_ball':ballr}
            ballspec={'state_id':sid,'status':'BALL_RESOLVED','c1':centers[sid][0],'c2':centers[sid][1],'c3':centers[sid][2],'r_raw':rawr,'r_ball':ballr,'kappa':KAPPA}
        else:
            ballspec={'state_id':sid,'status':'BALL_UNRESOLVED','c1':'','c2':'','c3':'','r_raw':'','r_ball':'','kappa':KAPPA}
        ball_rows.append(ballspec)
    csv_write(HERE/'ray_screening.csv',ray_rows,['state_id','direction_id','rho','eta1','eta2','eta3','successes','trials','screen_success','stage'])
    csv_write(HERE/'boundary_brackets.csv',brackets,['state_id','direction_id','rho_success','rho_failure','domain_limited'])
    csv_write(HERE/'robust_directional_radii.csv',robust_radii,['state_id','direction_id','limiting_rank','rho','eta1','eta2','eta3','successes','trials','B63','selected'])
    csv_write(HERE/'conservative_ball_parameters.csv',ball_rows,['state_id','status','c1','c2','c3','r_raw','r_ball','kappa'])
    # 4. Independent Sobol-to-ball inside validation.
    sob=qmc.Sobol(d=3,scramble=False).random_base2(4)[:12]
    unit=[]
    for u in sob:
        z=1-2*u[0]; theta=2*math.pi*u[1]; radial=float(u[2]**(1/3)); rr=math.sqrt(max(0,1-z*z)); unit.append(np.array([rr*math.cos(theta),rr*math.sin(theta),z])*radial)
    inside_points=[]; inside_validation=[]; outside_points=[]; outside_validation=[]
    outu=qmc.Sobol(d=2,scramble=False).random_base2(3)[:8]
    for sid,ball in balls.items():
        ctilde=norm_eta(ball['c'],affine,scale); rball=ball['r_ball']
        radii=np.array([np.linalg.norm(x) for x in unit]); forced={int(np.argmin(radii)),int(np.argmin(abs(radii-.5))),int(np.argmin(abs(radii-.9)))}
        for k,u in enumerate(unit):
            eta=raw_eta(ctilde+rball*u,affine,scale); normr=float(np.linalg.norm(u)*rball)
            result=oracle.ensure(taskset(sid,eta,8,point_id=f'I{k:02d}',inside=True),'inside_screen')
            suc,n=rows_summary(result); promote=(suc<8 or k in forced)
            final=result
            if promote: final=oracle.ensure(taskset(sid,eta,64,point_id=f'I{k:02d}',inside=True),'inside_b63')
            fs,fn=rows_summary(final)
            inside_points.append({'state_id':sid,'point_id':f'I{k:02d}','eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'normalized_radius':normr,'forced_B63':k in forced})
            inside_validation.append({'state_id':sid,'point_id':f'I{k:02d}','screen_successes':suc,'screen_trials':n,'promoted_B63':promote,'successes':fs,'trials':fn,'B63':fs>=63,'false_inclusion':promote and fs<63})
        for k,u in enumerate(outu):
            z=1-2*u[0]; theta=2*math.pi*u[1]; rr=math.sqrt(max(0,1-z*z)); direction=np.array([rr*math.cos(theta),rr*math.sin(theta),z])
            requested=raw_eta(ctilde+1.15*rball*direction,affine,scale); eta=np.clip(requested,low,high); clipped=not np.allclose(requested,eta,atol=1e-12)
            result=oracle.ensure(taskset(sid,eta,8,point_id=f'O{k:02d}',outside=True),'outside_shell')
            suc,n=rows_summary(result)
            outside_points.append({'state_id':sid,'point_id':f'O{k:02d}','eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'requested_eta1':requested[0],'requested_eta2':requested[1],'requested_eta3':requested[2],'clipped':clipped,'actual_normalized_radius':float(np.linalg.norm(norm_eta(eta,affine,scale)-ctilde))})
            outside_validation.append({'state_id':sid,'point_id':f'O{k:02d}','successes':suc,'trials':n,'screen_8of8':suc==8})
    csv_write(HERE/'inside_ball_points.csv',inside_points,['state_id','point_id','eta1','eta2','eta3','normalized_radius','forced_B63'])
    csv_write(HERE/'inside_ball_validation.csv',inside_validation,['state_id','point_id','screen_successes','screen_trials','promoted_B63','successes','trials','B63','false_inclusion'])
    csv_write(HERE/'outside_shell_points.csv',outside_points,['state_id','point_id','eta1','eta2','eta3','requested_eta1','requested_eta2','requested_eta3','clipped','actual_normalized_radius'])
    csv_write(HERE/'outside_shell_validation.csv',outside_validation,['state_id','point_id','successes','trials','screen_8of8'])
    # 5. Fixed anisotropy summaries; use last robust/screen-success ray radius for all directions.
    anis=[]
    for sid,details in ray_detail.items():
        rs=np.array([x['estimate'] for x in details],dtype=float); ratio=float(np.max(rs)/max(np.min(rs),1e-12)); cv=float(np.std(rs)/max(np.mean(rs),1e-12)); cat='mildly anisotropic' if ratio<=1.5 else ('moderately anisotropic' if ratio<=2.5 else 'strongly anisotropic')
        anis.append({'state_id':sid,'direction_count':len(rs),'min_radius':float(np.min(rs)),'max_radius':float(np.max(rs)),'anisotropy_ratio':ratio,'coefficient_variation':cv,'category':cat})
    csv_write(HERE/'anisotropy_statistics.csv',anis,['state_id','direction_count','min_radius','max_radius','anisotropy_ratio','coefficient_variation','category'])
    # 6. Read-only Q exploitation check only if exact state identity is in pilot.
    explo=[]
    for r in csv.DictReader(open(QGDIR/'critic_exploitation_audit.csv')):
        if r['critic_exploitation_candidate']!='True': continue
        sid=r['state_id']; eta=np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])])
        if sid not in balls: status='NOT_APPLICABLE'; inside='NOT_APPLICABLE'
        else:
            b=balls[sid]; inside=bool(np.linalg.norm(norm_eta(eta,affine,scale)-norm_eta(b['c'],affine,scale))<=b['r_ball']+1e-12); status='EXACT_STATE_MATCH'
        explo.append({'state_id':sid,'Qhat':r['Qhat'],'true_Q64':r['true_Q64'],'exact_state_match_status':status,'inside_ball':inside})
    csv_write(HERE/'q_exploitation_case_check.csv',explo,['state_id','Qhat','true_Q64','exact_state_match_status','inside_ball'])
    # Cache/reproducibility and descriptive final decision.
    reused=sum(1 for r in oracle.known.values() if r['_source']!='pilot')
    invalid_rows=[]
    zero_center_ids={sid for sid,c in centers.items() if not in_active(c,low,high)}
    for r in oracle.rows:
        eta=np.asarray(r['eta'],dtype=np.float64)
        if r['phase'].startswith('ray') and r['state_id'] in zero_center_ids:
            invalid_rows.append({'state_id':r['state_id'],'phase':r['phase'],'future_index':r['future_index'],'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'outcome':r['outcome'],'quarantine_reason':'pre-patch ray query from an isolated explicit-zero center; the zero-to-ACTIVE segment crosses the frozen eta-domain gap, so the complete ray query is excluded from geometry conclusions'})
    csv_write(HERE/'quarantined_prepatch_domain_queries.csv',invalid_rows,['state_id','phase','future_index','eta1','eta2','eta3','outcome','quarantine_reason'])
    cache={'exact_tuple_key':'state_id, float64 eta bytes, future_index, frozen Q-v2 current-Flow conditioning, future_root, implementation/projection/horizon','source_roots':[str(QDIR),str(DDIR),str(QGDIR)],'exact_reused_tuples_in_lookup':reused,'exact_source_tuples_used_by_protocol':len(oracle.used_source_keys),'prior_pilot_tuples_reused_on_finalization':len(oracle.used_prior_pilot_keys),'new_pilot_tuples_total':oracle.new,'quarantined_prepatch_domain_queries':len(invalid_rows),'new_valid_protocol_tuples':oracle.new-len(invalid_rows),'future_root_seed':FUTURE_ROOT}
    dump(HERE/'cache_reuse_audit.json',cache)
    resolved=len(balls); b63_false=sum(1 for r in inside_validation if r['promoted_B63'] and not r['B63']); screen_bad=sum(1 for r in inside_validation if r['screen_successes']<8)
    strong_aniso=sum(r['category']=='strongly anisotropic' for r in anis)
    zero_special_degenerate=sum(r['status']=='BALL_DEGENERATE_ZERO_SPECIAL' for r in ball_rows)
    if zero_special_degenerate:
        cls='BALL_PILOT_FAILS_FEASIBILITY'
    elif resolved<6: cls='BALL_PILOT_UNDERRESOLVED'
    elif b63_false>=2: cls='BALL_PILOT_FAILS_FEASIBILITY'
    elif strong_aniso>=4: cls='BALL_PILOT_SPHERE_TOO_RESTRICTIVE'
    elif b63_false==0: cls='BALL_PILOT_STRONGLY_SUPPORTED'
    else: cls='BALL_PILOT_PROMISING_BUT_CONSERVATIVE'
    decision={'classification':cls,'centers_resolved':len(centers),'balls_resolved':resolved,'zero_special_degenerate_centers':zero_special_degenerate,'inside_screening_failures':screen_bad,'inside_B63_false_inclusions':b63_false,'sphere_geometry_only':True,
              'future_H_label_experiment_justified':cls in ('BALL_PILOT_STRONGLY_SUPPORTED','BALL_PILOT_PROMISING_BUT_CONSERVATIVE'),
              'single_next_step': 'Do not train H yet; replicate this unchanged six-state protocol on the next frozen six-state prefix block.' if cls in ('BALL_PILOT_STRONGLY_SUPPORTED','BALL_PILOT_PROMISING_BUT_CONSERVATIVE') else 'Before any learned predictor, predeclare and audit a disconnected feasibility-set representation: the explicit zero singleton plus a continuous active-domain inner set.'}
    dump(HERE/'pilot_decision.json',decision)
    radii=[v['r_ball'] for v in balls.values()]
    report=['# OrthoFlow3 conservative basin-ball pilot6 v1','',f'## Decision\n\n**{cls}**','',f'- States: {len(selection)}; B63 centers: {len(centers)}; resolved balls: {resolved}.',f'- Explicit-zero special centers that cannot support a positive-radius active-domain ball: {zero_special_degenerate}.',f'- New continuations: {oracle.new} total ({oracle.new-len(invalid_rows)} valid-protocol; {len(invalid_rows)} quarantined pre-patch domain-gap queries); new physical steps: {oracle.steps}.',f'- Inside screening failures: {screen_bad}; B63 false inclusions among promoted inside points: {b63_false}.','', '## Frozen-state outcomes','']
    for st in selection:
        sid=st['state_id']; vals=common_by_state[sid]; zero=next(x for x in vals if x['probe_id']=='zero')
        report.append(f'- `{sid}`: zero={zero["successes"]}/8; max common-cloud result={max(int(x["successes"]) for x in vals)}/8; 8/8 candidates={sum(x["screen_8of8"] for x in vals)}.')
    report += ['', '## Interpretation', '', 'Three states resolved to `c0=(0,0,0)` with Q64=64/64.  The frozen eta domain treats zero as an explicit special point outside the continuous ACTIVE box `[0.5,1.25] x [-0.5,0.5] x [0,0.75]`; therefore any positive-radius Euclidean ball centered at zero necessarily includes unauthorized eta values.  These are correctly recorded as `BALL_DEGENERATE_ZERO_SPECIAL`, not treated as positive-radius balls.', '', 'The other three states had no 8/8 common-cloud candidate (their maximum screening result was 6/8), hence the frozen center rule provided no candidate eligible for B63 promotion.  Consequently no valid positive-radius ball exists in this pilot, and independent inside/shell validation and anisotropy are not applicable rather than zero-error results.', '', f'The two archived critic-exploitation cases have no exact state match in this six-state prefix, so their ball membership is NOT_APPLICABLE.  The {len(invalid_rows)} pre-patch zero-center ray records are preserved in a quarantine CSV and excluded from all geometry calculations.', '', 'Sparse directional probing did not construct a useful generic continuous 3-D inner ball under the exact frozen domain.  A robust zero center is a plausible controller value for its individual zero-sufficient states, but it is not a generic positive-radius ball-center label; these results do not justify training `H(h)->[c,r]`.']
    if radii: report += [f'- Ball radii: mean {np.mean(radii):.6f}, median {np.median(radii):.6f}, min {np.min(radii):.6f}, max {np.max(radii):.6f}.']
    if outside_validation: report += [f"- Outside-shell 8/8 successes: {sum(r['screen_8of8'] for r in outside_validation)}/{len(outside_validation)}."]
    report += ['', 'No network was trained and no controller semantics were modified.']
    (HERE/'basin_ball_pilot6_report.md').write_text('\n'.join(report)+'\n')
    runtime={'new_continuations_total':oracle.new,'new_valid_protocol_continuations':oracle.new-len(invalid_rows),'quarantined_prepatch_domain_queries':len(invalid_rows),'physical_steps':oracle.steps,'wall_time_seconds':time.monotonic()-oracle.started,'gpu_shards':1,'cpu_threads':8,'device':[str(x) for x in jax.devices()],'host':platform.node(),'outcomes_new':dict(Counter(r['outcome'] for r in oracle.rows))}
    dump(HERE/'runtime_statistics.json',runtime)
    files=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name!='manifest.json': files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(HERE/'manifest.json',{'schema':'orthoflow3_conservative_basin_ball_pilot6_v1','classification':cls,'files':files})
    print(json.dumps({'decision':decision,'runtime':runtime},indent=2),flush=True)


if __name__=='__main__':
    main()
