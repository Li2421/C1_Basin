#!/usr/bin/env python3
"""Execute the frozen capped robust-eta search for one true-t0 state."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhihan/research/Basin_C1')
HERE = ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
OLD_PATH = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_pilot6_v1/pilot6.py'
CONT_PATH = ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1/continuous_pilot6.py'
FUTURE_ROOT = 2026092811
BATCH = 64


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else ['state_id'])
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def ekey(eta) -> str:
    return np.asarray(eta, dtype=np.float64).tobytes().hex()


def tasks(sid: str, eta, n: int, **extra) -> list[dict]:
    e = np.asarray(eta, dtype=np.float64).tolist()
    return [{'state_id': sid, 'eta': e, 'future_index': i, **extra} for i in range(n)]


def load_runtime(out: Path, sid: str):
    text = OLD_PATH.read_text()
    text = text.replace('for begin in range(0,len(tasks),32):', f'for begin in range(0,len(tasks),{BATCH}):')
    text = text.replace('chunk=tasks[begin:begin+32]', f'chunk=tasks[begin:begin+{BATCH}]')
    # The frozen h0/current-Flow feature was generated with one state per policy call.
    # XLA's stochastic policy kernel has a small batch-shape numerical dependence, so
    # replay the physical t0 action one state at a time; later steps stay vectorized.
    old_action = '                actions=np.asarray(self.sample(jnp.asarray(obs),jnp.asarray(np.stack(keys))))'
    new_action = """\
                at_query_time=all(int(envs[i].step_count)==int(self.states[chunk[i]['state_id']]['absolute_step']) for i in active)
                if at_query_time:
                    actions=np.concatenate([np.asarray(self.sample(jnp.asarray(obs[j:j+1]),jnp.asarray(np.stack(keys[j:j+1])))) for j in range(len(active))],axis=0)
                else:
                    actions=np.asarray(self.sample(jnp.asarray(obs),jnp.asarray(np.stack(keys))))"""
    if old_action not in text:
        raise RuntimeError('first-step replay patch anchor missing')
    text = text.replace(old_action, new_action)
    old_guard = "                            if err>1e-10: raise RuntimeError(('feature replay',chunk[i]['state_id'],err))"
    new_guard = """\
                            if err>1e-10:
                                expected=self.features[int(state['feature_index'])]
                                imax=int(np.argmax(np.abs(feature-expected)))
                                raise RuntimeError(('feature replay',chunk[i]['state_id'],err,'imax',imax,'actual',float(feature[imax]),'expected',float(expected[imax]),'feature_index',int(state['feature_index'])))"""
    if old_guard not in text:
        raise RuntimeError('feature guard patch anchor missing')
    text = text.replace(old_guard, new_guard)
    spec = importlib.util.spec_from_loader('point_old_runtime', loader=None)
    old = importlib.util.module_from_spec(spec); old.__file__ = str(OLD_PATH)
    sys.modules[spec.name] = old; exec(compile(text, old.__file__, 'exec'), old.__dict__)
    old.HERE = out; old.FUTURE_ROOT = FUTURE_ROOT; old.CAP_CONT = 1800; old.CAP_STEPS = 2000000

    class Oracle(old.Oracle):
        def _load_prior(self_inner):
            if self_inner.record_path.exists():
                for line in self_inner.record_path.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line); self_inner.rows.append(row); self_inner._insert(row, 'pilot')
                self_inner.new = len(self_inner.rows)
                self_inner.steps = sum(int(r['continuation_steps']) for r in self_inner.rows)
    return old, Oracle


def load_geometry():
    spec = importlib.util.spec_from_file_location('point_cont_geom', CONT_PATH)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
    _, affine, scale, _, _, _, _, _, eq = mod.geometry()
    return np.asarray(affine), np.asarray(scale), np.asarray(eq), mod.domain_clearance, mod.ray_limit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train','val','test'], required=True)
    ap.add_argument('--rank', type=int, required=True)
    args = ap.parse_args()

    states_all = json.load(open(HERE/'eligible_state_manifest.json'))['selected_states']
    st = next(x for x in states_all if x['split'] == args.split and int(x['split_rank']) == args.rank)
    sid = st['state_id']; out = HERE/'state_runs'/sid
    out.mkdir(parents=True, exist_ok=True); (out/'raw').mkdir(exist_ok=True)
    if (out/'selected_target.json').exists():
        print(json.dumps({'state_id': sid, 'status': 'already_complete'})); return

    features = np.load(HERE/'conditioning_features.npz')['features']
    old, Oracle = load_runtime(out, sid)
    oracle = Oracle({sid: st}, features, np.zeros(3), np.ones(3))
    affine, scale, eq, domain_clearance, ray_limit = load_geometry()
    cloud_rows = list(csv.DictReader(open(HERE/'global_eta_sobol64.csv')))
    cloud = [np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])]) for r in cloud_rows]
    candidates = [
        {'candidate_index': i, 'candidate_id': f'sobol_{i:02d}', 'eta': e, 'kind': 'sobol'}
        for i,e in enumerate(cloud)
    ]
    candidates.extend([
        {'candidate_index': 64, 'candidate_id': 'anchor_zero', 'eta': np.zeros(3), 'kind': 'generic_anchor'},
        {'candidate_index': 65, 'candidate_id': 'anchor_intermediate_common', 'eta': np.array([.625,0.,.375]), 'kind': 'generic_anchor'},
    ])
    started = time.time(); screen_rows=[]; promotions=[]; local_rows=[]

    def summarize(rows):
        outcomes = Counter(r['outcome'] for r in rows)
        return sum(bool(r['success']) for r in rows), outcomes

    def screen(items, stage):
        all_tasks=[]
        for c in items:
            all_tasks.extend(tasks(sid,c['eta'],8,candidate_id=c['candidate_id'],candidate_index=c['candidate_index'],search_stage=stage))
        rows = oracle.ensure(all_tasks, f'{stage}_screen')
        result=[]
        for j,c in enumerate(items):
            rr=rows[j*8:(j+1)*8]; suc, outs=summarize(rr)
            rec={'state_id':sid,'split':args.split,'split_rank':args.rank,'stage':stage,
                 'candidate_id':c['candidate_id'],'candidate_index':c['candidate_index'],'candidate_kind':c['kind'],
                 'eta1':c['eta'][0],'eta2':c['eta'][1],'eta3':c['eta'][2],
                 'successes':suc,'trials':8,'screen_status':'SCREEN_STRONG' if suc==8 else ('SCREEN_INTERMEDIATE' if suc>=6 else 'SCREEN_WEAK'),
                 'deadlock':outs['safe_deadlock'],'timeout':outs['timeout'],'collision':outs['collision'],'numerical':outs['other_numerical']}
            screen_rows.append(rec); result.append(rec)
        return result

    def normalized_eta(rec):
        return (np.array([rec['eta1'],rec['eta2'],rec['eta3']],float)-affine)/scale

    def ranked(screened, allowed_successes, already):
        failures=[normalized_eta(x) for x in screened if int(x['successes'])<=5]
        result=[]
        for x in screened:
            if int(x['successes']) not in allowed_successes or x['candidate_id'] in already: continue
            et=normalized_eta(x); dd=domain_clearance(et,eq)
            df=min((float(np.linalg.norm(et-f)) for f in failures),default=math.inf)
            primary=df if math.isfinite(df) else dd
            result.append((-primary,-dd,int(x['candidate_index']),x['candidate_id'],x,df,dd))
        result.sort(key=lambda z:z[:4])
        return result

    exact_b63=[]; exact_non=[]; promoted=set()
    def promote(items, stage, limit):
        for order,item in enumerate(items[:limit]):
            _,_,_,_,x,df,dd=item; eta=np.array([x['eta1'],x['eta2'],x['eta3']],float)
            rr=oracle.ensure(tasks(sid,eta,64,candidate_id=x['candidate_id'],candidate_index=x['candidate_index'],promotion_stage=stage),f'{stage}_q64')
            suc,outs=summarize(rr); b63=suc>=63; promoted.add(x['candidate_id'])
            rec={**x,'promotion_stage':stage,'promotion_rank':order,'d_fail_screen':df,'domain_clearance':dd,
                 'successes64':suc,'Q64':suc/64,'B63':b63,'deadlock64':outs['safe_deadlock'],'timeout64':outs['timeout'],
                 'collision64':outs['collision'],'numerical64':outs['other_numerical']}
            promotions.append(rec); (exact_b63 if b63 else exact_non).append(rec)

    current=screen([*candidates[:32],*candidates[64:]],'stageA32')
    promote(ranked(current,{8},promoted),'stageA_strong',4)
    if not exact_b63:
        extension=screen(candidates[32:64],'stageB32')
        current=current+extension
        promote(ranked(current,{8},promoted),'stageB_strong',4)
        if not exact_b63:
            promote(ranked(current,{7},promoted),'stageB_borderline7',2)

    # Local robustness is a ranking diagnostic for every exact B63 candidate.
    for b in exact_b63:
        eta=np.array([b['eta1'],b['eta2'],b['eta3']],float); et=(eta-affine)/scale
        probe_defs=[]
        for axis in range(3):
            for sign in (-1,1):
                direction=np.zeros(3); direction[axis]=sign
                lim=ray_limit(et,direction,eq); actual=min(.025,max(0.,lim-1e-10))
                if actual <= 1e-9: continue
                pet=et+actual*direction; pe=affine+scale*pet
                probe_defs.append((axis,sign,actual,pe))
        all_tasks=[]
        for axis,sign,actual,pe in probe_defs:
            all_tasks.extend(tasks(sid,pe,8,parent_candidate_id=b['candidate_id'],axis=axis,sign=sign,requested_delta=.025,actual_delta=actual))
        rows=oracle.ensure(all_tasks,f'local_{b["candidate_id"]}') if all_tasks else []
        robust=0
        for j,(axis,sign,actual,pe) in enumerate(probe_defs):
            rr=rows[j*8:(j+1)*8]; suc,outs=summarize(rr); robust+=int(suc==8)
            local_rows.append({'state_id':sid,'parent_candidate_id':b['candidate_id'],'axis':axis+1,'sign':sign,
                               'requested_delta':.025,'actual_delta':actual,'eta1':pe[0],'eta2':pe[1],'eta3':pe[2],
                               'successes':suc,'trials':8,'screen_8of8':suc==8,'deadlock':outs['safe_deadlock'],
                               'timeout':outs['timeout'],'collision':outs['collision']})
        b['local_robust_count']=robust; b['valid_local_perturbations']=len(probe_defs)

    write_csv(out/'eta_screening.csv',screen_rows)
    write_csv(out/'q64_promotions.csv',promotions)
    write_csv(out/'local_robustness_probes.csv',local_rows)
    if exact_b63:
        fail_exact=[normalized_eta(x) for x in exact_non]
        fail_screen=[normalized_eta(x) for x in screen_rows if int(x['successes'])<=5]
        for x in exact_b63:
            et=normalized_eta(x)
            x['nearest_exact_nonB63_distance']=min((float(np.linalg.norm(et-f)) for f in fail_exact),default=math.inf)
            x['nearest_screen_failure_distance']=min((float(np.linalg.norm(et-f)) for f in fail_screen),default=math.inf)
        exact_b63.sort(key=lambda x:(-int(x['local_robust_count']),-float(x['nearest_exact_nonB63_distance']),
                                     -float(x['nearest_screen_failure_distance']),-float(x['domain_clearance']),int(x['candidate_index'])))
        target=exact_b63[0]
        count=int(target['local_robust_count'])
        tag='ROBUST_INTERIOR_STRONG' if count>=5 else ('ROBUST_INTERIOR_MODERATE' if count>=3 else 'ROBUST_POINT_ONLY')
        selected={'state_id':sid,'source_group':st['source_group'],'split':args.split,'split_rank':args.rank,
                  'status':'POINT_USABLE','candidate_id':target['candidate_id'],'candidate_index':target['candidate_index'],
                  'eta1':target['eta1'],'eta2':target['eta2'],'eta3':target['eta3'],'successes64':target['successes64'],
                  'Q64':target['Q64'],'B63':True,'local_robust_count':count,
                  'valid_local_perturbations':target['valid_local_perturbations'],
                  'nearest_exact_nonB63_distance':target['nearest_exact_nonB63_distance'],
                  'nearest_screen_failure_distance':target['nearest_screen_failure_distance'],
                  'domain_clearance':target['domain_clearance'],'quality_tag':tag,
                  'feature_index':st['feature_index'],'feature_sha256':st['feature_sha256']}
    else:
        selected={'state_id':sid,'source_group':st['source_group'],'split':args.split,'split_rank':args.rank,
                  'status':'POINT_UNRESOLVED','reason':'no B63 after frozen capped protocol','feature_index':st['feature_index'],
                  'feature_sha256':st['feature_sha256']}
    (out/'selected_target.json').write_text(json.dumps(selected,indent=2,sort_keys=True)+'\n')
    runtime={'state_id':sid,'split':args.split,'split_rank':args.rank,'new_continuations':oracle.new,
             'physical_steps':oracle.steps,'wall_seconds':time.time()-started,'screened_eta':len(screen_rows),
             'q64_promotions':len(promotions),'b63_candidates':len(exact_b63),'local_probe_eta':len(local_rows),
             'max_gpu_shards_for_worker':1,'batch_size':BATCH,'outcomes':dict(Counter(r['outcome'] for r in oracle.rows))}
    (out/'runtime.json').write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'target':selected,'runtime':runtime},indent=2),flush=True)


if __name__=='__main__':
    main()
