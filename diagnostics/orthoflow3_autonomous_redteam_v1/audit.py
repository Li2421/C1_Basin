"""Read-only evidence red team. No runtime construction, training or rollout calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path('/home/zhihan/research/Basin_C1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
from shared_rollout_db.src.rollout_db import canonical, eta_identity, uid

def read(p): return json.loads(Path(p).read_text())
def decode(x):
    for _ in range(8):
        if not isinstance(x, str): return x
        try: x = json.loads(x)
        except json.JSONDecodeError: return x
    raise ValueError('Unrecognized nested JSON')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def save(p, x):
    p = OUT / p; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')
def conn():
    c=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro', uri=True)
    c.row_factory=sqlite3.Row
    return c
def evidence(rows):
    valid=[r for r in rows if not r['numerical_failure']]
    s=sum(bool(r['success']) for r in valid); f=len(valid)-s
    u=sum(bool(r['numerical_failure']) for r in rows); missing=16-len(rows)
    assert missing>=0
    return dict(successes=s,valid_failures=f,numerical=u,missing=missing,
                Q16=s/16 if not u and not missing else None,Q_lower=s/16,
                Q_upper=(s+u+missing)/16,robust=True if s>=15 else False if f>=2 else None,
                collision=sum(bool(r['collision']) for r in valid))
def lookup(c, sid, eid, ctl):
    rr=[dict(r) for r in c.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0', (sid,eid,ctl))]
    rows={r['seed_key']:r for r in rr}
    return [rows[canonical({'future_index':i})] for i in range(16) if canonical({'future_index':i}) in rows]
def freeze():
    paths=[
      'diagnostics/orthoflow3_ring_k16_diagnostic_v1/frozen_proposals.json',
      'diagnostics/orthoflow3_ring_k16_diagnostic_v1/run_k16.py',
      'diagnostics/orthoflow3_ring_k16_diagnostic_v1/per_state_results.json',
      'diagnostics/orthoflow3_ring_revision_v1/fresh_proposals.json',
      'diagnostics/orthoflow3_ring_revision_v1/fresh_test_manifests.json',
      'diagnostics/orthoflow3_ring_revision_v1/analyze_train_dev.py',
      'diagnostics/orthoflow3_generator_critic_v1/train_evaluate.py',
      'diagnostics/orthoflow3_generator_critic_v1/normalization.json',
      'diagnostics/orthoflow3_generator_critic_v1/generator/seed41/checkpoint.msgpack',
      'diagnostics/orthoflow3_generator_critic_v1/critic/seed23/checkpoint.msgpack',
      'diagnostics/orthoflow3_generator_critic_frozen_test_v1/run_frozen_test.py',
      'new_benchmark_common/basin_dataset_v1.py','new_benchmark_common/safety_eta3.py',
      'new_benchmark_common/macflow.py','shared_control/basis_families.py',
      'shared_control/hard_projection.py','ring_exchange/safety.py',
      'ring_exchange/environment.py','ring_exchange/local_frame.py',
      'four_way_intersection/environment.py','four_way_intersection/safety.py',
      'datasets/orthoflow3_basin_dataset_v2_audited/states.parquet',
      'datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet']
    v2=pq.read_table(ROOT/paths[-2]).to_pylist()
    panel=[]
    for sc in ('four_way_intersection','ring_exchange'):
        for split in ('train','validation'):
            panel += sorted([r for r in v2 if r['scenario']==sc and r['split']==split],
                            key=lambda r:digest(['redteam-v1-panel',r['state_uid']]))[:4]
    doc={'created_at':datetime.now(timezone.utc).isoformat(),
         'source_root':str(ROOT),'worktree':str(OUT.parents[1]),
         'frozen_files':{p:sha(ROOT/p) for p in paths},
         'train_dev_inference_panel':[{'state_uid':r['state_uid'],'scenario':r['scenario'],'split':r['split']} for r in panel],
         'K16_artifact_population':'exact frozen 60 Ring states; semantic reanalysis only, no tuning',
         'new_rollout_limit':3000,'planned_new_rollouts':0,
         'canonical_writes':False,'phase_A_status_at_registration':'acquisition in progress; not accepted',
         'contract_audit_status':'no completed artifact yet',
         'hypotheses':['H1_exact_vs_nearest_label','H2_numerical_Q_bounds','H3_proposal_identity','H4_stopped_likelihood']}
    save('hypothesis_test_manifest.json',doc)
    print(json.dumps({'frozen_files':len(paths),'train_dev_panel':len(panel)}))
def k16():
    path=ROOT/'diagnostics/orthoflow3_ring_k16_diagnostic_v1'
    frozen=read(path/'frozen_proposals.json'); old=read(ROOT/'diagnostics/orthoflow3_ring_revision_v1/fresh_proposals.json')
    old={r['state_uid']:r for r in old['states'] if r['scenario']=='ring_exchange'}
    historical={r['state_uid']:r for r in read(path/'per_state_results.json')}
    results=[]; checks=Counter(); counts=Counter(); controllers={}; refs=[]
    with conn() as c:
        # Pin a WAL read snapshot; concurrent data acquisition is not part of this snapshot.
        c.execute('BEGIN')
        for p in frozen['states']:
            sid=p['state_uid']; etas=[p['mean'],*p['samples']]
            eid=[eta_identity(e)[0] for e in etas]
            assert len(set(eid))==17
            assert p['mean']==old[sid]['mean'] and p['samples'][:4]==old[sid]['samples']
            assert p['candidate_hashes']==[digest({'eta':e}) for e in etas]
            assert p['proposal_set_hash']==digest({'mean':p['mean'],'samples':p['samples']})
            checks['exact_nested_and_hashes']+=1
            # Find exact candidate controller from DB and require all frozen hash fields.
            options=c.execute('SELECT DISTINCT c.* FROM controller_config c JOIN rollout r USING(controller_uid) WHERE r.state_uid=? AND r.eta_uid=?', (sid,eid[-1])).fetchall()
            matches=[]
            for x in options:
                cfg=json.loads(x['config_json'])
                if (x['safety_config_hash']==frozen['ring_safety_hash'] and
                    x['orthoflow3_sha256']==frozen['orthoflow3_hash'] and
                    x['flow_checkpoint_sha256']==frozen['macflow_checkpoint_hash'] and
                    cfg.get('chain')=='orthoflow3'):
                    matches.append(dict(x))
            assert len(matches)==1,(sid,len(matches)); ctl=matches[0]['controller_uid']
            controllers[ctl]=matches[0]
            rr=[]
            for j,e in enumerate(eid):
                rows=lookup(c,sid,e,ctl); assert len(rows)==16,(sid,j,len(rows))
                ev=evidence(rows); counts['requested']+=16
                counts['exact_reuse']+=16-ev['numerical'];counts['uncertified_present']+=ev['numerical']
                refs.append({'state_uid':sid,'eta_uid':e,'controller_uid':ctl,'rollout_uids':[r['rollout_uid'] for r in rows]})
                name='mean' if j==0 else f'sample_{j}'
                prev=historical[sid]['candidate_evidence'][name]
                assert ev['robust'] is prev['robust'] and ev['successes']==prev['successes']
                if ev['numerical']: counts['candidate_partial_Q']+=1
                rr.append(ev)
            pick=p['critic_index']; assert pick==int(np.argmax(p['critic_scores']))
            assert pick==historical[sid]['critic_index']
            oracle=max(range(17),key=lambda j:(rr[j]['Q_lower'],-j))
            assert oracle==historical[sid]['oracle_index']
            sel=rr[pick]; olo=max(e['Q_lower'] for e in rr); ohi=max(e['Q_upper'] for e in rr)
            # Interval regret, nonnegative because selector is a member of oracle set.
            regret=[max(0.,olo-sel['Q_upper']),max(0.,ohi-sel['Q_lower'])]
            possible=[j for j,e in enumerate(rr) if e['Q_upper']>=olo]
            r={'state_uid':sid,'state_id':p['state_id'],'candidate_evidence':rr,
               'critic_index':pick,'oracle_lower_index':oracle,'oracle_possible_indices':possible,
               'oracle_Q_bounds':[olo,ohi],'selected_Q_bounds':[sel['Q_lower'],sel['Q_upper']],
               'regret_bounds':regret,'critic_robust':sel['robust'],
               'oracle_robust':any(e['robust'] is True for e in rr),
               'exploitation_confirmed':p['critic_scores'][pick]>=.9375 and sel['Q_upper']<.5,
               'exploitation_possible':p['critic_scores'][pick]>=.9375 and sel['Q_lower']<.5,
               'collision_seeds':sum(e['collision'] for e in rr),'numerical_seeds':sum(e['numerical'] for e in rr)}
            results.append(r)
    summary={'states':len(results),'oracle_B15':sum(r['oracle_robust'] for r in results),
             'critic_B15':sum(r['critic_robust'] is True for r in results),
             'critic_unknown':sum(r['critic_robust'] is None for r in results),
             'misses':sum(r['oracle_robust'] and r['critic_robust'] is False for r in results),
             'exploitation_confirmed':sum(r['exploitation_confirmed'] for r in results),
             'exploitation_possible':sum(r['exploitation_possible'] for r in results),
             'mean_regret_bounds':np.mean([r['regret_bounds'] for r in results],axis=0).tolist(),
             'critic_mean_Q_bounds':np.mean([r['selected_Q_bounds'] for r in results],axis=0).tolist(),
             'oracle_mean_Q_bounds':np.mean([r['oracle_Q_bounds'] for r in results],axis=0).tolist(),
             'candidate_partial_Q':counts['candidate_partial_Q'],
             'collision_seeds':sum(r['collision_seeds'] for r in results),
             'numerical_seeds':sum(r['numerical_seeds'] for r in results),
             'selected_partial_Q_states':sum(r['selected_Q_bounds'][0]!=r['selected_Q_bounds'][1] for r in results),
             'checks':dict(checks),'class':'CONFIRMED_PROTOCOL_DEFECT_Q_POINT_INSTEAD_OF_INTERVAL',
             'headline_B15_counts_unchanged':True,'new_rollouts':0}
    counts.update(partial_reuse=0,aggregate_reuse=0,truly_missing=0,executed=0)
    save('H2_k16_corrected_summary.json',summary);save('H2_k16_corrected_per_state.json',results)
    save('H2_cache_preflight.json',dict(counts));save('H2_rollout_refs.json',refs)
    save('H3_controller_configs.json',controllers)
    print(json.dumps(summary))
def h1():
    """Exact current safety vs nearest current label on already-frozen proposals."""
    states=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/states.parquet').to_pylist()
    ss={r['state_uid']:r for r in states}
    labels=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet').to_pylist()
    by=defaultdict(list)
    for r in labels: by[r['state_uid']].append(r)
    props=read(ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/phase_a/proposals.json')
    center=np.array([.875,0,.375]);radius=np.array([.375,.5,.375]);out=[]; cache=Counter()
    with conn() as c:
        c.execute('BEGIN')
        for p in props['states']:
            if p['scenario'] not in ('ring_exchange','four_way_intersection'):continue
            st=ss[p['state_uid']]; lab=by[st['state_uid']]
            ctl=st['controller_uid']; assert len({r['controller_uid'] for r in lab})==1
            assert ctl==lab[0]['controller_uid']
            points=np.array([decode(r['eta_raw']) for r in lab]);rowev=[]
            for j,eta in enumerate(p['etas']):
                eu=eta_identity(eta)[0]; exact=lookup(c,st['state_uid'],eu,ctl); ev=evidence(exact)
                cache['requested']+=16;cache['exact_reuse']+=16-ev['numerical']-ev['missing'];cache['numerical_present']+=ev['numerical'];cache['truly_missing']+=ev['missing']
                distances=np.linalg.norm((points-np.array(eta))/radius,axis=1);i=int(np.argmin(distances));nearest=lab[i]
                nq=nearest['success_count']/nearest['seed_count'] if nearest['seed_count'] else None
                rowev.append({'index':j,'eta':eta,'eta_uid':eu,**ev,
                     'nearest_eta_uid':nearest['eta_uid'],'nearest_distance':float(distances[i]),
                     'nearest_robust':nearest['robust_15of16'],'nearest_stopped_fraction':nq,
                     'nearest_seed_count':nearest['seed_count'],
                     'class_disagreement':ev['robust'] is not None and nearest['robust_15of16'] is not None and ev['robust']!=nearest['robust_15of16']})
            oldpick=p['old_index'];out.append({'scenario':p['scenario'],'split':p['split'],'state_uid':st['state_uid'],
                    'state_id':st['state_id'],'timestep':st['timestep'],'selected_index':oldpick,
                    'candidates':rowev,'complete':all(r['missing']==0 for r in rowev)})
    summary={}
    for sc in ('ring_exchange','four_way_intersection'):
        for split in ('train','validation'):
            rr=[r for r in out if r['scenario']==sc and r['split']==split]
            full=[r for r in rr if r['complete']];flat=[e for r in full for e in r['candidates']]
            exact=[e for e in flat if e['Q16'] is not None and e['nearest_stopped_fraction'] is not None]
            summary[f'{sc}/{split}']={'states':len(rr),'complete_states':len(full),'completed_candidates':len(flat),
                'nearest_exact_eta_matches':sum(e['nearest_distance']==0 for e in flat),
                'median_nearest_distance':float(np.median([e['nearest_distance'] for e in flat])) if flat else None,
                'proxy_class_disagreements':sum(e['class_disagreement'] for e in flat),
                'proxy_Q_MAE':float(np.mean([abs(e['Q16']-e['nearest_stopped_fraction']) for e in exact])) if exact else None,
                'proxy_oracle_B15':sum(any(e['nearest_robust'] is True for e in r['candidates']) for r in full),
                'actual_oracle_B15':sum(any(e['robust'] is True for e in r['candidates']) for r in full),
                'proxy_selected_B15':sum(r['candidates'][r['selected_index']]['nearest_robust'] is True for r in full),
                'actual_selected_B15':sum(r['candidates'][r['selected_index']]['robust'] is True for r in full),
                'selected_unknown':sum(r['candidates'][r['selected_index']]['robust'] is None for r in full)}
    cache.update(partial_reuse=0,aggregate_reuse=0,executed=0)
    save('H1_exact_vs_proxy.json',{'summary':summary,'states':out,'read_at':datetime.now(timezone.utc).isoformat(),
               'uses_current_v2_safety_for_both_sides':True,'no_new_K_sweep':True})
    save('H1_cache_preflight.json',dict(cache)); print(json.dumps(summary))
def h4():
    rows=pq.read_table(ROOT/'datasets/orthoflow3_basin_dataset_v2_audited/eta_labels.parquet').to_pylist()
    summary={}
    for sc in ('double_bottleneck','four_way_intersection','ring_exchange'):
        rr=[r for r in rows if r['scenario']==sc]
        summary[sc]={'labels':len(rr),'observed_seed_counts':dict(Counter(str(r['seed_count']) for r in rr)),
            'n_less_than16':sum(r['seed_count']<16 for r in rr),
            'full_Q16_no_numerical':sum(r['seed_count']==16 and r['numerical_failure_count']==0 for r in rr),
            'Q16plus_no_numerical':sum(r['seed_count']>=16 and r['numerical_failure_count']==0 for r in rr),
            'early_stops':dict(Counter(str(r['early_stop_reason']) for r in rr)),
            'uncertified':sum(r['robust_15of16'] is None for r in rr)}
    # Exhaustively integrate the canonical stopping tree. Not a physical rollout.
    experiment=[]
    for p in (.1,.5,.8,.9,.95):
        frontier=[(0,0,1.)]; terminal=[]
        while frontier:
            s,f,w=frontier.pop()
            if s>=15 or f>=2 or s+f==16:
                terminal.append((s,f,w));continue
            frontier.extend([(s+1,f,w*p),(s,f+1,w*(1-p))])
        mass=sum(w for s,f,w in terminal); assert abs(mass-1)<1e-12
        mean_ratio=sum(w*s/(s+f) for s,f,w in terminal)
        expected_score=sum(w*(s/p-f/(1-p)) for s,f,w in terminal)
        tilted=[]
        # Source critic_batch samples B15/non-B15 classes 50/50, not raw eta
        # rows uniformly. Outcome-dependent positive weighting is different
        # from optional stopping alone. Fixed ratio controls are analytic, not
        # claims about the trained network's causal error magnitude.
        for ratio in (1.,3.,3.5):
            es=sum(w*(ratio if s>=15 else 1.)*s for s,f,w in terminal)
            en=sum(w*(ratio if s>=15 else 1.)*(s+f) for s,f,w in terminal)
            tilted.append({'positive_selection_weight_ratio':ratio,'population_likelihood_optimum':es/en})
        experiment.append({'p':p,'expected_stopped_fraction':mean_ratio,'bias':mean_ratio-p,
                           'expected_binomial_score_at_truth':expected_score,'mass':mass,
                           'outcome_conditioned_sampling_control':tilted})
        assert abs(expected_score)<1e-10
    save('H4_stopped_evidence.json',{'dataset':summary,'exact_tree_control':experiment,
          'conclusion':'stopped fraction is not unbiased full-Q16; count-weighted Bernoulli likelihood remains valid under stopping alone. The original critic additionally resamples outcome-derived classes without inverse-probability correction, so that separate sampling scheme need not preserve probability calibration. No quantitative causal attribution to the trained critic is inferred.',
          'phase_A_existing_mitigation':'separately requires full valid Q for aligned critic ranking; not an accepted checkpoint yet'})
    print(json.dumps({'dataset':summary,'control':experiment}))
def verify():
    before=read(OUT/'hypothesis_test_manifest.json')['frozen_files']
    matches={p:sha(ROOT/p)==h for p,h in before.items()}
    save('hash_verification.json',{'matches':matches,'all_match':all(matches.values())})
    print(json.dumps({'all_match':all(matches.values()),'count':len(matches)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','k16','h1','h4','verify'])
    globals()[p.parse_args().stage]()
