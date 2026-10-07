"""Analyze frozen local-state probes from canonical DB records after postflight."""
from __future__ import annotations
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src.rollout_db import connect, canonical, lookup_exact

def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, sort_keys=True, allow_nan=False)+'\n')
def rows(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def csvwrite(p, rs):
    with Path(p).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
def ci(k,n):
    return (0.0 if k==0 else float(beta.ppf(.025,k,n-k+1)),
            1.0 if k==n else float(beta.ppf(.975,k+1,n-k)))

def main():
    post=read(OUT/'cache_postflight.json')
    assert post['summary']['exact_reusable']==320 and post['summary']['genuinely_missing']==0
    frozen=read(OUT/'frozen_proposals.json')
    protocol=read(OUT/'protocol.json')
    assert hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest()==frozen['protocol_sha256']
    for cp in frozen['critic_checkpoints']:
        assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest()==cp['sha256']
    predictions={(int(r['episode_index']),r['kind']):r for r in rows(OUT/'frozen_predictions.csv')}
    seeds=[canonical({'future_index':j}) for j in range(16)]
    cached_files={}
    integrity=[]
    # Independently confirm the matched RNG rollout_id omitted by legacy physical-state keys.
    def get(con, mapping, state):
        x=lookup_exact(con, *(mapping[k] for k in ('state_uid','eta_uid','controller_uid')), seeds)
        assert x['status']=='EXACT_REUSE'
        records=sorted(x['records'], key=lambda r:json.loads(r['seed_key'])['future_index'])
        assert len(records)==16
        for r in records:
            assert r['compatibility_quality']=='EXACT_REUSE' and not r['conflict_quarantined'] and not r['numerical_failure']
            sources=con.execute('SELECT sf.path,rs.source_line FROM rollout_source rs JOIN source_file sf USING(source_uid) WHERE rs.rollout_uid=?',(r['rollout_uid'],)).fetchall()
            found=False
            for src in sources:
                p=Path(src['path'])
                if not p.exists() or p.suffix not in ('.merged','.jsonl'): continue
                if str(p) not in cached_files: cached_files[str(p)]=p.read_text().splitlines()
                raw=json.loads(cached_files[str(p)][src['source_line']-1])
                if raw.get('rollout_id') is None: continue
                assert raw['rollout_id']==state['rollout_id']
                assert raw['initial_positions']==state['initial_positions']
                assert raw['eta']==state['eta'][mapping['kind']]
                assert raw['rng_semantics_version']==protocol['rng']
                found=True; break
            assert found, ('cannot verify original matched RNG',r['rollout_uid'])
        integrity.append({'state_uid':mapping['state_uid'],'eta_uid':mapping['eta_uid'],'controller_uid':mapping['controller_uid'],
                          'matched_records_verified':16,'rollout_id':state['rollout_id'],'source_group':state['source_group']})
        return records
    per=[]; paired=[]; refs={}; all_new=[]
    def summary(records):
        k=sum(r['success'] for r in records); n=len(records); lo,hi=ci(k,n)
        return {'n':n,'k':k,'Q16':k/n,'B15':int(k>=15),'Q_lower95':lo,'Q_upper95':hi,
                **{key:sum(r[key] for r in records) for key in ('deadlock','timeout','collision','numerical_failure')},
                'mean_J_def':float(np.mean([r['j_def'] for r in records])),
                'mean_episode_steps':float(np.mean([r['episode_length'] for r in records]))}
    with connect(True) as con:
        for mapping in rows(OUT/'reference_cache_keys.csv'):
            ep=int(mapping['episode_index']); kind=mapping['kind']
            state=next(s for s in frozen['references'] if s['episode_index']==ep)
            r=get(con,mapping,state); refs[(ep,kind)]=r
            p=state['reference_critic_probabilities'][0 if kind=='frozen_bad' else 1]
            s=summary(r)
            assert s['Q16']==state['original_bad_q16' if kind=='frozen_bad' else 'original_good_q16']
            per.append({'parent_episode_index':ep,'offset_m':0.0,'kind':kind,'critic_probability':p,
                        'h_normalized_rms_from_parent':0.0,**s,'overestimation':p-s['Q16'],
                        'observed_or_lower_success_binomial_p_at_prediction':float(binom.cdf(s['k'],16,p))})
        for mapping in rows(OUT/'candidate_cache_keys.csv'):
            index=int(mapping['episode_index']); kind=mapping['kind']; state=frozen['states'][index]
            r=get(con,mapping,state); all_new.extend(r); ep=state['parent_episode_index']; old=refs[(ep,kind)]
            pred=predictions[(index,kind)]; p=float(pred['critic_probability']); s=summary(r)
            per.append({'parent_episode_index':ep,'offset_m':state['physical_offset_m'],'kind':kind,'critic_probability':p,
                        'h_normalized_rms_from_parent':state['h_normalized_rms_from_parent'],**s,
                        'overestimation':p-s['Q16'],'observed_or_lower_success_binomial_p_at_prediction':float(binom.cdf(s['k'],16,p))})
            rescue=sum(a['success'] and not b['success'] for a,b in zip(r,old))
            brk=sum(b['success'] and not a['success'] for a,b in zip(r,old))
            paired.append({'parent_episode_index':ep,'offset_m':state['physical_offset_m'],'kind':kind,
                'parent_k':sum(a['success'] for a in old),'perturbed_k':s['k'],'matched_seed_rescue':rescue,
                'matched_seed_break':brk,'net_success_change':rescue-brk,'Q_change':(rescue-brk)/16})
    assert len(all_new)==320 and len({r['rollout_uid'] for r in all_new})==320
    per.sort(key=lambda r:(r['parent_episode_index'],r['offset_m'],r['kind']))
    csvwrite(OUT/'per_state_eta_results.csv',per)
    csvwrite(OUT/'matched_seed_change.csv',paired)
    csvwrite(OUT/'cache_identity_verified.csv',integrity)
    families=[]
    for ep in protocol['parent_episodes']:
        rr=[r for r in per if r['parent_episode_index']==ep]
        bad=sorted((r for r in rr if r['kind']=='frozen_bad'),key=lambda r:r['offset_m'])
        good=sorted((r for r in rr if r['kind']=='frozen_good'),key=lambda r:r['offset_m'])
        families.append({'parent_episode_index':ep,
            'bad_k_minus':bad[0]['k'],'bad_k_center':bad[1]['k'],'bad_k_plus':bad[2]['k'],
            'good_k_minus':good[0]['k'],'good_k_center':good[1]['k'],'good_k_plus':good[2]['k'],
            'bad_p_minus':bad[0]['critic_probability'],'bad_p_center':bad[1]['critic_probability'],'bad_p_plus':bad[2]['critic_probability'],
            'good_p_minus':good[0]['critic_probability'],'good_p_center':good[1]['critic_probability'],'good_p_plus':good[2]['critic_probability'],
            'bad_q_range':max(r['Q16'] for r in bad)-min(r['Q16'] for r in bad),
            'good_q_range':max(r['Q16'] for r in good)-min(r['Q16'] for r in good),
            'new_good_B15_count':sum(r['B15'] for r in good if r['offset_m']!=0),
            'new_bad_B15_count':sum(r['B15'] for r in bad if r['offset_m']!=0)})
    csvwrite(OUT/'family_summary.csv',families)
    bad=[r for r in per if r['kind']=='frozen_bad' and r['offset_m']!=0]
    good=[r for r in per if r['kind']=='frozen_good' and r['offset_m']!=0]
    pairs=[]
    for state in frozen['states']:
        rr=[r for r in per if r['parent_episode_index']==state['parent_episode_index'] and r['offset_m']==state['physical_offset_m']]
        chosen=max(rr,key=lambda r:r['critic_probability'])
        pairs.append({'parent_episode_index':state['parent_episode_index'],'offset_m':state['physical_offset_m'],
                      'critic_chosen_kind':chosen['kind'],'selected_Q16':chosen['Q16'],'selected_B15':chosen['B15'],
                      'two_eta_oracle_B15':max(r['B15'] for r in rr),'true_ordering_correct':int(chosen['Q16']==max(r['Q16'] for r in rr))})
    csvwrite(OUT/'frozen_two_eta_ordering.csv',pairs)
    high_false=sum(r['critic_probability']>=.9 and r['Q16']<=.5 for r in bad)
    result={'new_continuations':320,'cached_parent_continuations_reused':160,'postflight':post['summary'],
        'new_source_families':0,'diagnostic_parent_families':5,'new_perturbed_states':10,
        'no_training':True,'no_new_eta':True,'local_axis_only':'agent0 initial x +/-1cm',
        'new_bad_B15':sum(r['B15'] for r in bad),'new_good_B15':sum(r['B15'] for r in good),
        'bad_mean_Q16':float(np.mean([r['Q16'] for r in bad])),
        'good_mean_Q16':float(np.mean([r['Q16'] for r in good])),
        'bad_mean_frozen_prediction':float(np.mean([r['critic_probability'] for r in bad])),
        'good_mean_frozen_prediction':float(np.mean([r['critic_probability'] for r in good])),
        'high_confidence_low_Q_bad_cells':high_false,
        'two_eta_critic_selected_B15':sum(r['selected_B15'] for r in pairs),
        'two_eta_oracle_B15':sum(r['two_eta_oracle_B15'] for r in pairs),
        'new_outcomes':dict(Counter(r['outcome'] for r in all_new)),
        'state_dependence_variation_max_bad_Qrange':max(r['bad_q_range'] for r in families),
        'interpretation':'Local frozen-critic error quantified; consult per-family table before attributing to a boundary',
        'data_vs_model':'UNDERRESOLVED_WITHOUT_LOCAL_SUPERVISION_INTERVENTION',
        'representation_aliasing':'NOT_ESTABLISHED_BY_NEARBY_PHYSICAL_STATES; different inputs can have different Q',
        'fresh_seed_confirmation':False,'seed_note':'same 16 continuation indices and parent RNG namespace, intentionally matched not fresh independent certification',
        'scope_warning':'five outcome-selected diagnostic families; not population success estimates; perturbations not independent source groups'}
    write(OUT/'result_summary.json',result)
    write(OUT/'working_state.json',{'stage':'COMPLETE_ROLLOUT_AND_DB_ANALYSIS','new_rollout':320,'postflight_exact':320,'slurm_job_id':1424})
    now=datetime.now(timezone.utc).isoformat()
    with connect() as con:
        con.execute('UPDATE experiment SET end_time=?,reused_rollout_count=?,new_rollout_count=? WHERE experiment_uid=?',
                    (now,160,320,protocol['experiment_uid']))
    with (OUT/'experiment_ledger.csv').open('a',newline='') as f:
        csv.writer(f).writerow([now,'complete_postflight_and_analysis',320,320,'all 320 exact reusable; 160 matched parent records reused; no training'])
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
