"""Read-only, seed-verified partial-count supervision; never imputes trials."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,csv,hashlib,sqlite3
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
OLD=ROOT/'diagnostics/orthoflow3_loso_root_cause_v1';FIRST=ROOT/'diagnostics/orthoflow3_cross_scene_zero_shot_v1'
SCENES=('toy_giveway','double_bottleneck','four_way_intersection','ring_exchange')
FOLDS=dict(zip(('toy','db','four','ring'),SCENES))
VARIANTS=('full_continuation','partial_count','negative_binary')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def con():
    c=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row;return c
def baseline(fold):return FIRST if fold=='ring' else OLD/fold

def counts(rows):
    # B15 is a certificate about the frozen first sixteen seeds, not pooled Q64.
    seen={json.loads(r['seed_key']).get('future_index'):r for r in rows};first=[seen[i] for i in range(16) if i in seen]
    s=sum(r['success'] for r in rows);n=len(rows);s16=sum(r['success'] for r in first);f16=len(first)-s16
    return dict(s=s,f=n-s,n=n,observed_rate=s/n,standard_observed=len(first),standard_success=s16,standard_failure=f16,full_standard_16=len(first)==16,b15_confirmed=s16>=15,non_b15_confirmed=f16>=2,
                Q16=s16/16 if len(first)==16 else None,Q16_lower=s16/16,Q16_upper=(s16+16-len(first))/16,
                observed_rate_le_half=s/n<=.5,strong_failure_Q16_upper_le_half=(s16+16-len(first))/16<=.5)

def stopping_test():
    # Exhaustive stopped-path likelihood test; stops at second failure or 15 successes.
    paths=Counter()
    def visit(s,f):
        if f>=2 or s>=15 or s+f==16:paths[(s,f)]+=1;return
        visit(s+1,f);visit(s,f+1)
    visit(0,0);out=[]
    for p in (.01,.1,.5,.9,.99):
        mass=sum(v*p**s*(1-p)**f for (s,f),v in paths.items())
        grad=sum(v*p**s*(1-p)**f*((s+f)*p-s) for (s,f),v in paths.items())
        equal_pair_grad=sum(v*p**s*(1-p)**f*(p-s/(s+f)) for (s,f),v in paths.items())
        assert abs(mass-1)<1e-12 and abs(grad)<1e-12
        out.append({'p':p,'stopped_path_probability_sum':mass,'expected_partial_NLL_logit_gradient':grad,'naive_pair_equal_rate_gradient':equal_pair_grad})
    dump('likelihood_unit_tests.json',{'passed':True,'cases':out,'assumption':'conditional Bernoulli continuations under fixed state/controller; known deterministic stopping rule contributes only parameter-independent path indicator. Proposal acquisition and source-support bias are not removed by this likelihood.'})

def prepare():
    assert not (OUT/'protocol.json').exists(),'Already frozen; resume, do not redesign'
    protocol={'architecture':rep.VERSION,'architecture_sha256':sha(rep.__file__),'seeds':[17,23,41],
      'primary':'partial_count','controls':['frozen OLD_FULL_ONLY W1 reused','full_continuation: same old pairs, trial-equivalent weighting','negative_binary diagnostic'],
      'loss_partial':'mean_scenes mean_uniform_pairs (s*softplus(-logit)+f*softplus(logit))/mean_observed_n_in_scene_TRAIN',
      'scene_balance':'96 uniform canonical pairs per source scene per step; deterministic scene-specific total-trial normalization, no per-minibatch random denominator',
      'negative_binary':'full-count likelihood + certified early nonB15 event -log(1-P(B15|p)); P(B15|p)=p^16+16p^15(1-p). Each binary event counts as one observation; partial positives excluded only in this diagnostic. Not fake Q=0.',
      'validation':'source-only mean scene observed-trial NLL, same continuation scale; binary control uses its matching event likelihood; checkpoint/seed selected only by this criterion',
      'optimizer':'unchanged AdamW lr.001 wd.0001 clip5; max4000, eval100, stale10 after1500',
      'normalization':'reuse each original fold source-only normalization for all variants; no target stats; unchanged physical units',
      'target':'unchanged four independent true-t0 stochastic K16 pools; all predictions frozen before outcomes opened in this experiment',
      'source_pool':'old full pairs union audited-v2 train/val early-stop records only; no all-scene-generator acquired extra proposals',
      'new_rollout':0,'generator_changed':False,'target_labels_used_for_training':False,
      'strictness':'label-free target fit/selection under inherited fixed schema; historical target-informed schema and candidate generator ancestry remain disclosed; test pools reused for preregistered comparison, not new untouched confirmation'}
    dump('protocol.json',protocol);stopping_test()
    states=load(OLD/'all_source_states.json');stateby={s['state_uid']:s for s in states}
    oldrows=pq.read_table(OLD/'all_source_pairs.parquet').to_pylist();c=con();controllers={};excluded=[];records={};old_keys=set()
    def verify(rr,key):
        valid=[]
        for r in rr:
            z=c.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            if z is None or (z['state_uid'],z['eta_uid'],z['controller_uid'])!=key:return None,'key_mismatch'
            if z['compatibility_quality']!='EXACT_REUSE' or z['conflict_quarantined'] or z['numerical_failure']:return None,'incompatible_or_numerical'
            if 'seed_key' in r and z['seed_key']!=r['seed_key']:return None,'seed_mismatch'
            if 'success' in r and bool(z['success'])!=bool(r['success']):return None,'outcome_mismatch'
            valid.append(dict(z))
        if len({r['seed_key'] for r in valid})!=len(valid):return None,'duplicate_seed'
        if key[2] not in controllers:
            ct=dict(c.execute('SELECT * FROM controller_config WHERE controller_uid=?',(key[2],)).fetchone())
            if ct['compatibility_quality'] not in ('EXACT_REUSE','EXACT_PROFILE'):return None,'ambiguous_controller'
            controllers[key[2]]=ct
        return valid,None
    for r in oldrows:
        key=(r['state_uid'],r['eta_uid'],r['controller_uid']);rr=[{'rollout_uid':u,'seed_key':s} for u,s in zip(r['rollout_uids'],r['seed_keys'])];valid,err=verify(rr,key)
        assert not err,(key,err);co=counts(valid);assert co['s']==r['k'] and co['n']==r['n']
        records[key]={**r,**co,'old_full_member':True,'partial_record':False,'stopping_reason':'OLD_FROZEN_COMPLETE_BUDGET','q_label_exact_budget':True,'rollout_uids':[z['rollout_uid'] for z in valid],'seed_keys':[z['seed_key'] for z in valid]};old_keys.add(key)
    dataset=ROOT/'datasets/orthoflow3_basin_dataset_v2_audited'
    native={s['state_uid']:s for s in pq.read_table(dataset/'states.parquet').to_pylist()}
    labels=pq.read_table(dataset/'eta_labels.parquet').to_pylist();expected=Counter();audit=Counter()
    for r in labels:
        if r['split'] not in ('train','validation'):continue
        if r['robust_15of16'] is False:expected[(r['scenario'],r['split'])]+=1
        key=(r['state_uid'],r['eta_uid'],r['controller_uid'])
        if key in records:continue
        if r['seed_count']>=16:
            excluded.append({'state_uid':key[0],'eta_uid':key[1],'reason':'not_in_old_full_pool','scenario':r['scenario']});continue
        if r['numerical_failure_count'] or r['early_stop_reason'] not in ('ROBUST_IMPOSSIBLE_2_FAILURES','ROBUST_CONFIRMED_15_SUCCESSES'):
            excluded.append({'state_uid':key[0],'eta_uid':key[1],'reason':'numerical_or_unresolved_stopping','scenario':r['scenario']});continue
        if r['scenario']=='ring_exchange':assert r['label_semantics_version']=='ring_current_safety_v2'
        valid,err=verify(rep.decode(r['seed_outcomes']),key)
        if err:excluded.append({'state_uid':key[0],'eta_uid':key[1],'reason':err,'scenario':r['scenario']});continue
        co=counts(valid);assert co['n']==r['seed_count'] and co['s']==r['success_count'] and co['f']==r['failure_count']
        assert co['standard_observed']==co['n']
        indices=sorted(json.loads(z['seed_key'])['future_index'] for z in valid)
        assert indices==list(range(co['n'])),(key,indices)
        assert co['non_b15_confirmed'] if r['early_stop_reason']=='ROBUST_IMPOSSIBLE_2_FAILURES' else co['b15_confirmed']
        # Ensure the historical sequence did not continue beyond its stopping event.
        seq=sorted(valid,key=lambda z:json.loads(z['seed_key'])['future_index']);s=f=0
        for z in seq[:-1]:
            s+=z['success'];f+=1-z['success'];assert f<2 and s<15,(key,'not first stopping time')
        nr=native[key[0]]
        if key[0] not in stateby:stateby[key[0]]={'state_uid':key[0],'scenario':r['scenario'],'split':nr['split'],'family':nr['parent_episode_id'],'physical':rep.parse(nr)}
        assert nr['split']==r['split']
        records[key]={'state_uid':key[0],'eta_uid':key[1],'controller_uid':key[2],'eta':rep.decode(r['eta_raw']),'scenario':r['scenario'],'split':r['split'],'origin':'v2_verified_early_stop',**co,'old_full_member':False,'partial_record':True,'stopping_reason':r['early_stop_reason'],'q_label_exact_budget':False,'rollout_uids':[z['rollout_uid'] for z in valid],'seed_keys':[z['seed_key'] for z in valid],'label_semantics_version':r['label_semantics_version']}
        audit[r['early_stop_reason']]+=1
    c.close();rows=list(records.values());states=list(stateby.values());si={s['state_uid']:i for i,s in enumerate(states)}
    for r in rows:r['state_index']=si[r['state_uid']]
    for sc in SCENES:
        tr={s['family'] for s in states if s['scenario']==sc and s['split']=='train'};va={s['family'] for s in states if s['scenario']==sc and s['split']=='validation'};assert not tr&va
    targetids=set()
    for fold in FOLDS:targetids.update(r['state_uid'] for r in load(OLD/'targets'/fold/'manifest.json'))
    assert not targetids&set(stateby)
    pq.write_table(pa.Table.from_pylist(rows),OUT/'all_pairs.parquet');dump('states.json',states);np.savez_compressed(OUT/'entities.npz',**rep.batch([rep.entities(s['physical']) for s in states]));dump('controller_profiles.json',controllers)
    stats=[]
    for sc in SCENES:
        for sp in ('train','validation'):
            for mode in ('old_full','repaired'):
                rr=[r for r in rows if r['scenario']==sc and r['split']==sp and (mode=='repaired' or r['old_full_member'])];non=sum(r['non_b15_confirmed'] for r in rr)
                stats.append({'scenario':sc,'split':sp,'dataset':mode,'states':len({r['state_uid'] for r in rr}),'families':len({stateby[r['state_uid']]['family'] for r in rr}),'pairs':len(rr),'old_full_pairs':sum(r['old_full_member'] for r in rr),'early_stop_partial_pairs':sum(r['partial_record'] for r in rr),'full_standard16_pairs':sum(r['full_standard_16'] for r in rr),'B15_confirmed':sum(r['b15_confirmed'] for r in rr),'nonB15_confirmed':non,'observed_rate_le_half_not_Q16':sum(r['observed_rate_le_half'] for r in rr),'full_Q16_le_half':sum(r['Q16'] is not None and r['Q16']<=.5 for r in rr),'partial_Q16_upper_le_half':sum(r['partial_record'] and r['strong_failure_Q16_upper_le_half'] for r in rr),'observed_continuations':sum(r['n'] for r in rr),'unique_eta':len({r['eta_uid'] for r in rr}),'negative_retention_denominator':expected.get((sc,sp)),'negative_retention':non/expected[(sc,sp)] if expected[(sc,sp)] else None})
    csvout('dataset_audit.csv',stats)
    if excluded:csvout('excluded_records.csv',excluded)
    csvout('partial_count_semantics.csv',[{k:r.get(k) for k in ('state_uid','eta_uid','controller_uid','scenario','split','s','f','n','full_standard_16','stopping_reason','b15_confirmed','non_b15_confirmed','Q16','Q16_lower','Q16_upper','q_label_exact_budget')} for r in rows])
    for fold,target in FOLDS.items():
        sources=[s for s in SCENES if s!=target];d=OUT/fold;d.mkdir(exist_ok=True)
        dump(f'{fold}/normalization.json',load(baseline(fold)/'normalization.json'))
        for variant in VARIANTS:
            rr=[i for i,r in enumerate(rows) if r['scenario'] in sources and (variant!='full_continuation' or r['old_full_member']) and (variant!='negative_binary' or r['old_full_member'] or r['non_b15_confirmed'])]
            dump(f'{fold}/{variant}/dataset_manifest.json',{'row_indices':rr,'sources':sources,'target_scene':target,'pair_table_sha256':sha(OUT/'all_pairs.parquet'),'eta_norm_sha256':sha(d/'normalization.json'),'source_groups':{sc:{sp:sorted({str(stateby[rows[i]['state_uid']]['family']) for i in rr if rows[i]['scenario']==sc and rows[i]['split']==sp}) for sp in ('train','validation')} for sc in sources},'target_labels_used':False})
    dump('build_audit.json',{'verified_pairs':len(rows),'partial_added':len(rows)-len(old_keys),'stop_reasons':dict(audit),'excluded_reasons':dict(Counter(r['reason'] for r in excluded)),'state_family_overlap':0,'confirmation_state_overlap':0,'new_rollouts':0,'ring_safety':'current_v2_only','old_artifact_hashes':{f:sha(baseline(f)/'final_decision.json') for f in FOLDS}})
    dump('working_state.json',{'stage':'dataset_frozen','new_rollouts':0});print(json.dumps(load(OUT/'build_audit.json'),indent=2))

if __name__=='__main__':prepare()
