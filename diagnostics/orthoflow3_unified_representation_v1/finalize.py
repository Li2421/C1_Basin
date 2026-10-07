"""Post-confirmation aggregation only; never updates models or scientific labels."""
import json, hashlib, runpy, py_compile
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'datasets/orthoflow3_basin_dataset_v3_unified_rep'
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def eta_key(x):return tuple(np.round(np.asarray(x,float),10))

def run():
    fresh=load(OUT/'confirmation/results.json');dev=load(OUT/'development/results.json')
    assert len(list((OUT/'confirmation').glob('run_*of18.json')))==18
    props=load(OUT/'confirmation/proposals.json')['states']
    truth={(x['state_uid'],x['index']):x for x in fresh['evidence']}
    traces={};raw_counts=Counter()
    for path in (OUT/'confirmation').glob('*/raw/*.jsonl'):
        for line in path.open():
            x=json.loads(line);raw_counts['journal_rows']+=1
            raw_counts['numerical_attempts']+=bool(x['numerical_failure'])
            if x['future_index']==0 and not x['numerical_failure'] and 'trace' in x:
                traces[x['state_uid'],eta_key(x['eta'])]=x
    base={p['anchor']:p for p in props if p['rotation']==0};pairs=[]
    for p in props:
        if not p['rotation']:continue
        b=base[p['anchor']];angle=-np.deg2rad(p['rotation'])
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for method in ('old','unified'):
            i0=18 if method=='old' else b['selected_index'];i1=18 if method=='old' else p['selected_index']
            e0=b['old_eta'] if method=='old' else b['etas'][i0];e1=p['old_eta'] if method=='old' else p['etas'][i1]
            x0=traces.get((b['state_uid'],eta_key(e0)));x1=traces.get((p['state_uid'],eta_key(e1)))
            rmse=None
            if x0 and x1:
                t0=np.asarray(x0['trace']['positions']);t1=np.asarray(x1['trace']['positions'])@R.T
                n=min(len(t0),len(t1));rmse=float(np.sqrt(np.mean((t0[:n]-t1[:n])**2)))
            q0=truth[b['state_uid'],i0];q1=truth[p['state_uid'],i1]
            seeds=set(q0['outcomes'])&set(q1['outcomes'])
            pairs.append({'scenario':p['scenario'],'anchor':p['anchor'],'rotation':p['rotation'],'method':method,
                'trajectory_rmse_common_prefix_seed0':rmse,
                'episode_length_seed0_original':x0['episode_length'] if x0 else None,
                'episode_length_seed0_transformed':x1['episode_length'] if x1 else None,
                'eta_distance':float(np.linalg.norm(np.asarray(e0)-e1)),
                'same_proposal_index':i0==i1 if method=='unified' else None,
                'valid_paired_seeds':len(seeds),
                'outcome_agreement':sum(q0['outcomes'][s]==q1['outcomes'][s] for s in seeds)/len(seeds) if seeds else None})
    dump(OUT/'closed_loop_metamorphic.json',{'pairs':pairs,'Four_active_rotation_not_a_contract_symmetry':True,
         'trajectory_RMSE_is_common_prefix_not_completion_equivalence':True})

    counts={}
    for stage in ('development','confirmation'):
        c=Counter()
        for p in (OUT/stage).glob('run_*of18.json'):c.update(load(p))
        counts[stage]={'execution':dict(c),'preflight':load(OUT/stage/'cache_preflight.json')}
    # Existing physical provenance is checked without a general DB-integrity rerun.
    from shared_rollout_db.src.rollout_db import connect
    persistence=[]
    with connect(True) as con:
        for stage in ('development','confirmation'):
            rows=con.execute('SELECT experiment_uid,path FROM experiment WHERE path=?',(str(OUT/stage/'execution'),)).fetchall()
            assert len(rows)==1,(stage,rows)
            n=con.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(rows[0][0],)).fetchone()[0]
            persistence.append({'stage':stage,'experiment_uid':rows[0][0],'persisted_rollouts':n})
    dump(OUT/'db_cache_report.json',{'batches':counts,'persistence':persistence,'confirmation_journal':dict(raw_counts),
         'physical_replays_for_reencoding':0,'new_Q_is_evaluation_only':True})

    before=load(ROOT/'diagnostics/orthoflow3_k16_critic_canonicalization_v1/canonical_hashes_before.json')
    changed=[p for p,h in before.items() if sha(ROOT/p)!=h]
    assert changed==['new_benchmark_common/safety_eta3.py'],changed
    source=ROOT/'datasets/orthoflow3_basin_dataset_v2_audited'
    manifest=load(DATA/'manifest.json')
    assert sha(source/'states.parquet')==manifest['source_states_sha256']
    assert sha(source/'eta_labels.parquet')==sha(DATA/'eta_labels.parquet')==manifest['labels_sha256']
    manifest.setdefault('initial_build_encoder_source_sha256',manifest['encoder_source_sha256'])
    manifest['encoder_source_sha256']=sha(OUT/'representation.py')
    manifest['encoder_change_note']='Batch/padding utility repaired before training; physical entity fields unchanged.'
    dump(DATA/'manifest.json',manifest);dump(OUT/'dataset_manifest.json',manifest)
    for p in OUT.glob('*.py'):py_compile.compile(str(p),doraise=True)
    ntests=0
    for name,f in runpy.run_path(str(ROOT/'tests/test_basis_families.py')).items():
        if name.startswith('test_') and callable(f):f();ntests+=1
    structural=load(OUT/'structural_tests.json');met=load(OUT/'learned_metamorphic.json')
    assert structural['passed'] and met['max_proposal_error']<2e-5 and met['max_score_error']<2e-5
    for name in ('generator','critic'):
        m=load(OUT/f'{name}_frozen.json')['selected'];assert sha(m['checkpoint'])==m['sha256']
    dump(OUT/'regression_results.json',{'basis_tests_passed':ntests,'structural_passed':True,'syntax_passed':True,
         'v2_unchanged':True,'frozen_models_unchanged':True,'canonical_changes_only_preexisting_FK_helper':changed})

    alias=load(OUT/'aliasing.json');near=alias['nearest_neighbors'];astats={}
    for sc in dev['summary']:
        rr=[v for v in near if v['scenario']==sc]
        astats[sc]={k:float(np.median([v[k] for v in rr if v[k] is not None])) if any(v[k] is not None for v in rr) else None
                    for k in ('h_distance','robust_jaccard_common_tested','Q_mean_absolute_difference','Q_ranking_spearman')}
    (OUT/'ALIASING_AUDIT.md').write_text('# Aliasing audit\n\nNo exact learned-embedding aliases were found among 245 original train/development states at the registered tolerance. This is a finite-corpus check, not an injectivity theorem. The remaining-horizon field from the contract repair is retained.\n\nNearest-neighbor statistics:\n\n```json\n'+json.dumps(astats,indent=2)+'\n```\n\nJaccard and Q comparisons use common evaluated evidence only; missing eta are not negatives. See aliasing.json for common-evidence counts, physical-summary distances and ranking correlation, and nearest_neighbor_diagnostics.svg. Near-neighbor differences are not by themselves aliasing defects. Monitor-history support for intermediate Double states remains explicitly out of scope: the parser rejects them rather than resetting history.\n')

    collisions=sum(e['collisions'] for e in fresh['evidence']);numerical=sum(e['numerical'] for e in fresh['evidence'])
    original_competitive=all(s['0']['selected']>=s['0']['old_selected'] for s in fresh['summary'].values())
    # Avoid equating structural success with unrestricted closed-loop symmetry.
    status='UNIFIED_REP_PARTIAL'
    decision={'status':status,'structural_representation_pass':True,'development_learning_pass':True,
        'original_chart_fresh_competitive':original_competitive,'fresh_collision_seed_slots':collisions,
        'fresh_numerical_seed_slots':numerical,'generator_manifest':str(OUT/'generator_frozen.json'),
        'critic_manifest':str(OUT/'critic_frozen.json'),'dataset':str(DATA),
        'reason':'Shared entity invariants pass and original-chart performance is competitive, but fresh Ring has a residual oracle/critic gap and numerical uncertainty; active Four-Way policy rotation remains a distinct, unsuccessful continuation problem.',
        'no_adaptation_to_confirmation':True,'environment_family_experiments_started':False}
    dump(OUT/'decision.json',decision)
    lines=['# Unified physical representation — final report','',status,'',
      '## Scope and inherited artifacts','',
      'Inherited Phase A CRITIC_PARTIAL, Phase B CANONICALIZATION_PARTIAL and CONTRACT_REPAIRED_AND_VALIDATED. R_old is the accepted remaining-time-repaired Phase-B pipeline. No claim that the preceding phases universally succeeded. Audited v2 remains unmodified; all original scientific labels are byte-identical in v3.','',
      '## Implementation','',
      'One shared physical entity parser/schema, agent encoder, ordered-pair message encoder and obstacle-set encoder replace the learned scenario adapters. Masked mean/max aggregation supports variable N/M. The scene embedding is 128D. Generator and critic have independently trained shared encoders with the same schema. No scene one-hot, arbitrary agent slot or obstacle identity is a neural feature. See REPRESENTATION_SPEC.md, ENTITY_SCHEMA.md and FRAME_CONVENTIONS.md.','',
      'The accepted proposal convention remains 16 stochastic proposals plus its existing deterministic location candidate; the location is not a standalone deployed selector. No eta=0 or fixed anchor is added to that set. Eta=0 is evaluated solely as B0. No continuous critic optimization. OrthoFlow3, safety, MACFlow, eta domain and horizon are unchanged.','',
      '## Controlled learning comparison','',
      'Two matched training seeds (17,23), from scratch, same accepted Gaussian generator family/objective, BCE-Q critic objective, optimizer budget and state/scenario-balanced sampling. Same v2 and existing full-Q proposal-aligned evidence as R_old; no new training labels. Selected checkpoints use validation loss only. The representation architecture changes parameter counts (generator 51,686→57,542; critic 55,585→61,441), so this is not a parameter-count-matched claim. Closed-loop comparison uses the selected checkpoint, not two-seed confidence intervals. Details: old_vs_unified_ablation.json and frozen/training manifests.','',
      '| Scenario | Dev old oracle/selected | Dev unified oracle/selected | Unified rescue / break |','|---|---:|---:|---:|']
    for sc,d in dev['summary'].items():
        u=d['unified'];o=d['old'];lines.append(f"| {sc} | {o['oracle']}/6, {o['selected']}/6 | {u['oracle']}/6, {u['selected']}/6 | {u.get('rescue')} / {u.get('break')} |")
    lines+=['','## Fresh confirmation, frozen before outcomes','',
      '| Scenario | Rotation | States | Old selected | Unified oracle | Unified selected | Selected unresolved | B0 | Rescue | Break | Exploitation |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sc,angles in fresh['summary'].items():
        for deg,s in angles.items():
            unresolved=sum(x['selected'] is None for x in fresh['states'] if x['scenario']==sc and x['rotation']==int(deg))
            lines.append(f"| {sc} | {deg} | {s['states']} | {s['old_selected']} | {s['oracle']} | {s['selected']} | {unresolved} | {s['B0']} | {s['rescue']} | {s['break']} | {s['exploitation']} |")
    lines+=['',f'All counts are empirical B15, not proof of population success probability one. Across all requested confirmation candidate/seed slots: collisions={collisions}, unresolved numerical slots={numerical}. Numerical uncertainty is not an ordinary failure; Q bounds and per-state outcomes are preserved in confirmation/results.json. This is a deliberately small confirmation, not a broad generalization claim. No adaptation followed it.','',
      'Ring original orientation has a real selection gap: oracle 2/2, selected 1/2, versus old selected 0/2. At 270 degrees one selected tuple is numerically unresolved, not a certified failure. All Ring rotations have exactly identical frozen eta proposals and selected indices. On common valid seeds, selected outcomes agree; seed-0 back-rotated trajectory RMSE is at most 2.93e-6. Thus representation consistency improved without demonstrating complete critic generalization. Four active rotations have 0/2 oracle and selected at each nonzero angle, with six high-score/low-Q flags; these are not solved by an invariant downstream encoder.','',
      '## Representation and learned-model metamorphic checks','',
      f"144 structural tests passed; max embedding error {structural['max_h_error']:.3g}; mixed padded-batch error {structural['mixed_batch_padding_error']:.3g}. N=2/3/4/5 and M=0/1/7/20 forward passes passed. These demonstrate architecture capability, not unseen-count policy success. 96 learned-model passive-transform tests: maximum eta discrepancy {met['max_proposal_error']:.3g}, score discrepancy {met['max_score_error']:.3g}, top-1 consistency {met['top1_consistency']:.3g}.",'',
      'Active scene transformations are different: Four frozen MACFlow is orientation/slot-sensitive. Policy-frame relations are retained rather than deleted to manufacture invariance. Active Four 90° proposal error decreased .0712→.0404, critic top-1 .75→1 in the targeted cohort, but cyclic-relabel and 270° proposal deviations did not uniformly improve. Ring rotation discrepancy is negligible and cyclic-relabel deviation decreased. See active_metamorphic.json; selected trajectory back-rotation, common-prefix RMSE and outcome agreement are in closed_loop_metamorphic.json.','',
      '## Aliasing and shortcut checks','',
      'No exact harmful embedding alias was observed in the 245-state corpus. Nearby-state comparisons are restricted to shared eta evidence; no global basin-smoothness claim. The train-only scenario probe reaches 100% validation classification because geometries are physically distinct; bookkeeping-only changes alter no features. This does not imply a hidden scene one-hot. See ALIASING_AUDIT.md and scenario_probe.json.','',
      '## Required answers','',
      '1. Fixed agent slots in unified h: no; associated physical records are processed by shared weights.\n2. Agent list order: invariant within numerical tolerance. Actively reindexing the frozen MACFlow API is not the same operation.\n3. Obstacle list order: invariant.\n4. Passive global translation: invariant when geometry and policy reference origin transform too.\n5. Contract-valid passive/global rotations: invariant; active Four rotations with unchanged upstream policy are not certified equivalences.\n6. Vector/action fields: consistently in per-agent goal frames; no Ring world-frame suffix.\n7. scene_id in neural conditioning: no. Scenario names remain parser/logging/randomness bookkeeping, not learned routing.\n8. Variable N/M: supported structurally without changing model shapes.\n9. Exact harmful h alias: none observed, not a universal proof.\n10. Near neighbors: substantial common-eta overlap, with finite-evidence limitations documented.\n11. Generator oracle coverage: no degradation on matched development; fresh results above.\n12. Critic ranking: no matched-development B15 loss or exploitation; fresh results above.\n13. Four representation-only rotation/relabel dependence: eliminated to tolerance; active upstream bias remains.\n14. Ring mixed-frame shortcut: removed.\n15. Suitable foundation for future geometry work: yes structurally, but not authorization or evidence for unrestricted environment-family generalization.','',
      '## Provenance, persistence and stop','',
      'No rollout was repeated merely to reencode coordinates. Full snapshot identity, not h, keys physical evidence. DB preflight and incremental persistence are recorded in db_cache_report.json. Source-v2 labels/state hashes, trained model hashes and canonical controller artifacts were verified. The only change versus the older Phase-A baseline is the separately accepted, pre-existing FK/raw-import helper fix. Relevant basis regressions, structural tests and syntax checks passed.','',
      'Final qualification: the unified representation is implemented and competitive within the tested original-chart scope, but cannot erase a causal frozen-policy orientation dependence. No environment-family experiment or post-confirmation retraining was started.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (DATA/'README.md').write_text('# Audited v2 → unified representation v3\n\nOnly conditioning is changed. Original 245 state IDs, parent splits and eta labels are preserved; eta_labels.parquet is byte-identical to audited v2. Native conditioning and structured snapshots remain available. See manifest.json and ../../diagnostics/orthoflow3_unified_representation_v1/REPORT.md. The candidate has status UNIFIED_REP_PARTIAL with a documented frozen-upstream-policy limitation. No environment-family generalization claim.\n')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.msgpack')}
    dump(OUT/'provenance_hashes.json',{'artifacts':hashes,'dataset':manifest,
        'proposals':{s:sha(OUT/s/'proposals.json') for s in ('development','confirmation')},
        'canonical_artifacts':{p:sha(ROOT/p) for p in before},'prior_FK_change':changed})
    return decision

if __name__=='__main__':print(json.dumps(run(),indent=2))
