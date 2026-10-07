"""Finalize adjudication without changing any trained model or frozen result."""
import json,csv,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from .run import ROOT,OUT,FIRST,FOLDS,SCENES,load,sha,dump,rep
from .eval import write_csv

def main():
    assert sha(FIRST/'final_decision.json')==load(OUT/'protocol.json')['first_fold_decision_sha256']
    primary=[];seeds=[];cache=[]
    for fold in ('ring','toy','db','four'):
        folder=FIRST if fold=='ring' else OUT/fold;d=load(folder/'final_decision.json');m=d['models']['shared'];baseline=d['source_selected_eta_baseline'];b=d['models'][baseline];truth=load(folder/'target_truth.json');pair=d['paired'][baseline]
        r={'target':fold,'states':m['states'],'oracle_B15':m['oracle_b15'],'shared_B15':m['b15'],'shared_unresolved':m['unresolved'],'source_selected_eta_baseline':baseline,'eta_B15':b['b15'],'eta_unresolved':b['unresolved'],'eta_MLP_B15':d['models']['eta_only']['b15'],'shared_available_B15_selection_rate':m['b15']/m['oracle_b15'],'gap_lower':max(0,m['oracle_b15']-m['b15']-m['unresolved']),'gap_upper':m['oracle_b15']-m['b15'],'Q_lower':m.get('Q_lower',m.get('mean_selected_Q_lower')),'Q_upper':m.get('Q_upper',m.get('mean_selected_Q_upper')),'severe_false_positive':m['severe_false_positive'],'paired_rescue':pair['rescue'],'paired_break':pair['break'],'paired_95CI':json.dumps(pair['paired_95CI']),'paired_p':pair.get('exact_p',pair.get('exact_two_sided_p')),'candidate_B15':sum(v is True for t in truth for v in t['robust']),'candidate_nonB15':sum(v is False for t in truth for v in t['robust']),'candidate_unresolved':sum(v is None for t in truth for v in t['robust'])}
        r['adjudication']='FAILED_ZERO_SHOT' if fold=='ring' else 'NO_SUPPORTED_STATE_AWARE_GAIN'
        if fold=='four':r['adjudication']='FOLD_UNDERDISCRIMINATIVE_WITH_NUMERICAL_UNRESOLVED'
        if fold=='db':r['adjudication']='NO_GAIN_LOW_DISCRIMINABILITY'
        primary.append(r)
        for kind in ('shared','eta_only'):
            for seed in (17,23,41):
                z=d['models'][f'{kind}_seed{seed}'];seeds.append({'target':fold,'method':kind,'seed':seed,'B15':z['b15'],'unresolved':z['unresolved']})
        cache.append({'target':fold,**load(folder/'cache_preflight.json')['summary']})
    write_csv(OUT/'loso_summary.csv',primary);write_csv(OUT/'loso_seed_stability.csv',seeds);write_csv(OUT/'cache_summary.csv',cache)
    cap=list(csv.DictReader((OUT/'joint_capacity_diagnostic.csv').open()));state=list(csv.DictReader((OUT/'state_dependence_diagnostic.csv').open()));local=list(csv.DictReader((OUT/'local_control.csv').open()))
    capacity={}
    for fold in ('toy','db','four','ring'):
        rr={r['method']:r for r in cap if r['fold']==fold};sh=[int(r['b15']) for r in state if r['fold']==fold and r['method'].startswith('shuffled')]
        capacity[fold]={'generic_joint':rr['joint_shared'],'enriched_joint':rr['joint_enriched_shared'],'matched_scene_only':rr['single_'+fold+'_shared'],'enriched_seed_B15':[int(rr['joint_enriched_shared_seed'+str(s)]['b15']) for s in (17,23,41)],'shuffled_h_B15_mean':float(np.mean(sh)),'shuffled_h_B15_range':[min(sh),max(sh)],'scene_supervised_eta_only':next(r for r in state if r['fold']==fold and r['method']=='target_supervised_eta_only')}
    # Integrity checks beyond state UID: no identical physical t0 initial state
    # crosses source TRAIN/VAL or target confirmation. Include current Flow in
    # encoded-input identity separately; do not equate approximate neighbors.
    states=load(OUT/'joint_enriched/source_states.json');seen=defaultdict(list);encoded=defaultdict(list)
    def physical_key(s):
        return hashlib.sha256(json.dumps({k:s[k] for k in ('positions','velocities','goals')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for s in states:
        seen[(s['scenario'],physical_key(s['physical']))].append((s['split'],s['state_uid']))
        x=rep.entities(s['physical']);h=hashlib.sha256()
        for k in sorted(x):h.update(k.encode());h.update(str(x[k].shape).encode());h.update(x[k].tobytes())
        encoded[h.hexdigest()].append((s['scenario'],s['state_uid']))
    crosssplit=[v for v in seen.values() if len({z[0] for z in v})>1];crossscene=[v for v in encoded.values() if len({z[0] for z in v})>1]
    assert not crosssplit
    integrity={'first_fold_unchanged':True,'exact_physical_train_val_overlap':len(crosssplit),'exact_cross_scene_encoded_alias_groups':len(crossscene),'interpretation':'No observed exact aliases; does not prove representation sufficiency for unseen controller response. Summary-neighbor mismatch is not an identifiability counterexample.','toy_flow_replay':load(OUT/'targets/toy/input_replay.json'),'db_reference_replay':load(OUT/'targets/db/input_replay.json'),'four_reference_replay':load(OUT/'targets/four/input_replay.json'),'structural_tests':[load(OUT/f/'structural_tests.json') for f in FOLDS]}
    dump('integrity_audit.json',integrity)
    hypotheses=[
      {'hypothesis':'SUPPORT_FAILURE / outcome-dependent label ascertainment','confidence':'HIGH for observed mismatch; MODERATE causal attribution across folds','support':'13,356 ->12 Four and12,997 ->23 Ring known nonB15 TRAIN pairs survive full-count filter. Ring/Four original data contain no Q<=.5. All target physical supports lie beyond source leave-family p95. Same-architecture enriched joint Ring11->58; seeds56-58.','against':'Enrichment jointly changes state stage, eta distribution and negative coverage and uses target TRAIN labels, so cannot isolate each or establish source-only repair.'},
      {'hypothesis':'LEARNING_FAILURE / non-invariant scene-conditioned extrapolation','confidence':'HIGH for current models, not intrinsic impossibility','support':'Source-held-out Toy state-aware NLL .236-.270 vs eta-only .663-.717, yet zero-shot target Toy30/48 and Ring0/60. Generic joint Ring11 vs matched Ring-only47; enriched joint repairs this. Ring eta ordering has wrong sign and59 severe false positives.','against':'Source-only local model also Ring1/60; not uniquely a neural optimizer defect. No evidence that bigger capacity would solve it.'},
      {'hypothesis':'scene/global eta preference can explain strong target-supervised results','confidence':'HIGH on these fixed pools','support':'Ring enriched joint58/60; shuffling same-scene h gives mean57.4/60(range56-59); Ring-only eta-only53/60. DB supervised eta-only24/24 and shuffled24/24. Toy shuffle44->33.9 supports Toy state dependence.','against':'Does not prove h is globally unnecessary; proposal pools may suppress state-dependent ranking-reversal evidence.'},
      {'hypothesis':'IDENTIFIABILITY_FAILURE / omitted base-controller or safety-response context','confidence':'UNDERRESOLVED','support':'Representation includes one current Flow reference, not future Flow response function or full active-set response; geometry and base controller are confounded by scene.','against':'No observed identical cross-scene encoded-input conflict. Same representation learns enriched supervised data. No matched-physical-state/base-controller intervention available.'},
      {'hypothesis':'INVARIANCE_FAILURE of the underlying physical mapping','confidence':'UNDERRESOLVED','support':'Zero-shot eta ordering changes across scenes; only three geometric/controller regimes in source.','against':'Wrong learned extrapolation does not establish non-existence of a shared physical law.'},
      {'hypothesis':'implementation/preprocessing bug','confidence':'NO IDENTIFIED CAUSAL BUG in tested paths','support':'Historical full-count admission is a data-policy problem, not a numeric corruption.','against':'CPU/GPU/mask/passive-permutation and basis equivalence tests pass; DB/Four archived-reference replays agree <=2e-4; Toy archived Flow agreement7.42e-9; first fold immutable.'}
    ]
    dump('hypothesis_status.json',hypotheses)
    result={'status':'COMPLETE_LOSO_NO_VALIDATED_ZERO_SHOT_STATE_ETA_CRITIC','four_folds':primary,'capacity_diagnostic':capacity,'root_causes':hypotheses,'unified_representation':'KEEP','representation_scope':'retains demonstrated supervised capacity; sufficiency/invariance across unobserved controllers remains UNDERRESOLVED','generator_modified':False,'generator_gate_passed':False,'zero_shot_repaired':False,'recommended_retained_model':'joint_enriched/shared/seed17 for target-supervised diagnostic only; no new zero-shot deployment recommendation','new_rollouts':0,'cache':cache,'source_only_scope':'All target scene labels excluded from source fitting, normalization and sourceVAL selection. Inherited schema historically all-scene-informed; fixed proposal generators are not whole-pipeline zero-shot.','confirmation_scope':'Existing independent true-t0 pools, not newly untouched after prior public reports; post-hoc diagnostics explicitly separated.','remaining_minimal_evidence':['Source-only fixed-budget or stopping-rule-aware negative supervision under a preregistered acquisition protocol, tested on an independent confirmation pool.','Matched geometry/state with controlled base-controller/safety-response context to discriminate missing input from lack of support; do not claim invariance failure before this.'],'stop_reason':'All fourfold results and discriminating joint/data/local/shuffle controls complete; no evidence justifies architecture search or generator changes.'}
    dump('final_decision.json',result)
    report=['Success Basin C1: complete LOSO and root-cause adjudication','',
      'Conclusion: No reliable leave-one-scene-out state-aware advantage has been established. The strongest evidence is supervision ascertainment + source physical support failure, followed by harmful learned extrapolation. Neither intrinsic non-identifiability nor impossible shared dynamics has been proved.','',
      'FROZEN LOSO (all methods use identical stochastic K16; mean excluded)',
      'Target | Oracle | shared | source-VAL-selected eta-only | shared seeds17/23/41 | verdict']
    for r in primary:
        sr=[z for z in seeds if z['target']==r['target'] and z['method']=='shared']
        report.append(f"{r['target']} | {r['oracle_B15']}/{r['states']} | {r['shared_B15']} + {r['shared_unresolved']} unresolved | {r['eta_B15']} + {r['eta_unresolved']} unresolved ({r['source_selected_eta_baseline']}) | "+', '.join(f"{z['B15']}+{z['unresolved']}u" for z in sr)+f" | {r['adjudication']}")
    report += ['', 'Source eta-only selection compares continuous MLP and kernel using source VAL, never chooses the best target score. MLP-only B15 is Ring17, Toy32, DB23, Four22. Numerical unresolved is neither success nor failure; Four shared has zero confirmed nonB15 selections, so its gap is 0..3, not three proven errors. DB has350/384 robust candidates and weak adaptive discrimination; its shared20/24 is still not evidence of success.',
      'Paired state-aware vs source-selected eta control: Ring0 rescue/17 break; Toy8/10; DB1/4; Four1/0 among19 resolved pairs. Source-only kernel remains reported even where source selection chose MLP (Toy kernel37/48). No positive claim survives paired uncertainty.', '',
      'CAPACITY / DATA INTERVENTION (supervised, NOT zero-shot)',
      'Target | same-input scene-only | generic joint | enriched joint | enriched seed range']
    for f,c in capacity.items():
        def fmt(r):return f"{r['b15']}+{r['unresolved']}u"
        report.append(f"{f} | {fmt(c['matched_scene_only'])} | {fmt(c['generic_joint'])} | {fmt(c['enriched_joint'])} | {c['enriched_seed_B15']}")
    report += ['', 'Enrichment reused5332 exact full-Q16 TRAIN/VAL pairs from existing proposal-aligned caches; no confirmation states or outcomes entered training. Same physical encoder, critic, W1, learning rate, budget, seeds and eta normalization; four-scene minibatches remain equally weighted. Dataset enlargement changes sampled minibatch identities/order; this is not a matched-edge proof isolating only boundary vs true-t0 vs proposal support. Generic and enriched selected seed17 both trained4000 steps, so Ring11->58 is not explained by extra optimization duration alone.',
      'The enriched model is NOT a repaired zero-shot model and cannot use target-derived proposal acquisition to claim source-only transfer. Ring-only historical B15-classifier reference53/60 and historical joint56/60 remain unchanged. Our matched physical scene-only Ring reference47/60 uses far less failure supervision and is not the strongest historical upper bound.', '',
      'STATE DEPENDENCE CHECK',
      'Ring enriched58/60 becomes56..59 (mean57.4) after ten fixed within-Ring h derangements. Ring-specific eta-only53/60. DB stays24/24 after every shuffle; DB-specific eta-only24/24. Toy enriched44/48 drops to30..37 (mean33.9), consistent with real Toy state dependence. Thus strong joint results establish scene-conditioned fitting capacity, NOT general transferable state-eta interaction. Exact cross-scene source-VAL reversal cases with two shared eta and margin>=.25: zero eligible cases; do not manufacture reversals from near-but-distinct eta.', '',
      'ROOT CAUSES, EVIDENCE ORDER',
      '1. Data/support: Four known nonB15 TRAIN13356->12; Ring12997->23 after full>=16 filter. Original Ring/Four no Q<=.5. Early stop after two failures is informative negative evidence, not an unbiased fixed-Q16 fraction. Dropping all such evidence induces survivorship selection. Enriched exact Q16 Ring failure pairs rise0->329 and restore supervised success. Source physical coverage is also disjoint: target/source-neighbor p95 thresholds exceeded for all targets; Ring has new curvature and policy-frame applicability. Distance is descriptive, not proof of causality.',
      '2. Learned extrapolation/negative transfer: source-held-out state-aware fitting is useful in Toy/DB, yet fails outside source physical regimes. Generic joint Ring11 versus matched Ring-only47 is direct negative-transfer evidence under limited supervision. Local source-only kNN is Toy39, DB15, Four22, Ring1; it does not rescue Ring and shows the failure is not exclusively a neural fitting pathology.',
      '3. Current successful target-supervised scores often rely on scene/global eta preference; shuffled-state and eta-only controls prevent interpreting58/60 as proof of state-conditioned cross-scene learning.',
      'Not established: information-theoretic representation insufficiency, base-controller invariance failure, or insufficient network capacity. Same representation can learn supervised regimes; exact cross-scene input aliases were not found. We do not have matched-state/different-controller intervention data to settle missing future response context.', '',
      'DECISION TREE',
      'Strict LOSO state-aware > source-only eta-only? NO credible fold. Ring fails clearly; Toy/DB no supported gain; Four is weakly discriminative and numerical-limited.',
      'Joint target-supervised fit possible without new architecture? YES, especially Ring after support enrichment; Four numerical status prevents a blanket four-scene near-oracle claim.',
      'Does this prove transferable state-eta interaction? NO; Ring/DB shuffle and eta-only controls remain strong.',
      'Representation: KEEP as audited shared supervised baseline; do not REDESIGN or declare intrinsically sufficient for unseen controller regimes. Missing controller-response information remains UNDERRESOLVED.',
      'Generator: unchanged; current candidate oracle is already perfect for these four confirmation panels. No generator zero-shot claim or new generator training.',
      'Next minimum: source-only acquisition/training that retains valid negative evidence and independently confirms a repaired model; a matched geometry/controller intervention only if needed to distinguish support from missing response context. Stop here rather than tune on these test pools.', '',
      'INTEGRITY / COST / REPRODUCTION',
      'NEW ROLLOUT=0. All labels queried read-only from global DB; cached numerical unknowns remain unknown.39936 requested evaluation seed slots,39745 valid cached outcomes,191 numerical/uncertified slots, no reruns. No generator, control, safety, horizon, mode ID or success-definition change.',
      'Training: Slurm1726 and1727, each1GPU shard,2CPU. Seeds17/23/41. First Ring fold hash retained. Canonical data keys, source splits and normalization under each fold/source_pairs.parquet, source_states.json, normalization.json. Cached training expansion admission in enrichment_audit.json.',
      'Python=/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python; cwd=/home/zhihan/research/Basin_C1; PYTHONPATH=cwd; JAX_PLATFORMS=cpu for read-only audits. Entrypoints: run.py prepare/train (existing freezes protected), train.sbatch, eval.py inputs/predict/evaluate, diagnostics.py prepare/evaluate/state_dependence, audit.py support/ascertainment/source_fit/local_control, finalize.py. Training uses source TRAIN/VAL exclusively; diagnostic outputs never overwrite strict results.',
      'Full tables: loso_summary.csv, loso_seed_stability.csv, source_fit_metrics.csv, joint_capacity_diagnostic.csv, state_dependence_diagnostic.csv, label_ascertainment.csv, support_by_fold.csv, local_control.csv, integrity_audit.json, hypothesis_status.json.']
    (OUT/'final_report.txt').write_text('\n'.join(report)+'\n')
    dump('working_state.json',{'stage':'complete','strict_loso_folds_complete':4,'posthoc_diagnostics_complete':True,'new_rollouts':0,'generator_unchanged':True,'pending_jobs':[]})
    ledger=[{'stage':'strict_loso','status':'complete','jobs':'1726','new_rollouts':0,'labels':'source-only'}, {'stage':'joint_capacity_and_scene_references','status':'complete','jobs':'1727','new_rollouts':0,'labels':'all-scene TRAIN/VAL; NOT zero-shot'}, {'stage':'support_ascertainment_local_shuffle_audits','status':'complete','jobs':'CPU','new_rollouts':0,'labels':'post-hoc analysis; no retraining from confirmation outcomes'}]
    write_csv(OUT/'experiment_ledger.csv',ledger)
    tracked=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.parquet','.msgpack','.npz','.csv','.txt','.sbatch') and p.name!='artifact_hashes.json']
    dump('artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(tracked)})
    print(json.dumps({'status':result['status'],'representation':'KEEP','new_rollout':0,'report':str(OUT/'final_report.txt')},indent=2))

if __name__=='__main__':main()
