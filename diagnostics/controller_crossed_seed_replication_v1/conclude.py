"""Report the full independent-seed replication without changing checkpoints."""
import csv,json
from .study import HERE,read,write,sha


def run():
    r=read(HERE/'results.json');frozen=read(HERE/'analysis_frozen.json')
    assert sha(HERE/'analyze.py')==frozen['analysis_sha256']
    primary=next(x for x in r['mechanism'] if x['arm']=='union_native_slots' and x['interaction']=='state_controller_eta')
    low,high=primary['skill_over_zero']['censor_CI95']
    classification='SOURCE_INTERACTION_REPLICATES' if low>0 else 'SOURCE_INTERACTION_NOT_PREDICTED' if high<0 else 'SOURCE_INTERACTION_REPLICATION_UNCERTAIN'
    lines=['FROZEN SOURCE MODEL / INDEPENDENT ROLLOUT SEEDS',classification,
        'All32originalRingTRAINphysicalstates x5actualFFFlow-referencecontrollers x4prefixedeta. Oldseed0/1 remain unchanged; onlyseed2/3new. No newstates/controllers,candidatechange,modeltraining,normalizationorcheckpointselection.',
        'The12existingmodels/predictionswerehashedbeforeanynewoutcomes. SameFlow,execution,safetyandprecisionidentitieschecked. Existing1280outcomesreused;1280missingoutcomesgeneratedafterauthoritativecachepreflight. TT/H20repeated-seed evidence hasdifferentexecutionsemanticsandcannotreplace these labels.',
        'This tests whether fitted source physicalstates carry reproducibleQ effects. It is NOT fresh-state generalization, an unseencontroller test, four-sceneLOSO or B15. K4/Q2 selectionmetrics are diagnostics only.',
        '20,000physical-statebootstrap replicates; allC/eta/rollout measurementscluster together. Unknownnumericaloutcomeskeptascensorbounds. Noise-correctedMSEuses (prediction-Y_seed2)*(prediction-Y_seed3); centeredvariantsrecenterwithineachbootstrap. Negativeunbiasedestimatesarepreserved,notclipped. Unknowncenteringboundsareconservative.',
        'OldQ2lookup is an explicit fixed diagnostic predictor formed fromoriginalseed0/1; originalunknowns mapto0.5inthis predictor,notdeclaredtrueoutcomes. Controller-etaprior removesitsstateassociation. Allnewlabelsremainscensored; nounknownexclusion.',
        f"Unknownold={r['total_unknown_old']},unknownnew={r['total_unknown_new']},collisionsold={r['collision_old']},collisionsnew={r['collision_new']}.",'',
        'arm interaction midpoint_corr predictive_skill_bounds skill95CI noise_corrected_MSE_bounds MSE95CI']
    for x in r['mechanism']:
        lines.append(f"{x['arm']} {x['interaction']} {x['correlation_midpoint_sensitivity']} {x['skill_over_zero']['identification_bounds']} {x['skill_over_zero']['censor_CI95']} {x['noise_corrected_MSE']['identification_bounds']} {x['noise_corrected_MSE']['censor_CI95']}")
    lines.extend(['','Selection secondary: arm success matched-minus-mismatch95CI vs-perC-best-fixed95CI unknown collision'])
    for x in r['rows']:
        lines.append(f"{x['arm']} {x['success_bounds']} {x['matched_minus_mismatch']['censor_CI95']} {x['minus_reselected_best_per_C_fixed']['censor_CI95']} {x['selected_unknown']} {x['selected_collision']}")
    lines.extend(['','Decision boundary: positiveindependent-sourceinteractionskill would ruleout pureTRAIN-noise memorization as the whole explanation. It would not ruleout partialnoiseoverfit or insufficientunseen-statecoverage. Failure here would require precision/estimationdiagnosis before blindlyexpandingstates; it would not aloneproveinputinsufficiency.',
        'No automatic retraining or datasetexpansion is triggered. This finding is combined with the already-completed sourceVAL/allfourLOSOandnative512controls for the next single root-cause contrast.',
        'Artifacts: immutableprotocol/modelsmanifest/cacheaudit/preflight;rawper-seedJSON+journals+SQLitekeys;raw_labels.npz;individualmodelpredictions;matched/mismatch selections;allfixedandallqualifyingreversalresults;20kbootstrap;allstatecasesCSV. No videos required forpureQexperiments.'])
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    for name,rows in [('mechanism.csv',r['mechanism']),('selection.csv',r['rows'])]:
        keys=list(dict.fromkeys(k for x in rows for k in x))
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in x.items()} for x in rows)
    write(HERE/'completion.json',dict(status='COMPLETE_SOURCE_DIAGNOSTIC',classification=classification,new_rollouts=1280,reused_rollouts=1280,
        new_fits=0,root_cause_closed=False,report_sha256=sha(HERE/'REPORT.txt')))
    print(classification,primary['skill_over_zero'],flush=True)


if __name__=='__main__':run()
