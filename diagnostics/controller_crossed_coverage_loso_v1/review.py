"""Reuse completed all-fold analysis engines, with explicit coverage aliases."""
import argparse,csv,json
from .study import HERE,ROOT,BASE,MAT,ALL_FOLDS,LEGEND,read,write,freeze,sha


def matrix():
    from diagnostics.controller_native_mass_repair_v1 import review as engine
    assert sha(__file__)==read(HERE/'evaluation_protocol.json')['code_sha256']
    engine.HERE=HERE;engine.MAT=MAT;engine.FOLDS=ALL_FOLDS
    engine.matrix()
    d=read(HERE/'matrix_results.json')
    d['source_training']='FixedQ4 N32 versus N128crossedRingSOURCEstates;bothnative20sampler. Internal uniform_* aliases mean N32Q4,notuniformsampling. HeldRingmodelsidenticalreused;alltargetlabels excludedfromfit/selection/normalization.'
    d['model_legend']=LEGEND
    d['scope']='Opened DEVELOPMENT. All40heldstates/scene, K4. RingcachedQ4;other3originalQ2/Q4. JointsourceVAL8 descriptive, notuntouchedconfirmation. No newrollouts.'
    write(HERE/'matrix_results.json',d)


def interaction():
    from diagnostics.controller_native_mass_repair_v1 import interaction_audit as engine
    assert sha(__file__)==read(HERE/'evaluation_protocol.json')['code_sha256']
    engine.HERE=HERE;engine.MAT=MAT;engine.FOLDS=ALL_FOLDS
    freeze(HERE/'interaction_protocol.json',dict(question='Does crossedSOURCEcoverage32->128repairheldscene SxCxeta prediction?',
        code_sha256=sha(engine.__file__),wrapper_sha256=sha(__file__),model_legend=LEGEND,
        stats='Reuseexact20kphysical-statebootstrap,recenterperreplicate,unknownendpointbounds;allCpairscounted. Compareeveryarmwithzerointeraction andN128withN32.',
        scope='Allfourheldscenes40naturalstates;sourceVAL8 descriptive. HeldRingmodelunchanged. No newrollout.'))
    engine.run()


def conclude():
    d=read(HERE/'matrix_results.json');inter=read(HERE/'interaction_results.json');rows=[];transfers=[]
    for r in d['rows']:
        if r['reference']!='flow':continue
        if not ((r['fold']=='joint' and r['panel']=='validation') or (r['fold']!='joint' and r['panel']=='all_held')):continue
        key={k:r[k] for k in ('fold','scene','panel','reference','condition')}
        comps={z['comparison']:z for z in d['comparisons'] if all(z[k]==v for k,v in key.items())}
        mech=next(z for z in d['mechanisms'] if all(z[k]==v for k,v in key.items()))
        x=next(z for z in inter['rows'] if z['fold']==r['fold'] and z['scene']==r['scene'] and z['reference']=='flow' and z['interaction']=='state_controller_eta' and z['first']==r['condition'] and z['second']=='zero')
        row=dict(fold=r['fold'],scene=r['scene'],N=r['N'],model=LEGEND[r['condition']],
            success_bounds=r['selected_success_bounds'],Q_MSE_midpoint=r['Q_MSE_midpoint_sensitivity'],
            oracle_regret_bounds=r['oracle_regret_bounds'],oracle_success_bounds=r['oracle_success_bounds'],
            state_eta_corr=r['state_eta_corr_midpoint_sensitivity'],triple_skill_bounds=x['skill_identification_bounds'],triple_skill_CI=x['censor_CI95'],
            matched_minus_mismatch_bounds=comps['matched_minus_same_scene_wrong_state']['identification_bounds'],
            matched_minus_mismatch_CI=comps['matched_minus_same_scene_wrong_state']['censor_CI95'],
            minus_posthoc_perC_fixed_CI=comps['selection_minus_posthoc_best_fixed_per_controller']['censor_CI95'],
            selection_entropy=r['selection_entropy_bits'],selected_collision=r['selected_collision'],selected_unknown=r['selected_numerical_unknown'],
            reversal=mech['reversal'])
        if r['condition']=='union_native_slots' and r['fold']!='joint':
            g=next(z for z in inter['rows'] if z['fold']==r['fold'] and z['scene']==r['scene'] and z['reference']=='flow' and z['interaction']=='state_controller_eta' and z['first']=='union_native_slots' and z['second']=='uniform_native_slots')
            transfers.append(dict(fold=r['fold'],N=r['N'],Q_gain=comps['Q_MSE_improvement_over_uniform_native_slots'],
                selection_gain=comps['selection_improvement_over_uniform_native_slots'],interaction_gain=g,
                mechanism_and_decision_positive=bool(x['censor_CI95'][0]>0 and row['matched_minus_mismatch_CI'][0]>0 and row['minus_posthoc_perC_fixed_CI'][0]>0),
                no_treatment_reference=r['fold']=='held_ring'))
        rows.append(row)
    write(HERE/'results.json',dict(rows=rows,transfers=transfers,model_legend=LEGEND,
        complete_four_folds=True,new_rollouts=0,new_fits=36,root_cause_closed=False,
        boundary='OpenedDEVELOPMENT. Ringheldsourceunchanged;nativeRingdataexpansion cannotrepair itsfold. Jointfitsreused. No all16fixed/B15orfinalconfirmationclaim.'))
    with (HERE/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    lines=['CROSSED COVERAGE SOURCE-ONLY LOSO EXTENSION',
        'N32Q4 versusN128Q4. Samearchitecture,physical/API+compact64inputs,source-onlyfrozennormalization,native20mass,6000steps,sourceVALcheckpoint rule. EachN32/N128pairedfit onsameCPU.',
        '36necessaryfits for3folds containingRing;12jointfitsand6heldRingfitsreused. Zero newrollout/staticquery. HeldRingisuntreatedreference:itsSOURCEtrainingdataunchanged.',
        'Allfourheldnatural40statepanels reported,K4. ExistingRingseedreplicationenrichesitsQ2toQ4;remainingQ2/Q4retained. Allcorrect/wrong-state/fixed/reversalcontrols and20kphysicalbootstrap areinmatrix_results/interaction_results andmatrix_review/.',
        'AllopenedDEVELOPMENT;positivefindingsrequireuntouchedfreshconfirmation. Aliasesuniform_* meansN32Q4inthisexperiment,NOTuniformCweights.',
        '', 'fold model success Q_MSE Sxeta_corr SxCxeta_skill_CI matched_wrong_CI']
    for r in rows:lines.append(f"{r['fold']} {r['model']} {r['success_bounds']} {r['Q_MSE_midpoint']} {r['state_eta_corr']} {r['triple_skill_CI']} {r['matched_minus_mismatch_CI']}")
    lines+=['','PAIRED COVERAGE EFFECTS']+[json.dumps(z) for z in transfers]
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_COVERAGE_LOSO_DEVELOPMENT',new_fits=36,new_rollouts=0,
        candidate_folds=[x['fold'] for x in transfers if x['mechanism_and_decision_positive']],root_cause_closed=False,report_sha256=sha(HERE/'REPORT.txt')))
    print({'transfers':transfers},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['matrix','interaction','conclude']);a=p.parse_args();globals()[a.stage]()
