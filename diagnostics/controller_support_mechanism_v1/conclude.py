import json,csv
from pathlib import Path
p=Path(__file__).resolve().parent
sets=[json.load(open(p/f'{f}_results.json')) for f in ['toy','ring']]
rows=[r for d in sets for r in d['rows']];skills=[r for d in sets for r in d['skills']]
for name,data in [('summary',rows),('interaction_summary',skills)]:
    out=[{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items() if k!='reversal_pairs'} for r in data]
    with (p/f'{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in out))));w.writeheader();w.writerows(out)
lines=['Stage20 — reuse old crossed-support tests; calibration versus adaptation','',
'COMPLETE:30oldfits reused;zero newfits/queries/rollouts. Both sourcedirections,all12oldpanels incl32openedQ16states/scenewithunseenC6/7. PhysicalTRAIN/evaluationoverlap0.',
'Originaltreatment:64SOURCEstates x8analyticC;512extraoutcomes from8moreSOURCEstates or8TARGETstates x4C. Wrong-C/eta controlsalreadyfit. Same6000steps,source-onlynorm,fixedfinalcheckpoint. OLDstate_contact_bilinearinput;notcurrentcompact64repair.',
'Crossedarms useTARGETlabels;NOTzero-shotLOSO. All oldconfirmationisDEVELOPMENT. EmpiricalSOURCEeta-priorreferenceisnotatrained/fairpairedQeta.',
'Addedanalysis removesstate/eta/Cpriors;recenterinside20000physical-statebootstrap;same-scenemismatchresamplesdonor/recipient;all16fixed/bootstrapbest;8/8independenthalf reversal. Savedpredictions/permutations/per-seedlabels/bootstrap.',
'Keyresult:8targetstates improvedQ_MSE butdidNOTrestorestate-conditionedselection onindependentunseen-C targetpanels. Both mismatchCIsinclude0;SxCxeta skillnotpositive. Ringsource->Toytargetinteractionissignificantlynegativeevenwithcorrecttargetsupport.','',
'TargetnaturalQ16:sourcefold arm Q_MSE S_eta_corr success matched-mismatchCI best-fixedCI']
for r in rows:
    if r['target_scene'] and r['panel']=='opened_confirmation_unseen' and r['arm'] in ['source','more_source','crossed','crossed_wrong_controller','crossed_wrong_eta']:
        lines.append(f"{r['variant']} {r['arm']} {r['Q_MSE_midpoint']:.5f} {r['state_eta_corr']} {r['success_bounds']} {r['matched_minus_mismatch']['censor_CI95']} {r['minus_bootstrap_reselected_per_controller_fixed']['censor_CI95']}")
lines.append('\nIndependenttargetinteraction:positivegainmeansbetterthansecond')
for r in skills:
    if r['target_scene'] and r['panel']=='opened_confirmation_unseen' and r['first']=='crossed':
        lines.append(f"{r['variant']} {r['interaction']} crossed-{r['second']}: {r['identification_bounds']};CI={r['censor_CI95']}")
lines.extend(['','Boundary:WithdrawstrongerreadingofoldQcalibration/supportresultsasadaptive-transferproof. Notproofallcoverageexpansionsfail. Oldinput,8targetstates,Q1trainingremainlimitations.',
'Single-sourcetransferalsofailed:Stage19mixed-sceneinterferencecannotaloneexplainallhistoricaltransferfailure. Stage19stillisolatesa realnative-branchtrainingcause. RepairmusttestthatcausewithoutcallingitthecompleteLOSOrationale.',
'Artifacts:protocol,per-foldresults,summary/interactionCSV,bootstrapNPZ,matched/mismatchedpredictions;original30checkpoints/rawseedlabelsreferencedinprotocol.'])
(p/'REPORT.txt').write_text('\n'.join(lines)+'\n')
(p/'completion.json').write_text(json.dumps(dict(status='COMPLETE_NEGATIVE_MECHANISM_REANALYSIS',reused_fits=30,new_fits=0,new_queries=0,new_rollouts=0,panels=12,not_LOSO_confirmation=True),indent=2)+'\n')
print('Saved12-panelreport;Qcalibrationgainisnotadaptationproof')
