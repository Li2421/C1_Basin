"""Paired oracle-intermediate intervention, all folds and all controls."""
import csv
import numpy as np
from .study import HERE,BASE,FOLDS,read,write,sha


def run():
    rows=[];paired=[];boot={};sources=[]
    for fold in FOLDS:
        new=read(HERE/f'review_{fold}.json')['rows'];old=read(BASE/f'review_{fold}.json')['rows']
        rows.extend(new);sources.extend(r for r in new if r['split']=='source_validation' and r['metric']=='state_eta')
        a=np.load(HERE/f'bootstrap_{fold}.npz');b=np.load(BASE/f'bootstrap_{fold}.npz')
        for r in new:
            if r['split']!='held_all' or r['metric'] not in ('state_eta','state_controller_eta'):continue
            tag=r['scene']+'__held_all';np.testing.assert_array_equal(a[tag+'__indices'],b[tag+'__indices'])
            k=tag+'__'+r['metric']+'__skill';v=a[k]-b[k]
            o=next(z for z in old if z['split']=='held_all' and z['metric']==r['metric'])
            paired.append(dict(fold=fold,metric=r['metric'],native=o['interaction_skill'],mediated=r['interaction_skill'],
                mediated_CI95=r['skill_CI95'],paired_change=r['interaction_skill']-o['interaction_skill'],
                paired_CI95=np.quantile(v,[.025,.975]).tolist(),per_seed_skill=r['per_seed_skill'],
                true_energy_CI95=r['true_energy_CI95']))
            boot[fold+'__'+r['metric']]=v
    four=next(r for r in paired if r['fold']=='four' and r['metric']=='state_eta')
    gate=four['mediated_CI95'][0]>0 and four['paired_CI95'][0]>0
    qualifying=[r['fold'] for r in paired if r['metric']=='state_eta' and r['mediated_CI95'][0]>0 and r['paired_CI95'][0]>0]
    result=dict(rows=rows,paired=paired,focused_gate_passed=gate,qualifying_local_folds=qualifying,
        source_positive_CI_count=sum(r['skill_CI95'][0]>0 for r in sources),source_checks=len(sources),
        root_cause_closed=False,new_fits=12,baseline_fits_reused=24,new_rollouts=0,new_queries=0,
        decision='Inspectqualifyinglocalfolds/controllerinteraction andQhistory beforeaQrepair.' if qualifying else
        'CorrectdeltaPLUSunprojectedcontrollerendpoint didnotrepairlocalFFtransfer. Purepreprojectionalgebra alone isNOTasufficientrepairunderthisinterface. NoQfits/rollouts fromthisnegative; projectedpath/geometry learning remains unresolved.',
        boundary='Oracleintermediateprovidedforlearning, nothard-wired solver orperfectdecoder. Negative result falsifies thissufficientrepair, notallpossibleuseofcorrection; doesnotprovewhattheremaininguniqueQcauseis. Actualruntimealwayscomputedcorrectcorrection; failureisinlearnedpredictors. DEVELOPMENTsource-onlyfits, nofreshconfirmation.')
    write(HERE/'results.json',result);np.savez_compressed(HERE/'paired_bootstrap.npz',**boot)
    with (HERE/'summary.csv').open('w') as f:
        fields=sorted(set().union(*(r.keys() for r in rows)));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    lines=['ORACLE UNPROJECTED RESPONSE — LOCAL FF ACTION PREDICTION',
        f'Source state_eta positiveCI {result["source_positive_CI_count"]}/{len(sources)}; qualifyingheldfolds {qualifying}',result['decision'],result['boundary'],'',
        'Fold | metric | correctdelta-only skill | plus-unprojected skill [95%CI] | pairedchange [95%CI]']
    for r in paired:lines.append(f'{r["fold"]} | {r["metric"]} | {r["native"]:.8g} | {r["mediated"]:.8g} {r["mediated_CI95"]} | {r["paired_change"]:.8g} {r["paired_CI95"]}')
    lines.extend(['','AllStage35labels/K4/160naturalphysicalstates/source-onlynormalization/sourceVALcheckpoint/6000steps/networkshapes preserved. Actualdelta retained10:12; additionalunprojectedendpoint injected12:14 beforepooling, retainingalloldinformation. Everyfitseedandallscenecontrols saved.',
        '12newstatefits; old12eta andall24baselinefits reused. 0queries,0physicalsteps,0rollouts. Localactionmetrics NOTQ/success metrics. Physicalstatepairedbootstrap20000, recenter/donorresample, triplepowerweakRingreported.',
        'Relatedprimaryperspective consulted beforeprotocol: https://proceedings.mlr.press/v70/amos17a.html. Thisisanempiricaloracleintermediatecontrast, notnaturalcausalmediation identification.'])
    (HERE/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_ORACLE_UNPROJECTED_MEDIATION_LOCAL',focused_gate_passed=gate,
        root_cause_closed=False,new_rollouts=0,report_sha256=sha(HERE/'REPORT.txt')))
    print(dict(source_positive=result['source_positive_CI_count'],qualifying=qualifying))
    for r in paired:print(r['fold'],r['metric'],r['mediated_CI95'],r['paired_CI95'])


if __name__=='__main__':run()
