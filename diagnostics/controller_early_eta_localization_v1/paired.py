"""Paired bootstrap comparison; reuse original resamples and trained baselines."""
import numpy as np
from .study import HERE,BASE,FOLDS,read,write,sha


def run():
    rows=[];boot={}
    for fold in FOLDS:
        a=np.load(HERE/f'bootstrap_{fold}.npz');b=np.load(BASE/f'bootstrap_{fold}.npz')
        rr=read(HERE/f'review_{fold}.json')['rows'];oo=read(BASE/f'review_{fold}.json')['rows']
        for r in rr:
            if r['split']!='held_all' or r['metric'] not in ('state_eta','state_controller_eta'):continue
            tag=r['scene']+'__held_all';np.testing.assert_array_equal(a[tag+'__indices'],b[tag+'__indices'])
            k=tag+'__'+r['metric']+'__skill';v=a[k]-b[k]
            o=next(z for z in oo if z['split']=='held_all' and z['metric']==r['metric'])
            rows.append(dict(fold=fold,metric=r['metric'],early=r['interaction_skill'],early_CI95=r['skill_CI95'],
                late=o['interaction_skill'],paired_improvement=r['interaction_skill']-o['interaction_skill'],
                paired_CI95=np.quantile(v,[.025,.975]).tolist(),
                per_seed_improvement=(np.array(r['per_seed_skill'])-o['per_seed_skill']).tolist()))
            boot[fold+'__'+r['metric']]=v
    qualifying=[r['fold'] for r in rows if r['metric']=='state_eta' and r['early_CI95'][0]>0 and r['paired_CI95'][0]>0]
    result=dict(rows=rows,local_repair_gate=bool(qualifying),qualifying_folds=qualifying,
        root_cause_closed=False,new_fits=12,new_rollouts=0,
        decision='Review qualifyingfolds with allfalsificationcontrols.' if qualifying else
        'No early-eta Q fits: no held scene achieves positive state_eta skill CI. All4 triple skill CIs remain negative. Prune early routing as sufficient repair; no width/depth/lr sweep.')
    write(HERE/'paired_results.json',result);np.savez_compressed(HERE/'paired_bootstrap.npz',**boot)
    (HERE/'REPORT.txt').write_text('SINGLE-FACTOR EARLY ETA ROUTING\n'+result['decision']+'\n'+
        '\n'.join(str(r) for r in rows)+'\nSameinformation/parameterarrays/init/labels/optimization; eta copied into3previouslyzeroagentchannels beforepooling. Source-onlyfit; openedDEVELOPMENT.\n')
    write(HERE/'completion.json',dict(status='COMPLETE_EARLY_ETA_LOCAL_TEST',root_cause_closed=False,
        gate_passed=bool(qualifying),new_fits=12,new_rollouts=0,report_sha256=sha(HERE/'REPORT.txt')))
    print([(r['fold'],r['metric'],r['paired_CI95']) for r in rows])


if __name__=='__main__':run()
