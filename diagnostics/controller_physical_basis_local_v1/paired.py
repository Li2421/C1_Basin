"""Paired representation intervention; no repeated training or rollouts."""
import numpy as np
from .study import HERE,BASE,FOLDS,read,write,sha


def run():
    old=read(BASE/'results.json');new=read(HERE/'results.json');rows=[];boot={}
    for f,sc in FOLDS.items():
        a=np.load(HERE/f'bootstrap_{f}.npz');b=np.load(BASE/f'bootstrap_{f}.npz')
        tag=sc+'__held_all';np.testing.assert_array_equal(a[tag+'__indices'],b[tag+'__indices'])
        k=tag+'__state_eta__skill';v=a[k]-b[k]
        r=next(r for r in new['rows'] if r['fold']==f and r['split']=='held_all' and r['metric']=='state_eta')
        rel=next(r for r in new['basis_rows'] if r['fold']==f and r['split']=='held_all' and r['basis']=='B_relative')
        rows.append(dict(fold=f,paired_CI95=np.quantile(v,[.025,.975]).tolist(),new_skill_CI95=r['skill_CI95'],new_Brelative_skill_CI95=rel['state_skill_CI95']))
        boot[f]=v
    four=next(r for r in rows if r['fold']=='four')
    gate=all(four[k][0]>0 for k in ('paired_CI95','new_skill_CI95','new_Brelative_skill_CI95'))
    result=dict(rows=rows,gate_passed=gate,new_fits=12,reused_fits=24,new_rollouts=0,new_queries=0,
        root_cause_closed=False,decision='Focused primitive-inputrepair gate passed; inspect beforeQtest.' if gate else
        'Focused gate FAILED: explicitphysicalprimitive channels alone didnotrepairFour correctiontransfer. No furtherbasis-channel/width/regularization sweeps orQfits onthischange.')
    write(HERE/'paired_results.json',result);np.savez_compressed(HERE/'paired_bootstrap.npz',**boot)
    with (HERE/'REPORT.txt').open('a') as out:
        out.write('\nPAIRED SINGLE REPRESENTATION INTERVENTION\n'+result['decision']+'\n')
        for r in rows:out.write(str(r)+'\n')
    write(HERE/'completion.json',dict(status='COMPLETE_PHYSICAL_PRIMITIVE_INPUT_TEST',gate_passed=gate,root_cause_closed=False,new_rollouts=0,report_sha256=sha(HERE/'REPORT.txt')))
    print(result)


if __name__=='__main__':run()
