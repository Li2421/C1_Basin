"""Locked capability gates emit pass/exclusion only, never arm success rates."""
from common import *


def collect(phase,cohort):
    assert cohort in ('development','locked')
    p=read(HERE/f'{phase}_protocol.json');cases=read(HERE/f'{phase}_CASES.json')
    counts={name:{arm:[0,0,0] for arm in ('safety_baseline','S500','F10')} for name in p['controller_order']}
    with rollout_db.connect(True) as db:
        for c in cases:
            r=db.execute('SELECT success,numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if r is None:raise RuntimeError('Incomplete capability gate; missing records are not unknown')
            item=counts[c['base_controller_name']][c['arm']]
            item[0]+=int(r['success'] and not r['numerical_failure']);item[1]+=int(r['numerical_failure']);item[2]+=1
    eligible=[];rows=[]
    for name,armcounts in counts.items():
        assert all(v[2]==16 for v in armcounts.values())
        lower=np.asarray([v[0]/16 for v in armcounts.values()])
        upper=np.asarray([(v[0]+v[1])/16 for v in armcounts.values()])
        passed=bool(lower.max()>=.15);excluded=bool(upper.max()<.15)
        status='PASS' if passed else 'EXCLUDED' if excluded else 'GAP_UNKNOWN_CAPABILITY'
        if passed:eligible.append(name)
        row=dict(controller=name,status=status,not_needing_correction=bool(np.all(lower>=.98)),
                 reason=None if passed else 'All three-arm upper success bounds below 15%' if excluded else 'Unknown records prevent threshold decision')
        if cohort=='development':row['arm_lower_success_rates']={arm:armcounts[arm][0]/16 for arm in armcounts}
        rows.append(row)
    if cohort=='development':
        gates=read(HERE/'REPRODUCTION_GATE.json')['type_passed']
        old=read(HERE/'HISTORICAL_BASE_PROTOCOL.json')['profiles']
        for name,cp in old.items():
            kind=cp['payload'].get('controller_class','Flow')
            key='base_u' if kind=='Flow' else 'G' if kind=='G' else kind
            if gates.get(key,False):eligible.append(name)
    write(HERE/f'{cohort.upper()}_CAPABILITY.json',dict(utc=utc(),eligible=eligible,controllers=rows,
          locked_arm_rates_recorded=False if cohort=='locked' else None,missing_not_relabelled_unknown=True))
    print(cohort,'capability decisions recorded; eligible count',len(eligible),flush=True)

if __name__=='__main__':collect(sys.argv[1],sys.argv[2])
