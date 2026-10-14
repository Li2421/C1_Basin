"""Verify full rollout evidence, merge once, postflight, and run frozen analysis."""
from core import *
import subprocess


PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def call(args, log):
    with (HERE/log).open('w') as output:
        subprocess.run([PY,'-B',*args],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,check=True)


def verify():
    labels_allowed()
    assert read(HERE/'CONSISTENCY_GATE.json')['passed']
    counts={}
    for phase in ('gate','wave1'):
        rows=read(HERE/f'{phase}_CASES.json')
        assert len(rows)==(32 if phase=='gate' else 26112)
        assert len({c['rollout_uid'] for c in rows})==len(rows)
        outcomes={};unknown=[]
        for c in rows:
            path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
            assert path.is_file(),path
            r=read(path)
            assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json')
            assert r['execution_code_sha256']==sha(HERE/'run.py')
            assert r['jax_enable_x64'] is False
            marker=HERE/phase/'journaled'/f"{c['rollout_uid']}.json"
            if phase=='wave1':
                assert marker.is_file() and read(marker)['raw_sha256']==sha(path)
            if r['numerical_failure']:
                assert r['success']==0 and r['outcome']=='numerical_failure'
                unknown.append(c['rollout_uid'])
            outcomes[r['outcome']]=outcomes.get(r['outcome'],0)+1
        counts[phase]=dict(cases=len(rows),unknown=len(unknown),outcomes=outcomes)
    write(HERE/'ROLLOUT_COMPLETENESS_AUDIT.json',dict(status='PASS',utc=utc(),counts=counts,
         all_26112_main_cases_present=True,all_main_cases_journaled=True,
         unknown_retained_without_rerun=True))
    return counts


def main():
    counts=verify()
    call([str(HERE/'merge.py')],'finalize_merge.log')
    call(['-m','shared_rollout_db.plan','--manifest',str(HERE/'wave1_PLANNED_ROLLOUTS.json'),
          '--output',str(HERE/'completed_POSTFLIGHT.json')],'finalize_postflight.log')
    post=read(HERE/'completed_POSTFLIGHT.json')
    assert post['summary']['ambiguous']==0
    call([str(HERE/'postflight_unknown_audit.py')],'finalize_unknown_audit.log')
    unknown=read(HERE/'POSTFLIGHT_UNKNOWN_AUDIT.json')
    assert unknown['status']=='PASS' and unknown['true_absent']==0
    assert unknown['unknown_present_in_db']==counts['wave1']['unknown']
    write(HERE/'POSTFLIGHT_AUDIT.json',dict(status='PASS',utc=utc(),
          standard_postflight_sha256=sha(HERE/'completed_POSTFLIGHT.json'),
          unknown_audit_sha256=sha(HERE/'POSTFLIGHT_UNKNOWN_AUDIT.json'),
          completeness_audit_sha256=sha(HERE/'ROLLOUT_COMPLETENESS_AUDIT.json'),
          summary=post['summary'],true_absent=0,
          numerical_failure_retained_as_unknown=counts['wave1']['unknown']))
    call([str(HERE/'analyze.py')],'finalize_analysis.log')
    assert (HERE/'results.json').is_file() and (HERE/'REPORT.md').is_file() and (HERE/'HANDOFF.md').is_file()
    write(HERE/'DELIVERY_AUDIT.json',dict(status='PASS',utc=utc(),
          final_freeze_sha256=sha(HERE/'FINAL_FREEZE.receipt.json'),
          results_sha256=sha(HERE/'results.json'),report_sha256=sha(HERE/'REPORT.md'),
          handoff_sha256=sha(HERE/'HANDOFF.md'),counts=counts))
    print(json.dumps(dict(status='PASS',counts=counts)))


if __name__=='__main__':main()
