"""Independent aggregate checks against raw rows and historical published results."""
from core import *
import csv,collections

def main():
    result=read(HERE/'results.json');rows=[read(f) for f in (HERE/'main/raw').glob('*.json')];assert len(rows)==6272;by=collections.defaultdict(list)
    for r in rows:by[r['case']['arm']].append(r)
    for arm,rs in by.items():
        assert len(rs)==896
        lower=sum(r['success'] if not r['numerical_failure'] else 0 for r in rs)/896;upper=sum(r['success'] or r['numerical_failure'] for r in rs)/896
        table=result['student_table'][arm];assert abs(table['success']['mean']-lower)<1e-12;assert abs(table['success_upper']-upper)<1e-12;assert abs(table['collision']['mean']-sum(r['collision'] for r in rs)/896)<1e-12
    table=result['student_table'];assert abs(result['TS1']['mean']-(table['S500']['success']['mean']-table['safety baseline']['success']['mean']))<1e-12
    assert abs(result['TS3a']['regret']['mean']-(table['oracle']['success']['mean']-result['selectors']['A3_full']['success']['mean']))<1e-12
    assert abs(result['selectors']['A0_full']['success']['mean']-table['S500']['success']['mean'])<1e-12
    old=read(D/'decentralized_pipeline_v1/results.json')
    for mode in ['local','shared','central']:
        for arm in ['F10','F13','S300','S500']:assert abs(result['teacher_tables'][mode][arm]['success']['mean']-old['rates'][mode+'/'+arm]['success_lower'])<1e-12
    metric_rows=list(csv.DictReader((HERE/'rollout_metrics.csv').open()));assert len(metric_rows)==6272 and len({r['rollout_uid'] for r in metric_rows})==6272
    assert result['transitions']['total_nominal_collision']==sum(r['collision'] for r in by['nominal only'])
    write(HERE/'RESULTS_VALIDATION.json',dict(utc=utc(),passed=True,raw_arm_counts_and_rates_exact=True,paired_gain_regret_identities_exact=True,A0_equals_alwaysS500=True,all12_teacher_reference_rates_reproduced=True,metric_rows_unique=6272,bootstrap_draws=5000,bootstrap='controller then32state clusters with arms/seeds paired; matches round28 resampling unit'))
if __name__=='__main__':main()
