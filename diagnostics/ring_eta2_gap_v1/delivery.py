"""Final self-contained delivery receipt after frozen TEST analysis."""
from core import *

def main():
    test_labels_allowed()
    assert read(HERE/'train_DELIVERY_AUDIT.json')['status']=='PASS'
    assert read(HERE/'test_DELIVERY_AUDIT.json')['status']=='PASS'
    assert read(HERE/'CHECKPOINT_NOVELTY_AUDIT.json')['status']=='PASS'
    assert read(HERE/'COMPATIBILITY_AUDIT.json')['status']=='PASS'
    assert read(HERE/'CONSISTENCY_GATE.json')['passed']
    results=read(HERE/'results.json')
    assert set(results['hypotheses'])=={'B1','B2','B3','B4'}
    assert len(results['controllers'])==18
    files=['results.json','REPORT.md','HANDOFF.md','MODEL_REGRET.svg',
           'CONTROLLER_CHOICES.svg','ETA2_PROBE_SCATTER.svg',
           'FEATURES_FROZEN.receipt.json','TEST_FREEZE.receipt.json',
           'train_DELIVERY_AUDIT.json','test_DELIVERY_AUDIT.json']
    for n in files:assert (HERE/n).is_file(),n
    write(HERE/'DELIVERY_AUDIT.json',dict(status='PASS',utc=utc(),files={n:sha(HERE/n) for n in files},
          hypotheses=results['hypotheses'],train_cases=18432,test_cases=27648,
          gate_cases=32,unlabeled_probes=2896,checkpoint_count=36,
          test_unknown=results['unknown_count'],video_exception='pure Q experiment'))
    print(json.dumps(dict(status='PASS',hypotheses=results['hypotheses'])))

if __name__=='__main__':main()
