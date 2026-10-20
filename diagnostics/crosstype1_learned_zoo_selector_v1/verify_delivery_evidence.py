"""Check complete cohorts, immutable choices and unknown-preserving evidence."""
from common import *


def main():
    assert (HERE/'CONFIRMATORY_RESULTS.receipt.json').exists()
    counts={};inputs=[]
    for cohort,expected in (('DEVELOPMENT',56),('LOCKED',6)):
        path=HERE/f'{cohort}_BUNDLE.npz'
        for source,digest in read(HERE/f'{cohort}_BUNDLE.receipt.json')['files'].items():assert sha(source)==digest
        with np.load(path,allow_pickle=False) as z:
            assert len(z['names'])==expected and len(set(z['names']))==expected
            assert z['labels'].shape==(expected,32,8,2) and z['tuning'].shape==(expected,16,8)
            for key in ('labels','tuning','nominal'):assert np.isin(z[key],[-1,0,1]).all()
            assert np.array_equal(z['unknown'],z['labels']==-1)
            assert np.array_equal(z['nominal_unknown'],z['nominal']==-1)
            counts[cohort.lower()]=dict(controllers=expected,types=sorted(set(z['types'].tolist())),
                evaluation_requests=int(z['labels'].size+z['nominal'].size),tuning_requests=int(z['tuning'].size),
                menu_evaluation_unknown=int(np.sum(z['labels']==-1)),nominal_unknown=int(np.sum(z['nominal']==-1)),
                tuning_unknown=int(np.sum(z['tuning']==-1)),unavailable_probe_controllers=int(np.sum(~np.isfinite(z['features']).any(1))))
        inputs.append(HERE/f'{cohort}_BUNDLE.receipt.json')
    phases=('old_probe','old_evaluation','old_tuning_v2','new_probe','new_evaluation','new_tuning',
        'new_capability_all','locked_capability_initial','ppo_checkpoint','locked_probe','locked_evaluation','locked_tuning')
    checked={};file_cache={}
    for phase in phases:
        for suffix in ('COMPLETE.receipt.json','EXECUTION_FROZEN.receipt.json'):
            receipt=HERE/f'{phase}_{suffix}'
            for source,digest in read(receipt)['files'].items():
                actual=file_cache.setdefault(source,sha(source))
                assert actual==digest,(phase,source)
            inputs.append(receipt)
        post=HERE/f'{phase}_POSTFLIGHT_UNKNOWN.json'
        if post.exists():
            r=read(post);assert not r['absent_rows'] and r['unknown_reruns']==0
            checked[phase]=dict(missing=0,preserved_unknown=len(r['unknown_preserved']))
        else:
            r=read(HERE/f'{phase}_PROBE_POSTFLIGHT.json');assert r['task_label_rows_created']==0
            checked[phase]=dict(probes=r['planned'],unavailable=r['unknown_probe_count'],task_labels_created=0)
    lock=read(HERE/'CONFIRMATORY_RESULTS.json')
    assert read(HERE/'FROZEN_PIPELINE.receipt.json')['utc']<lock['choices_utc']<lock['first_label_submission_utc']
    assert read(HERE/'DEVELOPMENT_FINAL_SELECTION.json')['selected']==read(HERE/'FROZEN_PIPELINE.json')['iteration']
    assert read(HERE/'SCALE_CURVE_RESULTS.json')['final_m7_matches_development_outer_predictions']
    write(HERE/'FINAL_EVIDENCE_COVERAGE.json',dict(utc=utc(),passed=True,counts=counts,phases=checked,
        immutable_execution_receipts_verified=True,locked_chronology_verified=True,no_missing_as_unknown=True,
        locked_cohort_incomplete=True,implicit_gap_preserved=True))
    seal('FINAL_EVIDENCE_COVERAGE.receipt.json',[HERE/'FINAL_EVIDENCE_COVERAGE.json',HERE/'verify_delivery_evidence.py',*inputs])
    print(json.dumps(counts,indent=2),flush=True)


if __name__=='__main__':main()
