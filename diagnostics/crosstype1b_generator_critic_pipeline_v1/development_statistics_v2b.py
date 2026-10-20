"""Nine held-type development folds on the common observed seed."""
from common import *
from statistics_pipeline import evaluate


def main():
    assert read(HERE/'STATISTICS_TESTS.json')['passed']
    assert read(HERE/'SHRINKAGE_SELECTION_TESTS.json')['passed']
    for name in ('TRAINING_BUNDLE','DEVELOPMENT_CHOICES_V2B'):
        for path,digest in read(HERE/f'{name}.receipt.json')['files'].items():assert sha(path)==digest,path
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(HERE/'DEVELOPMENT_CHOICES_V2B.npz',allow_pickle=False) as z:
        assert np.array_equal(data['names'],z['names'])
        fixed={k:z[k] for k in z.files if k not in ('names','types')}
    # Generated candidates have only seed0 in the authorized design. Never mix
    # their one-seed outcomes with two-seed anchor estimates in paired scores.
    data={k:(v[...,:1] if k in ('labels','uids','collision','timeout','unknown') else v) for k,v in data.items()}
    reports=evaluate(data,fixed,5000);combined=reports['combined']['type_equal'];primary=combined['methods']['pipeline']
    type_gains={t:r['type_equal']['methods']['pipeline']['gain_vs_progress_pp'] for t,r in reports.items() if t!='combined'}
    goals=dict(paired_CI_lower_positive=primary['gain_CI95_pp'][0]>0,
        crossfit_regret_less_than_half_progress=None,
        crossfit_gap='Generator seed1 was not included in the original training-label design; no full-menu crossfit identification',
        no_held_type_loss_over_2pp=all(v>=-2 for v in type_gains.values()),
        zero_harm=primary['harm_possible']==0,
        all_targets_demonstrated=False)
    target=HERE/'DEVELOPMENT_RESULTS_V2B.json';assert not target.exists()
    write(target,dict(utc=utc(),evidence_level='DEVELOPMENT_HELD_TYPE_COMMON_SEED0',
        summaries=reports,development_targets=goals,per_type_gain_pp=type_gains,
        training_anchors_additional_seed_not_used_for_paired_evaluation=True,
        no_new_evaluation_outcomes=True))
    seal('DEVELOPMENT_RESULTS_V2B.receipt.json',[target,HERE/'DEVELOPMENT_CHOICES_V2B.receipt.json',
        HERE/'TRAINING_BUNDLE.receipt.json',HERE/'development_statistics_v2b.py',HERE/'statistics_pipeline.py',HERE/'STATISTICS_TESTS.receipt.json'])
    print('Development paired statistics complete',flush=True)


if __name__=='__main__':main()
