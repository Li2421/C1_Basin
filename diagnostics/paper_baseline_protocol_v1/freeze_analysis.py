"""Prospective endpoint/statistical declarations, frozen before formal tests."""
from core import *
from scene_catalog import catalog
from method_names import NAMES
from execution_policy import active_methods
import datetime


def main():
    path=HERE/'ANALYSIS_PLAN_FROZEN.json'
    if path.exists():
        value=read(path)
        assert value['parameters_sha256']==sha(HERE/'PARAMETERS_FROZEN.json')
        assert value['ours_fixed_sha256']==sha(HERE/'OURS_FIXED_FROZEN.json')
        return value
    assert (HERE/'PARAMETERS_FROZEN.json').exists()and(HERE/'OURS_FIXED_FROZEN.json').exists()
    assert not(HERE/'FORMAL_TEST_STARTED.json').exists()
    split=read(HERE/'SPLITS_FROZEN.json');scenes=catalog('A')
    categories=sorted({scenes[name]['factor']for name in split['benchmark_A']['test']})
    video_nominal=min(row['name']for row in read(HERE/'INPUT_AUDIT.json')['models'])
    videos={category:dict(scene=min(name for name in split['benchmark_A']['test']if scenes[name]['factor']==category),
                          state=min(split['benchmark_A']['test_states']),
                          nominal=video_nominal,nominal_selection='First frozen recorded model name in lexicographic order, before formal outcomes.')
            for category in categories}
    value=dict(frozen_at=datetime.datetime.now().astimezone().isoformat(),
        splits_sha256=sha(HERE/'SPLITS_FROZEN.json'),native_reproduction_sha256=sha(HERE/'NATIVE_REPRODUCTION_RESULTS.json'),
        method_policy_sha256=sha(HERE/'METHOD_EXECUTION_POLICY.json'),parameters_sha256=sha(HERE/'PARAMETERS_FROZEN.json'),
        ours_fixed_sha256=sha(HERE/'OURS_FIXED_FROZEN.json'),menus_sha256=sha(HERE/'PAPER_MENUS_FROZEN.json'),
        execution_release_sha256=sha(HERE/'EXECUTION_FROZEN.json'),
        relaxed_scenes_sha256=sha(HERE/'RELAXED_SCENES_FROZEN.json'),
        scene_names_sha256=sha(HERE/'SCENE_NAMES.json'),
        analysis_sources_sha256={name:sha(HERE/name)for name in ('statistic_rules.py','reporting.py','oracle.py',
            'tuning.py','formal_tests.py','time_limit_sensitivity.py','relaxed_sensitivity.py','classical_appendix.py','trace_time_diagnostic.py',
            'video_delivery.py','video_tools/batch_videos_generic.py')},
        primary=dict(benchmark='A',scenes=split['benchmark_A']['test'],states=split['benchmark_A']['test_states'],
            endpoint='Frozen learned correction success averaged over six nominal models, minus external global-training-tuned success, paired by physical initial state.',
            methods=active_methods(),family_size=3,
            withdrawn_methods=read(HERE/'METHOD_EXECUTION_POLICY.json')['withdrawn_methods'],
            native_excluded_methods=[m for m,r in read(HERE/'NATIVE_REPRODUCTION_RESULTS.json')['methods'].items()if not r['reproducible']],
            paired_unit='scene/state; six nominal controllers retained as one block'),
        intervals=dict(method='Scene resampling within fixed category, then physical-state resampling within scene; nominal models remain paired.',
                       resamples=10000,seed=2026102905,percentiles=[.025,.975]),
        hypothesis_test=dict(method='Two-sided paired scene-level sign flips; exact for<=18scenes, otherwise100000draws.',
            draws=100000,seed=2026102906,multiple_comparisons='Holm correction across the three author-code methods retained by the2026-10-30addendum; any native-reproduction exclusion has no fabricated p value.',
            superiority='Holm p<0.05 AND interval lower bound>0',
            noninferiority='Interval lower bound>-0.05',otherwise='无法区分'),
        descriptive_only=['author_original','validation','benchmark_B','type_tuned','Oracle_prefix16','mechanism_disabled'],
        nonmain_sensitivity=dict(
            relaxed='All archived v1/v2type10corridor and type16warehouse calibration geometries, including positive widening and zero-widening reference,states4–11,literal original horizons; frozen BenchmarkBglobal choices only. Missing frozen admission geometry unavailable, never fabricated.',
            time_limit='All primary-geometry BenchmarkAtest timeout rows, all method variants. For available saved traces: float64 positive least-squares slope of average progress over final20%of samples; use route-remaining if recorded, otherwise true-goal Euclidean distance. Re-run once with2xhorizon and explicit parent RNG identity, same frozen parameters. Missing traces have unknown eligibility. Main table unchanged.'),
        trajectory_diagnostics='Observed successful traces only: compare matched completion time, waiting fraction(speed<0.01m/s while goal error>0.08m), active speed and travelled/initial goal distance. Fold nominal pairs within physical state. Descriptive association only, no causal claim or extra rollout to fill old traces.',
        times=dict(units='simulated seconds',
            success_conditional='Mean completion time of successful episodes only, with success denominator. Historical missing times remain missing unless authorized full backfill completes.',
            all_queue_capped='Success completion time; every non-success gets frozen scene horizon. Average nominal models within physical state before aggregation.',
            matched_success='For each physical state where external succeeds and at least one nominal succeeds: mean ours-minus-external time over successful nominal pairs; then mean equally across physical states. Report physical count and nominal-pair count.',
            failure_actual_end='No imputation from capped times. Unavailable historical termination/CPU data stay missing.',
            compute='Measured one-CPU per-step median/p95; report sample coverage, setup, child import/exit and generator/critic startup separately. Frozen lookup does not imply measured zero inference cost.'),
        missing=dict(primary='No imputation as success/failure; report completeness and matched available units.',
            sensitivity='Missing results as non-success, clearly separated from primary. Withdrawn methods are excluded; unreproduced author-code methods are documented without fabricated comparative tests.'),
        oracles=dict(optional=True,max_candidates=16,order='Frozen prefix, stop at first success.',
            scope='Finite pool only; tuning may select17–32after saturation and then need not be contained in prefix16.',
            priority='Only after all active mainline steps and reports; main C1 family and active paper methods first, original four family last.'),
        applicability='At least95%aligned-author-default solo successes per scene; geometry-invalid state0 is unavailable rather than controller failure. Preserve primary scenes; applicable-only appendix separate.',
        numerical='Solver nonoptimal returns, first unhandled native exception, final unified outcome and infrastructure missingness are separate.',
        inherited_limits=['Representative17geometry informed admission design; scene-level design leakage on B.',
            'One test map per type in B: intervals describe physical initial-state variation conditional on these fixed maps, not uncertainty over a map population.',
            'Validation informed prior learned method design; descriptive only.',
            'Native paper controllers are documented common-plant ports.',
            'Consistent wall-collision variants follow CONSISTENT_BOX_SCENES.json and the predefined90%legal-state rule; learned models trained on original geometry semantics.',
            'Old learned logs omit times/traces; new reference replays and conditional timing backfill do not create historical timestamps.'],
        videos=dict(optional=True,selection=videos,source='Existing raw simulated traces only; unavailable trace is a gap, never synthesized.',
                    methods='Frozen learned correction plus active paper global-tuned methods; withdrawn methods excluded.',
                    renderer='Private historical shared-renderer copy; generic heterogeneous-scene schema, every saved state once at1/dtFPS, exact frame-count verification. Legacy four-scene/deadlock-role gate is not claimed applicable.'),
        timing_backfill_policy=read(HERE/'USER_TIME_BACKFILL_POLICY.json'))
    dump(path,value)
    with(HERE/'STATUS.md').open('a')as stream:stream.write('\n测试前分析计划SHA256：'+sha(path)+'\n')
    return value


if __name__=='__main__':main()
