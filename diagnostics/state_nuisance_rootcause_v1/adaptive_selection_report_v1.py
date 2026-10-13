"""Human-readable report and scientific figure for the held-out Q audit."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'heldout_adaptive_selection_v1'


def read(name):
    return json.loads((OUT / name).read_text())


def pct(number):
    return f'{100 * number:.2f}%'


def interval(values, scale=100):
    if values is None:
        return 'NA'
    return f'[{values[0] * scale:.2f}, {values[1] * scale:.2f}]'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    result = read('audit_results.json')
    verdict = read('verdict.json')
    protocol = read('protocol.json')
    natural = result['natural_eta_preference']
    cohorts = result['cohorts']
    reversal = result['reversal']
    choices = result['selection_behavior']
    collection = result['dataset_audit']
    plan = result['definitions']
    amendments = read('implementation_amendments.json')
    merge_fix = read('merge_fix_record.json')
    lines = [
        'STRICT HELD-OUT ADAPTIVE-SELECTION AUDIT',
        f"FINAL VERDICT: {verdict['label']}",
        verdict['reason'],
        '',
        '1. DATA GENERATION AND SAMPLING AUDIT',
        f"{len(protocol['states'])} wholly new, independent Ring true-t0 physical states from the unchanged native development generator.",
        f"Unfiltered contiguous draw seeds: {protocol['states'][0]['metadata']['draw_seed']}..{protocol['states'][-1]['metadata']['draw_seed']}.",
        'All states and all 16 eta x16 future seeds were frozen before any critic prediction or rollout analysis.',
        'No screening by critic, failures, prior preference region, or observed outcome; no overlap with earlier608 states or DB state IDs.',
        'Controller/Flow/safety projection/eta bank/training data/checkpoints are unchanged; no critic retraining.',
        f"{collection['attempted']} real FF rollout attempts; {collection['successes']} successes, {collection['failures']} valid failures, {collection['numerical_unknown']} numerical unknowns.",
        'All attempted state-eta-seed records agree with the authoritative shared rollout DB.',
        '',
        '2. FROZEN DISCOVERY/EVALUATION SPLIT AND RULES',
        f"Discovery seeds {plan['discovery_seed_indices']}; evaluation seeds {plan['evaluation_seed_indices']}.",
        f"Primary adaptive-demand gap={plan['adaptive_demand_primary_gap']:.3f}=3/8; sensitivities 2/8 and4/8; strong pair reversal gap=3/8.",
        'Discovery uses success lower and upper bounds; unknown numerical outcomes are not recoded as failures.',
        'Ties in eta preference and model argmax use the smallest eta index.',
        'The global fixed baseline comes from aggregate discovery outcomes; the eta-only model was frozen from TRAIN512.',
        'All policy scores below use only the independent evaluation half. Physical state is the bootstrap unit (5000 draws).',
        f"One analysis-code amendment during rollout collection added actual top-1 choice distributions for every strong reversal pair; it did not change grouping, thresholds or primary metrics: {amendments}.",
        f"A post-rollout DB merge-only placeholder fix was recorded before analysis; original rollout producer and analysis rules remained frozen: {merge_fix}.",
        '',
        '3. NATURAL ETA PREFERENCE DISTRIBUTION',
        f"TRAIN Q(eta) selects eta {natural['qeta_train_selected_eta']}; discovery-aggregate best fixed eta {natural['global_best_fixed_discovery_eta']}.",
        f"Discovery preferred eta counts by index: {natural['preferred_eta_frequency']}.",
        f"States with preferred eta different from discovery best fixed: {natural['preferred_differs_from_fixed_count']}/256; alternative preference eta indices: {natural['preferred_differs_from_fixed_distinct_etas']}.",
        f"States with tied empirical best eta by number of tied candidates: {natural['preferred_eta_tie_count_frequency']}.",
        f"Adaptive-demand counts by threshold: {[(k, v['count']) for k, v in natural['adaptive_demand_sensitivity'].items()]}.",
        f"Primary adaptive-demand preferred eta counts: {natural['adaptive_demand_preferred_eta_frequency']}.",
        f"Strong reversal pairs naturally present: {reversal['strong_reversal_pairs']}/120; participating states: {reversal['strong_reversal_states']}; pair-state cases: {reversal['strong_pair_state_cases']}.",
        'all_120_eta_pairs.csv records every pair and both discovery direction counts.',
        '',
        '4. ALL-STATE, ADAPTIVE-DEMAND, AND STRONG-REVERSAL SELECTION RESULTS',
        'Cohort | selector | states | success lower/upper | valid mean Q | oracle regret | eval-best match | pairwise ranking agreement',
    ]
    labels = {'qeta': 'TRAIN Q(eta)', 'best_fixed': 'discovery best fixed',
              'qstate512': 'Q(s,eta) TRAIN512', 'qstate32': 'Q(s,eta) TRAIN32 diagnostic'}
    for cohort, row in cohorts.items():
        if row['states'] == 0:
            lines.append(f"{cohort} | 0 | No state satisfied the frozen rule")
            continue
        for selector in ('qeta', 'best_fixed', 'qstate512', 'qstate32'):
            metric = row['selectors'][selector]
            lines.append(f"{cohort} | {labels[selector]} | {row['states']} | "
                         f"{pct(metric['top1_rollout_success_lower'])}/{pct(metric['top1_rollout_success_upper'])} | "
                         f"{metric['mean_ground_truth_q_selected_valid']:.4f} | "
                         f"{metric['oracle_regret_finite_eval_lower']:.4f} | "
                         f"{pct(metric['matches_eval_best_eta'])} | "
                         f"{pct(metric['pairwise_preference_ranking_agreement']['mean'])}")
        for name, comp in row['comparisons'].items():
            lines.append(f"{cohort} | {name}: mean={pct(comp['mean'])}; paired state 95% CI in percentage points={interval(comp['paired_physical_state_bootstrap_95'])}; fit-wise={comp['by_fit']}")
        fixed = row['all_fixed']
        lines.append(f"{cohort} | Q512 minus hindsight best of all16 fixed policies: {pct(fixed['adaptive_minus_hindsight_best_fixed_point'])}, bootstrap max-over-16 95% CI pp={interval(fixed['simultaneous_max_over_16_fixed_state_bootstrap_95'])}, hindsight eta={fixed['hindsight_eval_best_fixed_eta_descriptive']}.")
    lines += ['', '5. STRONG REVERSAL AND SWITCHING TEST',
              f"Strong reversal threshold is the same 3/8 discovery lower-vs-upper margin in both directions of a pair.",
              f"Evaluation of all {reversal['strong_pair_state_cases']} discovery cases: {reversal.get('independent_eval_decisive_cases', 0)} decisive, {reversal.get('independent_eval_confirmed_cases', 0)} confirmed, {reversal.get('independent_eval_contradicted_cases', 0)} contradicted, {reversal.get('independent_eval_ambiguous_cases', 0)} ambiguous."]
    if reversal['models']:
        for selector in ('qeta', 'best_fixed', 'qstate512', 'qstate32'):
            row = reversal['models'][selector]
            lines.append(f"{labels[selector]}: independent evaluation switch accuracy={pct(row['independent_eval_switch_accuracy'])}, state bootstrap 95% CI={interval(row['independent_eval_switch_accuracy_state_bootstrap_95'])}, three fit values={row['independent_eval_switch_accuracy_by_fit']}, pair fraction correct majority on both discovery groups={row['pair_fraction_majority_correct_on_both_discovery_groups_by_fit']}.")
        for name, row in reversal['paired_switch_accuracy_differences'].items():
            lines.append(f"{name} switch accuracy gain={pct(row['mean'])}; paired state CI pp={interval(row['paired_physical_state_bootstrap_95'])}.")
        for baseline, values in reversal['adaptive_gain_by_reversal_side'].items():
            for side, stats in values.items():
                lines.append(f"Q512 minus {baseline} on discovery side {side}: {pct(stats['mean'])}; state CI pp={interval(stats['paired_physical_state_bootstrap_95'])}; {stats['case_count']} cases / {stats['distinct_states']} states.")
        lines.append(f"Pairs with positive adaptive gain on BOTH sides: {reversal['pairs_with_positive_qstate512_gain_on_both_sides_vs_qeta']} versus Qeta; {reversal['pairs_with_positive_qstate512_gain_on_both_sides_vs_fixed']} versus discovery fixed, out of {reversal['strong_reversal_pairs']} pairs.")
        lines.append(f"Representative pair chosen by discovery support alone: {reversal['representative_pair_selected_only_by_discovery_support']}.")
    lines += ['', '6. ALL-FIXED-ETA FALSIFICATION',
              'Every one of the16 fixed eta policies was evaluated on the same held-out states; evaluation-best fixed eta is descriptive and never constructs a selector.',
              'Full natural cohort: eta index | fixed success lower | Q512 minus fixed | paired physical-state 95% CI (percentage points)']
    for row in cohorts['all']['all_fixed']['all_16_policies']:
        lines.append(f"{row['eta_index']} | {pct(row['success_lower'])} | {pct(row['adaptive_minus_this_fixed'])} | {interval(row['paired_physical_state_bootstrap_95'])}")
    lines += ['', '7. STATE-DEPENDENT SELECTION AND 32-STATE DIAGNOSTIC',
              'Frequencies, entropy and two-state eta-change rates below are listed for each frozen optimizer seed.']
    for selector in ('qeta', 'best_fixed', 'qstate512', 'qstate32'):
        behavior = choices[selector]
        lines.append(f"{labels[selector]}: {behavior['by_fit']}")
    lines += ['', '8. FAILURE CASES',
              'Twelve most negative state-level Q512-versus-Qeta evaluation differences, ties then ranked by oracle regret:']
    for row in result['worst_failure_cases']:
        lines.append(str(row))
    lines += ['', '9. INTERPRETATION AND EVIDENCE BOUNDARY']
    lines += verdict['evidence']
    lines += result['interpretation_limits']
    lines += ['', '10. RAW REPRODUCIBILITY',
              'protocol.json and analysis_plan.json freeze all state/eta/seed identities, rules and hashes.',
              'fresh_rollouts.npz is the complete [256,16,16,3] real outcome tensor; raw/*.json contains each individual original rollout.',
              'seed_outcomes.csv and state_eta_results.csv expose every seed and state/eta count; frozen_predictions.npz contains all critic logits.',
              'audit_results.json contains all numerical metrics and confidence intervals.',
              'No video was requested or produced for this Q audit.']
    (OUT / 'FINAL_REPORT.txt').write_text('\n'.join(lines) + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    frequency = np.asarray(natural['preferred_eta_frequency'])
    axes[0].bar(np.arange(16), frequency, color='#4477AA')
    axes[0].axvline(natural['global_best_fixed_discovery_eta'], color='#CC6677', ls='--',
                    label='Discovery best fixed')
    axes[0].set(title='Natural discovery eta preference', xlabel='Eta index', ylabel='Physical states')
    axes[0].legend(fontsize=8)
    full = cohorts['all']
    fixed_rates = [v['success_lower'] for v in full['all_fixed']['all_16_policies']]
    axes[1].bar(np.arange(16), fixed_rates, color='#999999')
    axes[1].axhline(full['selectors']['qstate512']['top1_rollout_success_lower'],
                    color='#228833', label='Adaptive Q512')
    axes[1].set(title='All 16 fixed policies, held-out evaluation', xlabel='Fixed eta index',
                ylabel='Success lower bound', ylim=(0, 1.02))
    axes[1].legend(fontsize=8)
    for cohort, color in [('all', '#4477AA'), ('adaptive_demand', '#CC6677'),
                          ('strong_reversal', '#228833')]:
        row = cohorts[cohort]
        if not row['states']:
            continue
        left = row['comparisons']['qstate512_minus_qeta']
        y = ('all', 'adaptive_demand', 'strong_reversal').index(cohort)
        ci = left['paired_physical_state_bootstrap_95']
        axes[2].errorbar(left['mean'], y, xerr=[[left['mean']-ci[0]], [ci[1]-left['mean']]],
                         fmt='o', color=color, capsize=4, label=f"{cohort} (n={row['states']})")
    axes[2].axvline(0, color='black', lw=.8)
    axes[2].set(yticks=[0, 1, 2], yticklabels=['All', 'Adaptive demand', 'Strong reversal'],
                xlabel='Q512 minus Q(eta) success', title='Paired physical-state bootstrap')
    axes[2].legend(fontsize=8, loc='best')
    fig.suptitle('Held-out Q adaptive selection; 256 globally sampled states; discovery/evaluation seeds 8+8')
    fig.savefig(OUT / 'adaptive_selection_audit.png', dpi=180)
    fig.savefig(OUT / 'adaptive_selection_audit.pdf')
    plt.close(fig)
    print('Wrote FINAL_REPORT.txt and scientific figures.')


if __name__ == '__main__':
    main()
