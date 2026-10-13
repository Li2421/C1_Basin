"""Require every FF/TT and LOSO result, then render final comparative report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics.physical_state_critic_formal_v1.evaluate import PANELS

HERE=Path(__file__).resolve().parent
VERDICTS=(
 'STATE_CONDITIONING_CONSISTENTLY_VALIDATED',
 'WITHIN_SCENE_VALIDATED_LOSO_MIXED',
 'LOSO_VALIDATED_WITHIN_SCENE_MIXED',
 'STATE_SIGNAL_REAL_BUT_SELECTION_GAIN_LIMITED',
 'GENERALIZATION_FAILURE_REMAINS',
)


def read(path):return json.loads(Path(path).read_text())
def pct(x):return f'{100*x:+.2f} pp'
def frac(x):return f"{x['B15_confirmed']}/128"+(f" (possible {x['B15_possible']}/128)" if x['B15_unresolved'] else '')
def ci(row):
    a,b=row['ci95'];return f'[{100*a:+.2f}, {100*b:+.2f}] pp'


def main(verdict,rationale):
    assert verdict in VERDICTS and rationale.strip()
    rows={p:read(HERE/'evaluation'/p/'metrics.json') for p in PANELS}
    assert len(rows)==10
    matched=read(HERE/'matched_protocol.json');loso=read(HERE/'loso_protocol.json')
    frozen=read(HERE/'all_models_frozen.json')
    assert frozen['test_labels_used'] is False
    assert all(r['test_states']==128 for r in rows.values())
    lines=['# Formal physical-only Q(s,η) critic evaluation','',
           f'**Verdict: {verdict}.** {rationale.strip()}','',
           'This report separates within-scene generalization from four-scene FF leave-one-scene-out (LOSO). '
           'The critic receives only physical state and η; labels use frozen Flow, controller, safety projection, '
           'horizon and success evaluator. Eta-only critics use the same Q4 labels and candidate sets.','',
           '## Protocol and sampling audit','',
           '- Four independent global physical-state panels: 512 TRAIN, 32 VAL and 128 fresh TEST states per scene; '
           'state files and source hashes are in this directory. DB retains the original accepted-global-state expert check. '
           'All candidate proposals and IDs were frozen before test labels.',''
           '- The historical Toy/Ring matched TT/FF panels share their fixed Sobol K16 bank; DB/Four within-scene and '
           'the four LOSO folds share the historical scenario-specific stochastic K16 generator. Every selector and oracle '
           'sees the same 16 candidates for a given state.','',
           '- TRAIN uses four fixed rollout seeds per state–η. VAL/TEST use 16 matched seeds and B15 means at least 15 '
           'successful rollouts. Safety uses the same states and future seeds. Each attempted numerical failure remains '
           'unknown and is reported as a success/B15 bound.','',
           '- Physical-only input: Ring matched uses the validated physical23 encoder; other fits use the approved '
           'masked physical-entity adapter with Flow/controller/context channels zero. No scene ID enters LOSO; only '
           'source-scene TRAIN and VAL labels select LOSO checkpoints.','',
           '- Checkpoints were selected before TEST by lowest validation state-level selection regret, with decisive '
           'pairwise ranking and NLL as ordered tie-breaks. Bootstrap samples independent physical states, not rollouts.','',
           '## Table A — within-scene B15 top-1 success','',
           '| Scene | Placement | Safety | Q(η) | Q(s,η) | Oracle | Δ state–eta | 95% state bootstrap CI |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    labels={'toy_give_way':'Toy Give-Way','double_bottleneck':'Double-Bottleneck',
            'four_way_intersection':'Four-Way','ring_exchange':'Ring'}
    for panel,r in rows.items():
        if not panel.startswith('within_'):continue
        m=r['methods'];d=r['paired_comparisons']['Q_state_minus_Q_eta']['B15_confirmed']
        lines.append(f"| {labels[r['scene']]} | {r['placement']} | {frac(m['Safety'])} | {frac(m['Q_eta'])} | {frac(m['Q_state'])} | {frac(m['Oracle'])} | {pct(d['mean'])} | {ci(d)} |")
    lines+=['','## Table B — FF leave-one-scene-out B15 top-1 success','',
            '| Held-out scene | Safety | Q(η) | Q(s,η) | Oracle | Δ state–eta | 95% state bootstrap CI |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for panel,r in rows.items():
        if not panel.startswith('loso_'):continue
        m=r['methods'];d=r['paired_comparisons']['Q_state_minus_Q_eta']['B15_confirmed']
        lines.append(f"| {labels[r['scene']]} | {frac(m['Safety'])} | {frac(m['Q_eta'])} | {frac(m['Q_state'])} | {frac(m['Oracle'])} | {pct(d['mean'])} | {ci(d)} |")
    lines+=['','## Mean rollout success, oracle regret, collision','',
            '| Experiment | Method | Mean Q lower–upper | Oracle regret lower–upper | Collisions / 2048 seed rollouts |',
            '|---|---|---:|---:|---:|']
    for panel,r in rows.items():
        for method,v in r['methods'].items():
            lines.append(f"| {panel} | {method} | {v['mean_success_lower']:.4f}–{v['mean_success_upper']:.4f} | "
                         f"{v['oracle_regret_lower']:.4f}–{v['oracle_regret_upper']:.4f} | {v['collisions']} |")
    lines+=['','## Table C — state-conditioning diagnostics','',
            '| Experiment | State-effect corr. | Reversal Q(η) | Reversal Q(s,η) | Local pairwise Q(η) / Q(s,η) | Selection entropy (bits) |',
            '|---|---:|---:|---:|---:|---:|']
    for panel,r in rows.items():
        z=r['diagnostics'];rv=z['strong_reversal']
        corr='n/a' if z['state_effect_correlation'] is None else f"{z['state_effect_correlation']:.3f}"
        er='n/a' if rv is None or not rv['states'] else f"{rv['eta_only_accuracy']['mean']:.3f} (N={rv['states']})"
        sr='n/a' if rv is None or not rv['states'] else f"{rv['state_accuracy']['mean']:.3f} (N={rv['states']})"
        pair=f"{z['local_pairwise_eta_only']['accuracy']:.3f} / {z['local_pairwise_state']['accuracy']:.3f}" if z['local_pairwise_eta_only']['accuracy'] is not None and z['local_pairwise_state']['accuracy'] is not None else 'n/a'
        lines.append(f"| {panel} | {corr} | {er} | {sr} | {pair} | {z['selection_entropy_bits']:.2f} |")
    lines+=['','For stochastic K16 panels, candidate index j denotes a different η in each state. '
            'The fixed-η state-effect and cross-state reversal definitions are therefore not identifiable from that panel; '
            'the local within-state pairwise diagnostic is still reported. Candidate-index entropy there is slot entropy, '
            'not evidence of η switching.','',
            '## Historical comparison and interpretation','',
            'The archived strict four-scene LOSO reports found no stable state-aware advantage over matched source-only '
            'eta-only baselines: the original shared model selected B15 in Toy 30/48 versus 32/48, DB 20/24 versus '
            '23/24, Four-Way 21/24 plus three unresolved versus 21/24 plus two unresolved, and Ring 0/60 versus 17/60. '
            'The later partial-count repair likewise reported no positive primary fold. These older targets, labels, '
            'candidate pools, input context and checkpoints differ from this new fresh-state experiment, so numerical '
            'differences are not paired causal effects of removing context. The old strict reports do not support a '
            'claim that LOSO had already been validated.','',
            '## Paired Safety comparisons and oracle headroom','',
            '| Experiment | Δ state–Safety B15 | 95% CI | Δ state–Safety mean Q lower | 95% CI | Oracle recovery from Safety / eta-only |',
            '|---|---:|---:|---:|---:|---:|']
    for panel,r in rows.items():
        d=r['paired_comparisons']['Q_state_minus_Safety'];b=d['B15_confirmed'];q=d['mean_success_lower']
        rec=r['oracle_recovery'];a=rec['from_Safety'];c=rec['from_Q_eta']
        recovery=(f'{a:.2f}' if a is not None else 'n/a')+' / '+(f'{c:.2f}' if c is not None else 'n/a')
        lines.append(f"| {panel} | {pct(b['mean'])} | {ci(b)} | {pct(q['mean'])} | {ci(q)} | {recovery} |")
    lines+=['','## Failure cases and evidence boundaries','',
            'Each panel has `evaluation/<panel>/metrics.json` with its twelve highest-regret physical states and '
            '`selected_state_rows.csv` with every selector’s per-state choice, B15 status, success, numerical and collision '
            'counts. Full per-state/per-η/per-seed outcomes remain in the panel `outcomes.npz`, with raw seed JSON and '
            'journal records retained. Any B15 uncertainty is displayed as a confirmed/possible interval; it is not '
            'silently counted as an observed failure.','',
            'The historical matched TT/FF scope was Toy and Ring only. DB and Four-Way within-scene FF use their '
            'historical stochastic generator panel; these are distinct candidate mechanisms and are not averaged with '
            'the fixed-bank TT/FF panel. The LOSO conclusion refers only to FF.','',
            '## Frozen artifact pointers','',
            '- `matched_protocol.json`, `loso_protocol.json`, `unified_train_plan.json`',
            '- `states_*.json`, `loso/*/*/proposals.json`, panel `cases.json` and `planned_rollouts.json`',
            '- `fits/*/models_frozen.json`, `ring_ff_models_frozen.json`, `all_models_frozen.json`',
            '- `evaluation/*/metrics.json`, `evaluation/*/selected_state_rows.csv`',
            '- `matched_merge_audit.json`, `loso_merge_audit.json`, and the raw rollout files','']
    target=HERE/'FINAL_REPORT.md'
    target.write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(target),verdict=verdict,panels=len(rows))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--verdict',choices=VERDICTS,required=True)
    p.add_argument('--rationale',required=True)
    a=p.parse_args();main(a.verdict,a.rationale)
