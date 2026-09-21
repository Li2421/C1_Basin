"""Report boundary coverage in addition to the unchanged two acceptance gates."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_c1_event_history import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    folder = parser.parse_args().folder
    result = analyze(folder)
    diagnostics = json.loads((folder/'diagnostics.json').read_text())
    rows = json.loads((folder/'records.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    assert len(protocol['prediction_seeds']) == 8
    assert protocol['prediction_gradient_rollouts'] == 250
    assert len(diagnostics) == 25
    active = {d['rid'] for d in diagnostics if d['gradient_norm'] > 0}
    base_deadlocks = [r for r in rows if r['arm'] == 'reference' and r['either_deadlock']]
    weights = np.array([d['importance_weights'] for d in diagnostics])
    assert np.isfinite(weights).all() and ((weights > 0) & (weights <= 2+1e-12)).all()
    result['integration'] = dict(
        pilot_rollouts=50, importance_prediction_rollouts=200,
        total_prediction_gradient_rollouts=250,
        paired_cases=50, physical_replays=len(rows),
        target_boundary_hits=sum(d['boundary_hits_target'] for d in diagnostics),
        target_draws=100,
        shifted_boundary_hits=sum(d['boundary_hits_shifted'] for d in diagnostics),
        shifted_draws=100,
        nonzero_gradient_starts=len(active),
        original_deadlock_cases_with_nonzero_gradient=sum(r['rid'] in active for r in base_deadlocks),
        total_original_deadlock_cases=len(base_deadlocks),
        weight_min=float(weights.min()), weight_max=float(weights.max()),
        weight_mean=float(weights.mean()),
        batch_ess_range=[min(d['importance_ess'] for d in diagnostics), max(d['importance_ess'] for d in diagnostics)],
        calibration_failures=[dict(rid=d['rid'], arm=a, rms=v['rms'])
                              for d in diagnostics if d['rid'] in active
                              for a,v in d['calibration'].items() if not v['matched']],
        note='Zero-gradient starts have zero interventions in all three arms, as preregistered. '
             'No gradient direction is claimed there. A missing hit does not prove zero population derivative.')
    (folder/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('gates','arms','integration')}, indent=2))


if __name__ == '__main__':
    main()
