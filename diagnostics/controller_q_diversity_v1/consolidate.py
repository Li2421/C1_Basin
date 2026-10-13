"""Collect completed frozen confirmation analyses without selecting new models.

This is a reporting layer. It refuses incomplete experiments and never opens
rollout labels, trains models, changes rankings, or chooses an experimental arm.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return list(csv.DictReader(path.open()))


def write_csv(path, values):
    keys = list(dict.fromkeys(k for row in values for k in row))
    with path.open('w', newline='') as f:
        out = csv.DictWriter(f, keys)
        out.writeheader()
        out.writerows(values)


def number(value):
    return float(value) if value not in ('', None, 'None') else None


def mean_field(group, key):
    values = [number(r.get(key)) for r in group]
    return sum(values) / len(values) if all(v is not None for v in values) else None


def run():
    panels = {'joint': ROOT, 'toy_source': ROOT/'confirmation_scene_transfer/toy',
              'ring_source': ROOT/'confirmation_scene_transfer/ring'}
    required = [v/'evaluation_complete.json' for v in panels.values()]
    required += [ROOT/'failure_audit.json', ROOT/'interaction_decomposition.json',
                 ROOT/'fixed_falsification_bounds.json',
                 ROOT/'checks/operational_target/evaluation_complete.json']
    missing = [str(v) for v in required if not v.exists()]
    if missing:
        raise SystemExit('Incomplete; no consolidation: ' + ', '.join(missing))
    selection, paired, mechanism, per_controller = [], [], [], []
    metadata = {}
    for panel, folder in panels.items():
        meta = read(folder/'evaluation_complete.json')
        metadata[panel] = meta
        primary = meta['primary']
        methods = {'eta_only', 'state_only', primary, 'Safety', 'Oracle',
                   'controller_mismatch', 'state_mismatch_exact',
                   'TRAIN_fixed', 'discovery_fixed', 'evaluation_posthoc_fixed'}
        table = rows(folder/'selection_tables.csv')
        scenes = sorted(set(r['scene'] for r in table))
        for scene in scenes:
            source = meta.get('source_scene')
            scope = ('known-scene/new-controller' if source is None or source == scene
                     else 'held-scene/new-controller')
            for method in sorted(methods):
                group = [r for r in table if r['scene'] == scene and r['method'] == method]
                if not group:
                    continue
                assert len(group) == 2, (panel, scene, method, len(group))
                assert {r['controller'] for r in group} == {'analytic6', 'analytic7'}
                count = sum(number(r['B15_count']) for r in group)
                n = int(group[0]['physical_states'])
                selection.append(dict(panel=panel, scope=scope, scene=scene,
                    method=method, physical_states=n, controllers=2,
                    B15_state_controller_cases=count, total_state_controller_cases=2*n,
                    B15_rate=mean_field(group, 'B15_rate'),
                    evaluation8_success=mean_field(group, 'evaluation8_success'),
                    oracle_regret=mean_field(group, 'oracle_regret'),
                    Q_MSE_known=mean_field(group, 'Q_MSE_known_outcomes'),
                    collision_rate=mean_field(group, 'selected_collision_rate'),
                    selected_numerical_mean_per_controller=mean_field(group, 'selected_numerical_count')))
                per_controller.extend(dict(panel=panel, scope=scope, **r) for r in group)
        paired.extend(dict(panel=panel, **r) for r in rows(folder/'paired_comparisons.csv'))
        mechanism.extend(dict(panel=panel, **r) for r in read(folder/'mechanism_metrics.json')
                         if r['model'] in {primary, 'eta_only', 'state_only', 'controller_mismatch'})
    write_csv(ROOT/'consolidated_selection.csv', selection)
    write_csv(ROOT/'consolidated_per_controller.csv', per_controller)
    write_csv(ROOT/'consolidated_paired_comparisons.csv', paired)
    (ROOT/'consolidated_mechanism.json').write_text(json.dumps(mechanism, indent=2)+'\n')
    digest = dict(panels=metadata, selection=selection,
        safety=read(ROOT/'failure_audit.json'),
        fixed_falsification=read(ROOT/'fixed_falsification_bounds.json'),
        operational_target=read(ROOT/'checks/operational_target/paired_comparisons.json'),
        artifacts=['consolidated_selection.csv', 'consolidated_per_controller.csv',
                   'consolidated_paired_comparisons.csv', 'consolidated_mechanism.json',
                   'interaction_decomposition.json', 'failure_cases.csv'],
        limitations=[
            'N is 32 physical states per scene, not 64 independent state/controller cases.',
            'Intervals condition on two held analytic controllers; no controller-population CI.',
            'Joint is known-scene/new-state/new-controller, not cross-scene LOSO.',
            'Single-source fits use source-only labels/norm/checkpoint selection, but both development scenes informed representation design.',
            'Operational success and unresolved physical outcomes have distinct estimands.',
            'No four-scene FF LOSO or unseen native Flow family claim is made here.'])
    (ROOT/'consolidated_digest.json').write_text(json.dumps(digest, indent=2)+'\n')
    print('Consolidated all completed panels; no model selection or new rollout.')


if __name__ == '__main__':
    run()
