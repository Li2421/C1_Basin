"""Select old strict-LOSO cases by exact cached outcome, without new Q evaluation."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from shared_rollout_db.src.rollout_db import canonical, eta_identity

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HIST = ROOT / 'diagnostics/orthoflow3_loso_root_cause_v1'
STRICT = ROOT / 'diagnostics/physical_state_critic_minimal_v1/strict'
DB = ROOT / 'shared_rollout_db/rollout.sqlite'

# Paired reversals were selected from the already-opened historical labels.
# These indexes are a visualization sample, not a fresh test-set analysis.
CASES = [
    ('toy', 'within', 15, 'success', None),
    ('toy', 'loso', 15, 'deadlock', None),
    ('toy', 'within', 20, 'deadlock', None),
    ('toy', 'loso', 20, 'success', None),
    ('db', 'within', 6, 'success', 2),
    ('db', 'loso', 6, 'timeout', 2),
    ('db', 'loso', 1, 'success', 0),
    ('db', 'safety_baseline', 6, 'timeout', None),
    ('four', 'within', 0, 'success', 0),
    ('four', 'loso', 0, 'success', 0),
    ('four', 'safety_baseline', 0, 'timeout', None),
    ('ring', 'within', 12, 'timeout', 0),
    ('ring', 'loso', 12, 'success', 0),
    ('ring', 'within', 16, 'success', 0),
    ('ring', 'loso', 16, 'timeout', 0),
]


def main():
    OUT.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    result = []
    requests = []
    for scene, panel, index, wanted, preferred_seed in CASES:
        manifest = json.loads((HIST / 'targets' / scene / 'manifest.json').read_text())
        source = manifest[index]
        if panel == 'safety_baseline':
            eta = [0.0, 0.0, 0.0]
            eta_uid = eta_identity(eta)[0]
            controllers = con.execute('SELECT controller_uid FROM rollout WHERE state_uid=? AND eta_uid=? AND outcome=? AND compatibility_quality=? AND conflict_quarantined=0 GROUP BY controller_uid',
                                      (source['state_uid'], eta_uid, wanted, 'EXACT_REUSE')).fetchall()
            if len(controllers) != 1:
                raise RuntimeError(f'{scene} baseline controller ambiguity: {controllers}')
            controller_uid = controllers[0][0]
            selected_index = None
        else:
            pred_path = STRICT / f'{panel}_{scene}' / 'predictions.npz'
            with np.load(pred_path, allow_pickle=False) as pred:
                assert str(pred['state_uids'][index]) == source['state_uid']
                selected_index = int(pred['state_choice'][index])
            eta = source['eta'][selected_index]
            eta_uid = eta_identity(eta)[0]
            controller_uid = source['controller_uid']
        rows = con.execute('SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND outcome=? AND compatibility_quality=? AND conflict_quarantined=0 AND numerical_failure=0',
                           (source['state_uid'], eta_uid, controller_uid, wanted, 'EXACT_REUSE')).fetchall()
        options = []
        for row in rows:
            key = json.loads(row['seed_key'])
            if 'future_index' in key:
                options.append((int(key['future_index']), row))
        options.sort(key=lambda x: (x[0] != preferred_seed if preferred_seed is not None else False, x[0]))
        if not options:
            raise RuntimeError(f'No exact cached {wanted}: {scene}/{panel}/{index}')
        seed, row = options[0]
        name = f'{scene}_{panel}_state{index:02d}_{wanted}'
        result.append({'name': name, 'scene': scene, 'panel': panel, 'state_index': index,
                       'state_uid': source['state_uid'], 'selected_index': selected_index,
                       'eta': eta, 'eta_uid': eta_uid, 'controller_uid': controller_uid,
                       'future_index': seed, 'expected': {'outcome': wanted,
                           'episode_length': int(row['episode_length']),
                           'rollout_uid': row['rollout_uid'],
                           'original_source_file': row['original_source_file']},
                       'prediction_file': str(STRICT / f'{panel}_{scene}' / 'predictions.npz') if panel != 'safety_baseline' else None})
        requests.append({'state_uid': source['state_uid'], 'eta_uid': eta_uid,
                         'controller_uid': controller_uid,
                         'seed_keys': [canonical({'future_index': seed})]})
    (OUT / 'selection.json').write_text(json.dumps({'cases': result, 'source_report': str(STRICT / 'REPORT.md'),
        'selection_note': 'Historical-label visualization sample; no new model selection or Q estimate.'}, indent=2) + '\n')
    (OUT / 'planned_rollouts.json').write_text(json.dumps({'requests': requests}, indent=2) + '\n')
    print(json.dumps({'selected': len(result), 'cases': [(x['name'], x['future_index'], x['expected']['episode_length']) for x in result]}, indent=2))


if __name__ == '__main__':
    main()
