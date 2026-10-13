"""Reproducibility and one-fold held-label firewall check."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import ablate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    receipt = []
    for line in (HERE / 'frozen_hashes.sha256').read_text().splitlines():
        digest, filename = line.split(None, 1)
        filename = filename.strip()
        path = ROOT / filename
        assert ablate.sha(path) == digest, filename
        receipt.append(filename)
    names, blocks, _, _, _, _, q = ablate.load()
    held = 0
    sources = list(range(1, 14))
    best, _ = ablate.chosen_configuration(blocks['R5'], q, sources)
    pred = ablate.predictions(blocks['R5'], q, sources, held, best[0])[best]
    tampered = q.copy()
    tampered[held] = 1 - tampered[held]
    alternate, _ = ablate.chosen_configuration(blocks['R5'], tampered, sources)
    pred_alternate = ablate.predictions(blocks['R5'], tampered, sources, held, alternate[0])[alternate]
    assert best == alternate and np.array_equal(pred, pred_alternate)
    result = json.loads((HERE / 'results.json').read_text())
    assert result['new_rollouts'] == 0 and result['status'] == 'TRANSFER_GATE_FAILED'
    assert [r['held'] for r in result['per_checkpoint']] == names
    with np.load(HERE / 'outer_predictions.npz') as z:
        assert z['predictions'].shape == (6, 14, 24, 16)
        assert z['controller_names'].tolist() == names
        assert np.isfinite(z['predictions']).all()
    report = ROOT / 'docs/part3_round3_transfer_feasibility_20261012.md'
    missing = []
    for link in re.findall(r'\]\(([^)]+)\)', report.read_text()):
        if '://' not in link and not (report.parent / link.split('#')[0]).exists():
            missing.append(link)
    assert not missing, missing
    out = {'frozen_hashes_verified': receipt, 'held_label_firewall_fold': names[held],
           'R5_chosen_configuration_unchanged_after_target_label_tamper': True,
           'R5_outer_prediction_bitwise_unchanged_after_target_label_tamper': True,
           'all_14_checkpoint_predictions_finite': True, 'report_missing_links': missing}
    (HERE / 'qa.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
