"""CPU-only retrospective audit; never opens the reserved independent test.

Uses one development noise realization to score another at the same start.
This is NOT an ex-ante planning test: scores use complete realized rollouts.
Bootstrap resamples starts, preserving both noise realizations together.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + .5 * (differences == 0)))


def audit(folder):
    path = folder / 'validation.json'
    validations = json.loads(path.read_text())
    history = json.loads((folder / 'history.json').read_text())
    summaries = []
    for validation in validations:
        groups = {}
        for episode in validation['episodes']:
            groups.setdefault(episode['rid'], []).append(episode)
        pairs = [sorted(groups[rid], key=lambda x: x['noise_seed'])
                 for rid in sorted(groups)]
        assert all(len(pair) == 2 for pair in pairs)
        assert len({tuple(e['noise_seed'] for e in pair) for pair in pairs}) == 1
        scores = np.array([[e['J_live'] for e in pair] for pair in pairs])
        labels = np.array([[e['either_deadlock'] for e in pair] for pair in pairs])
        cross_labels = labels[:, ::-1]
        bootstrap = []
        rng = np.random.default_rng(20260918)
        for _ in range(2000):
            indices = rng.integers(0, len(pairs), len(pairs))
            value = auc(scores[indices].ravel(), cross_labels[indices].ravel())
            if value is not None:
                bootstrap.append(value)
        summaries.append(dict(
            update=validation['update'], starts=len(pairs),
            deadlocks=int(labels.sum()), mean_risk=float(scores.mean()),
            zero_score_fraction=float(np.mean(scores == 0)),
            same_rollout_auc=auc(scores.ravel(), labels.ravel()),
            cross_noise_auc=auc(scores.ravel(), cross_labels.ravel()),
            cross_noise_auc_cluster_ci95=(np.quantile(bootstrap, [.025, .975]).tolist()
                                          if bootstrap else None),
            bootstrap_valid=len(bootstrap),
            discordant_start_count=int(np.sum(labels[:, 0] != labels[:, 1]))))
    batch_groups = {}
    for name, zero in [('zero_risk', True), ('positive_risk', False)]:
        rows = [r for r in history if (r['pre']['J_live'] == 0) == zero]
        ratios = [r['post']['J_def'] / r['pre']['J_def'] for r in rows
                  if r['pre']['J_def'] > 1e-12]
        batch_groups[name] = dict(count=len(rows),
            restart_count=sum(r['decision']['proposal'] == 'restart' for r in rows),
            median_gradient_norm=float(np.median([r['gradient_norm'] for r in rows]))
                if rows else None,
            median_post_pre_deviation_ratio=float(np.median(ratios)) if ratios else None)
    return dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                validation=summaries, training_batch_groups=batch_groups)


if __name__ == '__main__':
    result = dict(scope='Retrospective development-only cross-noise association; no GPU, no test opening',
        limitations=['Complete-rollout scoring, not demonstrated early prediction',
                     'Only two noise realizations per start',
                     'Checkpoints and validation already inspected; not confirmatory evidence',
                     'Checkpoint comparisons do not isolate optimizer causality'],
        runs=[audit(Path('results') / name) for name in
              ['c1_exact_margin_v1', 'c1_ordered_parallel_v2']])
    destination = Path('results/c1_cross_noise_audit_v1.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
