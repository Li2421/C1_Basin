#!/usr/bin/env python3
"""Finalize the gated PACT/readiness audit when no fresh validation is allowed."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_t0_pact_training_readiness_v1')
BASIS = Path('/home/zhihan/research/Basin_C1/diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py')


def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def write(path, rows, fields=None):
    fields = fields or (list(dict.fromkeys(k for r in rows for k in r)) if rows else ['state_id', 'status'])
    with Path(path).open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    metrics = read(HERE / 'pact_cached_metrics.csv')
    selected = [r for r in metrics if r.get('selected') == 'True']
    if len(selected) != 8:
        raise RuntimeError(f'expected 8 selected cached representations, got {len(selected)}')
    pre_usable = [r for r in selected if r.get('shape_usable_before_fresh') == 'True']
    validation_manifest = read(HERE / 'pact_validation_manifest.csv')
    if pre_usable or validation_manifest:
        raise RuntimeError('offline finalizer is valid only when the pre-fresh shape gate selects no states')
    write(HERE / 'pact_validation_results.csv', [],
          ['state_id', 'probe_id', 'category', 'successes', 'trials', 'Q64', 'B63',
           'deadlock', 'timeout', 'collision', 'numerical'])

    recalls = [float(r['independent_B63_recall']) for r in selected]
    triangle_count = sum(int(r['triangle_count']) for r in selected)
    thickness_minus = [float(r['median_delta_minus']) for r in selected]
    thickness_plus = [float(r['median_delta_plus']) for r in selected]
    gate = {
        'classification': 'PACT_NOT_TRAINING_READY',
        'pass': False,
        'shape_usable_before_fresh': 0,
        'verified_pact_usable': 0,
        'fresh_validation_eta': 0,
        'fresh_inside_false_inclusions': 0,
        'fresh_validation_status': 'NOT_RUN_BY_PREDECLARED_GATE',
        'selected_total_cells': triangle_count,
        'selected_scale_by_state': {r['state_id']: float(r['bridge_scale']) for r in selected},
        'independent_B63_recall_by_state': {r['state_id']: float(r['independent_B63_recall']) for r in selected},
        'median_independent_B63_recall_selected': statistics.median(recalls),
        'median_independent_B63_recall_usable': None,
        'cached_false_inclusions': sum(int(r['cached_false_inclusions']) for r in selected),
        'median_delta_minus': statistics.median(thickness_minus),
        'median_delta_plus': statistics.median(thickness_plus),
        'universal_solution_audit': 'NOT_APPLICABLE_ZERO_USABLE_RETAINED_SETS',
        'failed_criteria': [
            'verified_pact_usable_at_least_6_of_8',
            'median_independent_B63_recall_at_least_0.50',
        ],
        'reason': 'zero cached-false-inclusion PACT cells exist, but held-out independent-Sobol B63 recall is 0 for six states and 0.0625 for two states; no state reaches the predeclared 0.30 shape-usability threshold',
    }
    dump(HERE / 'pact_gate.json', gate)

    set_rows = read(HERE / 'set_dataset_candidates.csv')
    for row in set_rows:
        row['verified_pact_usable'] = False
        row['set_usable'] = False
        row['rejection_reason'] = 'PACT_PRE_FRESH_SHAPE_GATE_FAIL'
    write(HERE / 'set_dataset_candidates.csv', set_rows)
    set_counts = Counter(r['split'] for r in set_rows if r['set_usable'] == 'True')
    set_gate = {
        'pass': False,
        'counts': {k: set_counts[k] for k in ('train', 'val', 'test')},
        'required': {'train': 24, 'val': 8, 'test': 8},
        'missing': {'train': 24, 'val': 8, 'test': 8},
        'reason': 'no state has a VERIFIED-PACT-USABLE label',
    }
    dump(HERE / 'set_dataset_gate.json', set_gate)
    point_gate = json.load(open(HERE / 'point_dataset_gate.json'))
    inventory = read(HERE / 'unique_t0_state_inventory.csv')
    exact = read(HERE / 'exact_q64_manifest.csv')
    point_rows = read(HERE / 'point_dataset_candidates.csv')

    decision = {
        'classification': 'PACT_NOT_READY_AND_T0_DATA_INSUFFICIENT',
        'pact_gate_pass': False,
        'set_dataset_gate_pass': False,
        'point_dataset_gate_pass': False,
        'training_executed': False,
        'exact_q64_eta': len(exact),
        'exact_q64_states': sum(int(r.get('exact_q64_eta', '0')) > 0 for r in inventory),
        'unique_true_t0_state_records': len(inventory),
        'true_t0_states_with_exact_h0': sum(r.get('h0_available') == 'True' for r in inventory),
        'set_usable_states': 0,
        'point_usable_states': len(point_rows),
        'achievable_set_split': set_gate['counts'],
        'achievable_point_split': point_gate['counts'],
        'point_split_missing': point_gate['missing'],
        'point_supervision_alone_sufficient_if_shape_failed': False,
        'answer_pact_encoding': 'No. It is cached-consistent but captures almost none of the held-out independent robust cloud.',
        'answer_set_reliability': 'Not established; no state passed the pre-fresh recall gate, so fresh false-inclusion rollout was correctly not run.',
        'answer_training_now': 'No. Only 8 unique true-t0 states have defensible exact-B63 robust point targets, versus the required 40 in a 24/8/8 split.',
        'next_scientific_step': 'Acquire 32 additional source-diverse true-t0 robust-interior point labels in the frozen 19/7/6 split deficit, then train the clean G_POINT_T0 baseline before revisiting set learning.',
    }
    dump(HERE / 'final_decision.json', decision)
    runtime = {
        'new_rollouts': 0, 'new_continuations': 0, 'physical_steps': 0,
        'gpu_jobs_submitted': 0, 'max_gpu_shards': 0,
        'networks_trained': 0, 'offline_geometry_seconds_approx': 5,
        'reason_no_rollout': '0/8 states passed the predeclared cached shape-usability gate',
    }
    dump(HERE / 'runtime_statistics.json', runtime)

    state_lines = '\n'.join(
        f"| {r['state_id']} | {r['bridge_scale']} | {r['triangle_count']} | "
        f"{float(r['independent_B63_recall']):.4f} | {r['cached_false_inclusions']} | "
        f"{float(r['median_delta_minus']):.3f}/{float(r['median_delta_plus']):.3f} |"
        for r in selected)
    report = f'''# OrthoFlow3 true-t0 PACT encoding and training readiness v1

## Outcome

**PACT_NOT_READY_AND_T0_DATA_INSUFFICIENT.** No rollout or network training was launched after the deterministic gates failed.

## PACT cached geometry

The audit used 877 exact-Q64 eta records from eight true-t0 states: 762 B63 and 115 non-B63. Independent `coverage64`/Sobol B63 points were excluded from triangulation and used as held-out cached recall.

| State | selected L/d5 | cells | independent B63 recall | cached false inclusions | median delta-/delta+ |
|---|---:|---:|---:|---:|---:|
{state_lines}

All selected PACT candidates have zero cached false inclusions, but six states have zero independent recall and two have recall 1/16=0.0625. Median recall is 0. This fails the per-state 0.30 gate for all eight states. Consequently there were zero SHAPE-USABLE states, and the preregistered fresh validation stage was not run. The 0.10 cached thickness values are therefore not evidence of transferable normal thickness; they only reflect absence of cached negatives inside the accepted local cells.

## State-level readiness

The repository contains 312 deduplicated true-t0 state records. Of these, 301 have an exact frozen 214-D h0 vector, but only eight states have compatible exact-Q64 clouds and defensible independently verified robust-center targets. The remaining records are single-episode fresh-WIDE states or historical state snapshots/lower-seed evidence—not B63 supervision.

- SET-USABLE: 0; achievable split 0/0/0 versus required 24/8/8.
- POINT-USABLE: 8; frozen candidate split 5/1/2, missing 19/7/6.

Thus even if PACT encoding had failed completely—which it did under the recall criterion—the existing true-t0 point supervision alone would **not** be sufficient to train. Hundreds of eta evaluations at eight h0 values remain eight state-level examples.

## Interpretation

This is a representation failure and a state-count shortfall, not evidence that basin supervision itself is invalid. PACT interpolates the structured construction cloud without cached false inclusions, yet its surfaces fail to cover the independent robust observations. Training was prohibited by both dataset gates.

## Next step

{decision['next_scientific_step']}
'''
    (HERE / 'final_report.md').write_text(report)

    required = [
        'protocol.md', 'exact_q64_manifest.csv', 'pact_candidates.csv', 'pact_cells.csv',
        'pact_cached_metrics.csv', 'pact_validation_manifest.csv', 'pact_validation_results.csv',
        'pact_gate.json', 'unique_t0_state_inventory.csv', 'state_deduplication.json',
        'set_dataset_candidates.csv', 'point_dataset_candidates.csv', 'set_dataset_gate.json',
        'point_dataset_gate.json', 'final_decision.json', 'runtime_statistics.json', 'final_report.md',
    ]
    dump(HERE / 'manifest.json', {
        'experiment': 'ORTHOFLOW3_T0_CONSERVATIVE_TUBE_ENCODING_AND_TRAINING_READINESS_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'orthoflow3_sha256': sha(BASIS), 'new_rollouts': 0, 'networks_trained': 0,
        'artifacts': {name: sha(HERE / name) for name in required},
    })
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
