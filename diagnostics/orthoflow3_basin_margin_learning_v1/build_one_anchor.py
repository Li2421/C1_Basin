#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/zhihan/research/Basin_C1')
SOURCE = ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1/continuous_pilot6.py'
QDIR = ROOT/'diagnostics/orthoflow3_q_learnability_v2'
REF = ROOT/'diagnostics/orthoflow3_conservative_basin_ball_v1'
BASIS = ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
EXPECTED_BASIS = '51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def load_source():
    spec = importlib.util.spec_from_file_location('continuous_anchor_source', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def prepare(mod, state_id: str, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out/'raw').mkdir(exist_ok=True)
    qstates = {x['state_id']: x for x in json.load(open(QDIR/'eligible_state_manifest.json'))['selected_states']}
    if state_id not in qstates:
        raise RuntimeError(f'unknown Q-v2 state {state_id}')
    state = qstates[state_id]
    if sha(BASIS) != EXPECTED_BASIS:
        raise RuntimeError('authoritative OrthoFlow3 hash mismatch')
    dump(out/'frozen_state_manifest.json', {
        'schema': 'orthoflow3_basin_margin_learning_v1_anchor_state',
        'selection_rule': 'next required split state in the pre-outcome frozen conservative-ball permutation',
        'states': [state],
    })
    for name in ('ebridge_definition.json','ebridge_halfspaces.csv','eta_normalization.json',
                 'ebridge_sobol_sequence.csv','frozen_ray_directions.csv'):
        shutil.copy2(ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1'/name, out/name)
    rows, scanned = mod.compatible_rows({state_id})
    grouped = {}
    for (sid, key, future_index), row in rows.items():
        grouped.setdefault((sid, key), {})[future_index] = row
    b63, non = [], []
    for (sid, key), by_seed in grouped.items():
        if all(i in by_seed for i in range(64)):
            successes = sum(bool(by_seed[i]['success']) for i in range(64))
            rec = {'state_id': sid, 'eta': by_seed[0]['eta'], 'successes': successes}
            (b63 if successes >= 63 else non).append(rec)
    dump(out/'cache_reuse_audit.json', {
        'exact_compatibility': 'Q-v2 h/current-Flow conditioning, future_root=2026092702, future_index, fixed eta, same horizon/projections/hash',
        'compatible_source_files': [str(x) for x in mod.source_files()],
        'records_scanned': scanned,
        'compatible_exact_tuples': len(rows),
        'known_B63_state_eta': b63,
        'known_nonB63_state_eta': non,
        'new_rollouts_launched_during_cache_inventory': 0,
    })
    # One-state worst-case preflight, preserving every core stage.
    parts = {
        'common_stage32_plus_zero': 33*8,
        'possible_extension64': 32*8,
        'center_promotions_allowance': 3*56,
        'ray_screen_and_bisection': 18*7*8,
        'limiting_direction_promotions': 8*56,
        'inside_screen_and_mandatory_promotions': 12*8+3*56,
        'outside_shell': 8*8,
    }
    projected = sum(parts.values())
    dump(out/'cost_preflight.json', {
        'components': parts,
        'projected_new_continuations_before_cache_reuse': projected,
        'projected_physical_steps_at_276_5_per_continuation': int(np.ceil(projected*276.5)),
        'per_anchor_continuation_cap': 5000,
        'per_anchor_physical_step_cap': 1500000,
        'within_caps': projected <= 5000 and projected*276.5 <= 1500000,
    })
    (out/'protocol.md').write_text(
        '# New verified anchor ball\n\n'
        'This anchor uses the unchanged accepted continuous-domain conservative-ball protocol: '
        'the frozen common E_bridge Sobol sequence, empirical max-margin B63 center, 18 frozen rays, '
        'limiting-direction B63 promotion, kappa=0.85, independent inside validation, and outside shell. '
        'No learned Q/J or outcome-adaptive candidate coordinates are used.\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-id', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    mod = load_source()
    mod.HERE = out
    mod.STATE_IDS = [args.state_id]
    mod.CAP_CONT = 5000
    mod.CAP_STEPS = 1500000
    prepare(mod, args.state_id, out)
    mod.run()


if __name__ == '__main__':
    main()
