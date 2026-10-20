"""Strict, isolated replay of the frozen 32 historical intervention records."""
from __future__ import annotations

import datetime
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from adapter import historical_state, intervention_runtime


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits(value):
    return None if value is None else struct.pack('>d', float(value)).hex()


def main():
    receipt = read(HERE / 'PREREGISTRATION.receipt.json')
    selection_receipt = read(HERE / 'G1A_SELECTION.receipt.json')
    assert receipt['preregistration_sha256'] == sha(HERE / 'PREREGISTRATION.md')
    assert selection_receipt['selection_sha256'] == sha(HERE / 'G1A_SELECTION.json')
    selected = read(HERE / 'G1A_SELECTION.json')['records']
    assert len(selected) == 32
    states = read(ROOT / 'diagnostics/orthoflow3_loso_partial_count_v1/states.json')
    runtimes = {}
    results = []
    for number, row in enumerate(selected):
        ck = Path(row['checkpoint_path'])
        assert sha(ck) == row['checkpoint_sha256']
        index = int(row['original']['initial_state_index'])
        old = states[index]
        assert old['state_uid'] == row['state_uid']
        assert old['physical']['positions'] == row['original']['initial_positions']
        assert old['physical']['velocities'] == row['original']['initial_velocities']
        assert old['physical']['goals'] == row['original']['goals']
        assert sha(row['source_path']) == row['source_sha256']
        if row['run'] not in runtimes:
            runtimes[row['run']] = intervention_runtime(ck)
        rt, reset_counter = runtimes[row['run']]
        reset_counter()
        actual = rt.rollout(historical_state(old, index), np.asarray(row['original']['eta']),
                            int(row['original']['future_index']), 'orthoflow3')
        fields = {
            'outcome': actual['outcome'] == row['expected']['outcome'],
            'episode_length': actual['episode_length'] == row['expected']['episode_length'],
            'min_agent_distance_float64_be': bits(actual['minimum_agent_clearance']) == row['expected']['min_agent_distance_float64_be'],
            'min_wall_distance_float64_be': bits(actual['minimum_wall_clearance']) == row['expected']['min_wall_distance_float64_be'],
            'future_root_seed': actual['future_root_seed'] == row['original']['future_root_seed'],
            'rng_namespace': actual['rng_namespace'] == row['original']['rng_namespace'],
            'state_uid': actual['state_uid'] == row['state_uid'],
            'initial_state_index': actual['initial_state_index'] == index,
            'initial_positions': actual['initial_positions'] == row['original']['initial_positions'],
            'initial_velocities': actual['initial_velocities'] == row['original']['initial_velocities'],
            'goals': actual['goals'] == row['original']['goals'],
            'eta': actual['eta'] == row['original']['eta'],
            'future_index': actual['future_index'] == row['original']['future_index'],
            'controller_chain': actual['controller_chain'] == 'orthoflow3',
        }
        result = dict(ordinal=number, rollout_uid=row['rollout_uid'], run=row['run'],
                      outcome_stratum=row['outcome_stratum'], passed=all(fields.values()),
                      checks=fields, expected=row['expected'], actual={
                          'outcome': actual['outcome'], 'episode_length': actual['episode_length'],
                          'min_agent_distance_float64_be': bits(actual['minimum_agent_clearance']),
                          'min_wall_distance_float64_be': bits(actual['minimum_wall_clearance']),
                          'execution_error': actual.get('execution_error')})
        results.append(result)
        print(json.dumps({'gate': 'G1a', 'ordinal': number + 1, 'passed': result['passed'],
                          'uid': row['rollout_uid']}), flush=True)
    passed = len(results) == 32 and all(x['passed'] for x in results)
    report = dict(gate='G1a', status='PASS' if passed else 'FAIL',
                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  selection_sha256=sha(HERE / 'G1A_SELECTION.json'),
                  adapter_sha256=sha(HERE / 'adapter.py'),
                  runtime_code_sha256=sha(ROOT / 'new_benchmark_common/safety_eta3.py'),
                  results=results)
    (HERE / 'G1A_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'gate': 'G1a', 'status': report['status'],
                      'passed': sum(x['passed'] for x in results), 'required': 32}), flush=True)
    if not passed:
        raise SystemExit('STOP: G1a failed')


if __name__ == '__main__':
    main()
