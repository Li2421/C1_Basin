"""Audit reusable physical-response context against cached field-policy pairs.

The independent POC directory is read-only. No simulator, JAX, training, rollout
or database mutation. This checks measured distinguishability, not whether a
context is sufficient or whether a critic transfers between paradigms.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004/validation_v2')
OUT = ROOT/'cross_paradigm_context_audit'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def csvwrite(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def response(positions, goals, action):
    """Common physical units; no policy/paradigm ID and no future outcome."""
    goal_vector = goals-positions
    unit = goal_vector/np.maximum(np.linalg.norm(goal_vector, axis=1, keepdims=True), 1e-12)
    progress = np.sum(action*unit, axis=-1)
    speed = np.linalg.norm(action, axis=-1)
    pairs = np.triu_indices(len(positions), 1)
    dp = (positions[:, None]-positions[None, :])[pairs]
    da = (action[:, None]-action[None, :])[pairs]
    separating_speed = np.sum(dp*da, axis=-1)/np.maximum(np.linalg.norm(dp, axis=-1), 1e-12)
    return np.array([progress.mean(), progress.min(), speed.mean(), speed.max(),
                     separating_speed.mean(), separating_speed.min()], dtype=float)


def main():
    files = [SOURCE/'protocol.json', SOURCE/'q16_extension.json', SOURCE/'run_validation.py',
             SOURCE/'adapters.py', SOURCE.parent/'poc/field_controller.py']
    hashes = {str(p): sha(p) for p in files}
    cases = [c for c in read(files[0])['cases'] if c['eta_name'] in ('zero', 'generator_location')]
    cases += read(files[1])['cases']
    assert len(cases) == len({c['id'] for c in cases}) == 320
    rows = []
    grouped = defaultdict(list)
    prefixes = SOURCE/'results'
    for case in cases:
        paths = [prefixes/f"{case['id']}_{chain}" for chain in ('old', 'field')]
        results = [read(p.with_suffix('.json')) for p in paths]
        traces = [np.load(p.with_suffix('.npz')) for p in paths]
        a, b = results
        assert a['controller_uid'] != b['controller_uid']
        for r in results:
            assert r['state_uid'] == case['state']['uid'] and r['seed'] == case['seed']
            assert np.array_equal(r['eta'], case['eta'])
        np.testing.assert_array_equal(traces[0]['positions'][0], traces[1]['positions'][0])
        np.testing.assert_array_equal(traces[0]['goals'], traces[1]['goals'])
        prefix = min(len(t['noise_keys']) for t in traces)
        np.testing.assert_array_equal(traces[0]['noise_keys'][:prefix], traces[1]['noise_keys'][:prefix])
        if prefix:
            features = [response(t['positions'][0], t['goals'], t['actions'][0]) for t in traces]
            first_action_distance = float(np.linalg.norm(traces[0]['actions'][0]-traces[1]['actions'][0]))
            response_distance = float(np.linalg.norm(features[0]-features[1]))
        else:
            first_action_distance = response_distance = None
        row = dict(case_id=case['id'], scenario=a['scenario'], state_uid=a['state_uid'],
                   eta_name=case['eta_name'], seed=case['seed'],
                   old_controller_uid=a['controller_uid'], field_controller_uid=b['controller_uid'],
                   old_success=int(a['success']), field_success=int(b['success']),
                   old_numerical=a['error'] is not None, field_numerical=b['error'] is not None,
                   first_action_distance=first_action_distance, physical_response_distance=response_distance,
                   old_source=str(paths[0].with_suffix('.json')), field_source=str(paths[1].with_suffix('.json')))
        rows.append(row)
        grouped[(a['scenario'], a['state_uid'], case['eta_name'])].append(row)
    groups = []
    for (scene, state, eta), rr in sorted(grouped.items()):
        assert {r['seed'] for r in rr} == set(range(16))
        valid = [r for r in rr if not r['old_numerical'] and not r['field_numerical']]
        rescue = sum(not r['old_success'] and r['field_success'] for r in valid)
        brk = sum(r['old_success'] and not r['field_success'] for r in valid)
        os = sum(r['old_success'] for r in rr)
        fs = sum(r['field_success'] for r in rr)
        on = sum(r['old_numerical'] for r in rr)
        fn = sum(r['field_numerical'] for r in rr)
        def status(s, n):
            return 'B15' if s >= 15 else ('non_B15' if s+n < 15 else 'UNRESOLVED')
        groups.append(dict(scenario=scene, state_uid=state, eta_name=eta,
                           old_successes=os, field_successes=fs, old_numerical=on, field_numerical=fn,
                           old_B15=status(os, on), field_B15=status(fs, fn),
                           matched_valid=len(valid), rescue=rescue, brk=brk,
                           paired_exact_p=float(binomtest(rescue, rescue+brk, .5).pvalue) if rescue+brk else 1.,
                           median_first_action_distance=float(np.median([r['first_action_distance'] for r in rr if r['first_action_distance'] is not None])),
                           median_physical_response_distance=float(np.median([r['physical_response_distance'] for r in rr if r['physical_response_distance'] is not None]))))
    order = np.argsort([r['paired_exact_p'] for r in groups])
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, groups[i]['paired_exact_p']*(len(groups)-rank))
        groups[i]['Holm_p'] = float(min(running, 1.))
    summary = dict(new_rollouts=0, new_fits=0, source_directory_written=False,
                   matched_seed_pairs=len(rows), canonical_pairs=len(groups),
                   numerical_records=sum(r['old_numerical']+r['field_numerical'] for r in rows),
                   outcome_disagreements=sum(r['old_success'] != r['field_success'] for r in rows if not r['old_numerical'] and not r['field_numerical']),
                   pairs_Holm_below_005=sum(r['Holm_p'] < .05 for r in groups),
                   distinguishable_first_action=sum(r['first_action_distance'] is not None and r['first_action_distance'] > 1e-8 for r in rows),
                   distinguishable_response_summary=sum(r['physical_response_distance'] is not None and r['physical_response_distance'] > 1e-8 for r in rows),
                   input_information='Shared raw Flow, state, eta and safety set are not the whole policy; placement/integration changes executable response.',
                   response_columns=['goal-progress mean', 'goal-progress minimum', 'speed mean', 'speed maximum', 'pair separating speed mean', 'pair separating speed minimum'],
                   interpretation='A common executable-action response interface distinguishes old versus field policies on this cached panel. This is not a sufficiency, critic-learning or cross-scene generalization result.',
                   limitations=['Small previously inspected POC panel, not a new confirmation set.',
                                'Per-seed action features here audit behavior; using realized evaluation noise as learned-critic input would change conditioning. Training/deployment probes must use a separate frozen noise namespace.',
                                'Do not transfer old Q/B15 labels or cache identity to the field policy.',
                                'No assertion that one-step response is sufficient; existing late-response aliases contradict universal short-horizon sufficiency.'],
                   source_file_hashes=hashes)
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    OUT.mkdir(exist_ok=True)
    csvwrite(OUT/'paired_records.csv', rows)
    csvwrite(OUT/'controller_pair_results.csv', groups)
    write(OUT/'audit.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'source_file_hashes'}, indent=2))
    print(json.dumps([r for r in groups if r['Holm_p'] < .05], indent=2))


if __name__ == '__main__':
    main()
