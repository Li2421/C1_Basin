"""Freeze a 320-continuation diagnostic; this script performs no rollouts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path[:0] = [str(SYSROOT), str(ROOT)]
import numpy as np
from scipy.special import expit
from shared_rollout_db.src.rollout_db import canonical, uid, eta_identity, connect
from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.prepare_cache_plan import (
    SCENARIO, CORRECTION_CONFIG, RNG)
sys.path.insert(0, str(SYSROOT))

OLD = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
K16 = ROOT / 'diagnostics/orthoflow3_mode_free_k_sweep_latency_v1'
AUDIT = ROOT / 'diagnostics/orthoflow3_toy_critic_root_cause_v1'
EXPERIMENT = 'exp_orthoflow3_toy_critic_local_state_probe_v1'
EPISODES = (102, 115, 162, 176, 177)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p, v): Path(p).write_text(json.dumps(v, indent=2, sort_keys=True, allow_nan=False) + '\n')
def csvwrite(p, rows):
    with Path(p).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def keys(state, kind, eta):
    content = hashlib.sha256(canonical({'initial_positions': state['initial_positions']}).encode()).hexdigest()
    return {'episode_index': state['episode_index'], 'kind': kind,
            'state_uid': uid('state', {'scenario': SCENARIO, 'content': content}),
            'eta_uid': eta_identity(eta)[0], 'controller_uid': uid('ctl', CORRECTION_CONFIG)}

def requests(mapping):
    return {'requests': [{k: r[k] for k in ('state_uid', 'eta_uid', 'controller_uid')} |
                         {'seed_keys': [canonical({'future_index': i}) for i in range(16)]}
                         for r in mapping]}

def main():
    if (OUT / 'frozen_proposals.json').exists():
        raise RuntimeError('Protocol already frozen; do not regenerate it')
    (OUT / 'logs').mkdir(exist_ok=True)
    old = read(OLD / 'frozen_proposals.json')
    ext = read(K16 / 'frozen_proposals.json')
    with (ROOT / 'diagnostics/orthoflow3_toy_critic_uncertainty_local_v1/frozen_k16_proposal_predictions.csv').open() as f:
        qrows = list(csv.DictReader(f))
    basis = ROOT / 'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
    assert sha(basis) == CORRECTION_CONFIG['basis_sha256']
    widepath = ROOT / 'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json'
    wide = read(widepath)
    from single_integrator.environment import Config, GiveWayEnv
    env = GiveWayEnv(Config(**wide['environment']))
    refs, states, position_checks = [], [], []
    for ep in EPISODES:
        parent = next(s for s in old['states'] if s['episode_index'] == ep)
        extra = next(s for s in ext['states'] if s['episode_index'] == ep)
        etas = parent['eta'] | extra['eta']
        qr = sorted((r for r in qrows if int(r['episode_index']) == ep), key=lambda r: int(r['proposal_index']))
        assert len(qr) == 16
        bad = int(np.argmax(extra['all_critic_scores']))
        good = int(max(qr, key=lambda r: float(r['Q16_lower']))['proposal_index'])
        assert float(qr[bad]['Q16_lower']) <= 5 / 16
        assert int(float(qr[good]['B15'])) == 1
        eta = {'frozen_bad': etas[f'sample_{bad}'], 'frozen_good': etas[f'sample_{good}']}
        refs.append({**parent, 'eta': eta, 'bad_proposal_index': bad, 'good_proposal_index': good,
                     'original_bad_q16': float(qr[bad]['Q16_lower']), 'original_good_q16': float(qr[good]['Q16_lower'])})
        for offset in (-0.01, 0.01):
            pos = np.asarray(parent['initial_positions'], np.float64).copy()
            pos[0, 0] += offset
            env.reset(pos)
            wall, agent = env.distances()
            local = len(states)
            states.append({'episode_index': local, 'parent_episode_index': ep,
                'state_id': f'local_probe_parent{ep:04d}_agent0_x_{"minus" if offset < 0 else "plus"}1cm',
                'source_group': parent['source_group'], 'rollout_id': parent['rollout_id'],
                'initial_positions': pos.tolist(), 'physical_offset_m': offset,
                'eta': eta, 'bad_proposal_index': bad, 'good_proposal_index': good})
            position_checks.append({'episode_index': local, 'parent_episode_index': ep,
                                    'minimum_wall_clearance': float(wall.min()), 'agent_clearance': agent})
    protocol = {'experiment_uid': EXPERIMENT, 'freeze_time': datetime.now(timezone.utc).isoformat(),
        'parent_episodes': list(EPISODES), 'parent_selection': 'five previously audited Q16<=5/16 critic-selected failures; diagnostic not population evaluation',
        'perturbation_rule': 'agent 0 longitudinal x += -0.01 or +0.01 metres; all other physical initial conditions unchanged',
        'outcome_blind_new_perturbations': True, 'parent_selection_outcome_blind': False,
        'reference_selection': 'bad=original deployed critic argmax; good=lowest proposal index maximizing cached Q16',
        'test_interpretation': 'targeted local-state diagnostic; not an unbiased generalization or deployment comparison',
        'unchanged': ['generator', 'critic', 'normalization', 'eta coordinates', 'physics', 'horizon', 'parent rollout_id', 'matched continuation seeds'],
        'seeds': list(range(16)), 'rng': RNG, 'correction_config': CORRECTION_CONFIG,
        'requested_new_state_continuations': 320, 'cached_parent_reference_continuations': 160,
        'source_group_rule': 'perturbations inherit parent source_group; none may be treated as independent training or test families',
        'no_training': True, 'new_eta_candidates': 0,
        'runner_sha256': sha(OLD / 'run_rollouts.py'), 'basis_sha256': sha(basis),
        'frozen_benchmark_sha256': sha(widepath),
        'original_proposals_sha256': sha(OLD / 'frozen_proposals.json'), 'k16_proposals_sha256': sha(K16 / 'frozen_proposals.json'),
        'interpretation_preregistered': {
            'stable_low_Q_bad_high_Q_good': 'local systematic critic overestimation; sharp local state-boundary not necessary to explain these failures',
            'strong_Q_change_under_small_h_change': 'local sensitivity/identifiability question; does not itself prove omitted input or irreducible aliasing',
            'both': 'mixed, report per source family',
            'remaining_limit': 'no retraining/intervention on local supervision; DATA versus MODEL cannot be fully separated'}}
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'hard_state_manifest.json', {'states': states, 'references': refs, 'position_checks': position_checks})
    # Only deterministic t0 feature construction and frozen network inference; no env.step.
    from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.freeze_proposals import features_for_cohort, critic_scores
    h = features_for_cohort(states, wide['environment'], wide['cbf'])
    rh = features_for_cohort(refs, wide['environment'], wide['cbf'])
    original_h = np.load(OLD / 'cohort_features.npz')['h_raw']
    assert np.allclose(rh, original_h[list(EPISODES)], rtol=1e-6, atol=1e-7)
    norm = read(ROOT / 'diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json')['state_normalization']['Toy']
    sd = np.asarray(norm['std'], np.float32)
    samples = np.asarray([[s['eta'][k] for k in ('frozen_bad', 'frozen_good')] for s in states], np.float64)
    scores, critics = critic_scores(h, samples)
    rsamples = np.asarray([[s['eta'][k] for k in ('frozen_bad', 'frozen_good')] for s in refs], np.float64)
    rscores, _ = critic_scores(rh, rsamples)
    predictions = []
    for i, state in enumerate(states):
        state['h_raw'] = h[i].astype(float).tolist()
        state['h_sha256'] = hashlib.sha256(h[i].astype(np.float64).tobytes()).hexdigest()
        dh = (h[i] - original_h[state['parent_episode_index']]) / sd
        state['h_normalized_rms_from_parent'] = float(np.sqrt(np.mean(dh**2)))
        state['h_normalized_max_from_parent'] = float(np.max(np.abs(dh)))
        for j, kind in enumerate(('frozen_bad', 'frozen_good')):
            predictions.append({'episode_index': i, 'parent_episode_index': state['parent_episode_index'],
                'offset_m': state['physical_offset_m'], 'kind': kind, 'critic_logit': float(scores[i,j]),
                'critic_probability': float(expit(scores[i,j])), 'h_normalized_rms_from_parent': state['h_normalized_rms_from_parent']})
    for i, ref in enumerate(refs):
        ref['reference_critic_probabilities'] = expit(rscores[i]).tolist()
    frozen = {'states': states, 'references': refs, 'critic_checkpoints': critics,
              'protocol_sha256': sha(OUT / 'protocol.json'), 'predictions_frozen_before_new_outcomes': True}
    write(OUT / 'frozen_proposals.json', frozen)
    np.savez_compressed(OUT / 'cohort_features.npz', h_raw=h)
    csvwrite(OUT / 'frozen_predictions.csv', predictions)
    mapping = [keys(s, k, eta) for s in states for k, eta in s['eta'].items()]
    refmapping = [keys(s, k, eta) for s in refs for k, eta in s['eta'].items()]
    csvwrite(OUT / 'candidate_cache_keys.csv', mapping)
    csvwrite(OUT / 'reference_cache_keys.csv', refmapping)
    write(OUT / 'planned_rollouts.json', requests(mapping))
    write(OUT / 'reference_requests.json', requests(refmapping))
    with connect() as con:
        existing = [r['state_uid'] for r in mapping if con.execute('SELECT 1 FROM state WHERE state_uid=?', (r['state_uid'],)).fetchone()]
        con.execute('INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,start_time,metadata_json) VALUES(?,?,?,?,?,?)',
                    (EXPERIMENT, OUT.name, str(OUT), sha(OUT/'protocol.json'), protocol['freeze_time'], canonical(protocol)))
    write(OUT / 'preflight_identity_audit.json', {'new_state_keys_already_present': sorted(set(existing)),
          'note': 'legacy planner labels unseen state keys INCOMPATIBLE; if absent they are genuinely missing, not controller mismatches',
          'reference_feature_replay_max_abs': float(np.max(np.abs(rh-original_h[list(EPISODES)]))),
          'new_state_count': len(states), 'exact_eta_reused_from_original_proposals': True})
    write(OUT / 'working_state.json', {'stage': 'FROZEN_AWAITING_PREFLIGHT', 'new_rollout': 0, 'requested': 320})
    csvwrite(OUT / 'experiment_ledger.csv', [{'timestamp': protocol['freeze_time'], 'stage': 'freeze', 'requested': 320, 'executed': 0, 'note': 'fixed 1cm +/- state probes; no new eta or models'}])
    print(json.dumps({'states': len(states), 'pairs': len(mapping), 'requested': len(mapping)*16,
                      'max_normalized_h_rms': max(s['h_normalized_rms_from_parent'] for s in states)}))

if __name__ == '__main__': main()
