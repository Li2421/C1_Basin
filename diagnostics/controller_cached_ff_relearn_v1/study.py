"""Re-encode existing paired-controller FF evidence; no new simulation."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIELD = Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
OLD = FIELD / 'field_pipeline_v1'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def freeze(path, value):
    if Path(path).exists():
        assert read(path) == value, str(path)
    else:
        write(path, value)


def prepare():
    source = read(OLD / 'decision_pair_critic_v3/source_pairs.json')
    prior_test = read(OLD / 'decision_pair_test_v3/manifest.json')
    split = {}
    for row in source['records']:
        sid = row['state_uid']
        assert sid not in split or split[sid] == row['split']
        split[sid] = row['split']
    conn = sqlite3.connect(f'file:{OLD}/datasets/field_controller_v2/rollout.sqlite?mode=ro', uri=True)
    states = []
    for sid, sp in sorted(split.items()):
        found = conn.execute('SELECT state_json FROM state WHERE state_uid=?', (sid,)).fetchone()
        assert found, sid
        state = json.loads(found[0]); state['study_split'] = sp
        states.append(state)
    conn.close()
    for state in prior_test['states']:
        assert state['uid'] not in split
        state = dict(state); state['study_split'] = 'development'
        states.append(state)
    hashes = [hashlib.sha256(json.dumps(s['physical'], sort_keys=True).encode()).hexdigest() for s in states]
    assert len(hashes) == len(set(hashes)), 'Physical state leakage'
    protocol = dict(
        schema='cached_real_ff_controller_relearn_v1', new_rollouts=0,
        purpose='Diagnostic re-learning on cached true FF; BOTH controllers seen in TRAIN. Not unseen-controller or LOSO confirmation.',
        source=str(OLD / 'decision_pair_critic_v3/source_pairs.json'),
        source_sha256=sha(OLD / 'decision_pair_critic_v3/source_pairs.json'),
        old_test_now='DEVELOPMENT: previously inspected outcomes; cannot be final confirmation',
        development_manifest_sha256=sha(OLD / 'decision_pair_test_v3/manifest.json'),
        split=dict(Counter(s['study_split'] for s in states)), states=states,
        controller_uids=prior_test['controller_uids'], eta=prior_test['etas'],
        label_policy='Reuse exact state/eta/controller/seed. Deduplicate all phase copies; conflict is fatal. Preserve Q4 screens and selected Q16; no imputation. Numerical outcomes remain unknown.',
        controller_input='Static policy-function queries at prespecified physical probes, separately current and future PROGRAM branches; no simulated future state, rollout preview, eta query, ID or raw controller weights.',
        current_branch='base Flow for both controllers; exact-identical negative control',
        future_branch='base Flow for base program, alternate Flow for held program; query the known branch at the same prespecified physical probes',
        representation='Existing masked physical entity encoding; controller features use nominal14 and compact field64 per program branch.',
        models=['eta_only','state_only','current_field','branch_nominal','branch_field','branch_field_contrast'],
        fit_seeds=[17,23,41], steps=6000, validation_every=200, batch_states=8,
        learning_rate=0.0003, weight_decay=0.0001,
        training='Uniform physical TRAIN states; both controllers and all available eta; binomial NLL divided by observed seeds per cell, mean over observed cells. Same labels/weighting/steps for all models.',
        contrast_objective='Only branch_field_contrast: add 0.5 paired-controller Q-difference MSE and 0.5 eta-centered paired-controller difference MSE, on cells observed for both controllers.',
        checkpoint='Minimize validation probability MSE + controller-difference MSE + eta-centered controller-difference MSE; ties use selection regret then NLL. No development-label model selection.',
        primary='Lowest mean three-seed source-validation score among controller-conditioned models; also report every fit and negative control.',
        bootstrap='Physical state unit, both controllers and all eta retained; 20000 replicates. CI conditional on these two seen controller programs.',
        controls=['swap controller description at fixed state and eta', '100 within-scene state permutations with controller program fixed', 'all 16 fixed eta and per-controller best fixed, reselect inside bootstrap'],
        reversal='Prespecified abs discovery8 controller-specific eta gap >=3/8, opposite signs for the two controllers; independent evaluation8. Include all qualifying cases; report evaluation stability and two-sided correctness.',
        publications=[
            'https://arxiv.org/abs/2002.11833',
            'https://proceedings.mlr.press/v119/raileanu20a.html'],
        literature_boundary='Policy fingerprints can support policy-value learning; PDVF also conditions on environment and uses target interactions. Neither implies arbitrary zero-shot scene transfer.',
        final_confirmation='No final states in this study. Any subsequent confirmation requires fresh outcome-blind states after the method is frozen.')
    freeze(HERE / 'protocol.json', protocol)
    sid_ix = {s['uid']: i for i, s in enumerate(states)}
    eta_key = lambda e: np.asarray(e, np.float64).tobytes()
    eta_ix = {eta_key(e): i for i, e in enumerate(protocol['eta'])}
    ctl_ix = {protocol['controller_uids'][c]: i for i, c in enumerate(('base','held'))}
    conn = sqlite3.connect(f'file:{OLD}/field_rollout.sqlite?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    by_key = {}; phase_counts = Counter(); duplicates = 0
    for row in conn.execute('SELECT * FROM continuation WHERE controller_uid IN (?,?)', tuple(ctl_ix)):
        row = dict(row)
        if row['state_uid'] not in sid_ix:
            continue
        key_eta = eta_key(json.loads(row['eta_json']))
        if key_eta not in eta_ix:
            continue
        assert 0 <= row['seed'] < 16
        key = (sid_ix[row['state_uid']], ctl_ix[row['controller_uid']], eta_ix[key_eta], row['seed'])
        result = json.loads(row['result_json'])
        outcome = (row['success'], row['collision'], row['numerical_unresolved'], result['termination'])
        assert not row['success'] or not (row['collision'] or row['numerical_unresolved'])
        phase_counts[row['phase']] += 1
        if key in by_key:
            assert by_key[key]['outcome'] == outcome, ('Conflicting cached outcomes', key)
            by_key[key]['copies'].append(dict(id=row['id'], phase=row['phase']))
            duplicates += 1
        else:
            by_key[key] = dict(key=key, outcome=outcome, result=result,
                               copies=[dict(id=row['id'], phase=row['phase'])])
    conn.close()
    shape = (len(states), 2, 16, 16)
    # -2 absent, -1 numerical, 0 observed failure, 1 success.
    outcome = np.full(shape, -2, np.int8)
    collision = np.zeros(shape, bool)
    for key, row in by_key.items():
        s, col, u, _ = row['outcome']
        outcome[key] = -1 if u else s; collision[key] = bool(col)
    success = (outcome == 1).sum(-1); failure = (outcome == 0).sum(-1)
    unknown = (outcome == -1).sum(-1); absent = (outcome == -2).sum(-1)
    for row in source['records']:
        i = sid_ix[row['state_uid']]; c = ('base','held').index(row['controller_condition'])
        j = eta_ix[eta_key(row['eta'])]
        assert [int(success[i,c,j]), int(failure[i,c,j]), int(unknown[i,c,j])] == [row['success'],row['fail'],row['unresolved']], (row['state_uid'],c,j)
    dev = np.array([s['study_split'] == 'development' for s in states])
    assert (absent[dev] == 0).all()
    with gzip.open(HERE / 'raw_seed_evidence.jsonl.gz', 'wt') as f:
        for _, row in sorted(by_key.items()):
            f.write(json.dumps(row, sort_keys=True) + '\n')
    np.savez_compressed(HERE / 'labels.npz', outcome=outcome, collision=collision,
        success=success, failure=failure, unknown=unknown, absent=absent,
        eta=np.asarray(protocol['eta'], np.float32),
        split=np.array([s['study_split'] for s in states]), state_uid=np.array([s['uid'] for s in states]))
    audit = dict(unique_reused_seed_outcomes=len(by_key), duplicate_phase_copies=duplicates,
        conflicting_outcomes=0, source_q16_aggregate_checks=len(source['records']),
        physical_overlap=0, new_rollouts=0, phase_counts=dict(phase_counts),
        labels_sha256=sha(HERE / 'labels.npz'), raw_sha256=sha(HERE / 'raw_seed_evidence.jsonl.gz'),
        split_summary={sp:dict(states=int(np.sum(np.array([s['study_split'] for s in states]) == sp)),
            seed_outcomes=int(np.sum(outcome[np.array([s['study_split'] for s in states]) == sp] != -2)),
            numeric_unknown=int(np.sum(outcome[np.array([s['study_split'] for s in states]) == sp] == -1)),
            collision=int(collision[np.array([s['study_split'] for s in states]) == sp].sum()),
            cell_seed_counts=dict(Counter(map(int,(16-absent[np.array([s['study_split'] for s in states]) == sp]).ravel()))))
            for sp in ('train','validation','development')})
    write(HERE / 'cache_reuse_audit.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k in ('unique_reused_seed_outcomes','duplicate_phase_copies','split_summary','new_rollouts')}), flush=True)


def features():
    import jax
    jax.config.update('jax_enable_x64', False)
    sys.path.insert(0, str(FIELD))
    from field_pipeline_v1.held import AlternateFourRingRuntime
    from field_pipeline_v1.ring_t0_source import runtime, controller_uid as base_uid
    from field_pipeline_v1.held_ring_k16 import controller_uid as held_uid
    from diagnostics.physical_state_critic_formal_v1.physical_entities import entities
    from diagnostics.controller_q_diversity_v1.runtime import controller_context
    from diagnostics.controller_q_diversity_v1.features import field_context
    from diagnostics.controller_q_diversity_v1.train import compact_field
    p = read(HERE / 'protocol.json')
    assert p['controller_uids'] == dict(base=base_uid(), held=held_uid())
    bases = [runtime(), AlternateFourRingRuntime('ring_exchange')]
    jax.config.update('jax_enable_x64', False)
    assert sha(bases[0].checkpoint) == read(OLD / 'decision_pair_test_v3/manifest.json')['source_checkpoint_sha256']
    assert sha(bases[1].checkpoint) == read(OLD / 'decision_pair_test_v3/manifest.json')['alternate_checkpoint_sha256']
    # Fail closed if static feature extraction ever attempts an environment step.
    def forbidden_step(*args, **kwargs):
        raise RuntimeError('No simulator step allowed during controller feature extraction')
    for rt in bases:
        rt.env_cls.step = forbidden_step
    data = defaultdict(list)
    for i, state in enumerate(p['states']):
        physical = entities('ring_exchange', state['physical'])
        for k, v in physical.items():
            data['entity_' + k].append(v)
        raw = []
        for rt in bases:
            nominal = controller_context(rt, state)
            field = field_context(rt, state)
            raw.append(np.concatenate((nominal,field),-1))
        raw = np.stack(raw)
        data['branch_raw'].append(raw)
        data['branch_compact'].append(np.asarray(compact_field(raw)))
        if (i+1) % 24 == 0:
            print('Static feature states', i+1, flush=True)
    np.savez_compressed(HERE / 'features.npz', **{k:np.stack(v) for k,v in data.items()})
    # Current branch is exactly base for both complete controller programs.
    a = np.stack(data['branch_raw'])
    write(HERE / 'features_audit.json', dict(states=len(p['states']), simulator_steps=0,
        eta_queries=0, success_labels_used=False, H20_used=False,
        controller_uids_recomputed_and_verified=True,
        current_branch_difference=0., future_branch_mean_abs_difference=float(np.mean(np.abs(a[:,0]-a[:,1]))),
        feature_sha256=sha(HERE / 'features.npz'), code_sha256=sha(__file__),
        limitation='Only two seen controller programs. A branch signature is not a complete generic policy representation.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare','features'])
    args = p.parse_args()
    (prepare if args.stage == 'prepare' else features)()
