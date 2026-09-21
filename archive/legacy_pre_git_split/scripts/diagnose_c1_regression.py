"""Read-only model diagnostics: fixed-noise ablations and training-batch replay.

Never publishes a trained checkpoint or selects a new residual scale.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from single_integrator.c1.differentiable_rollout import ResidualFlowField
from single_integrator.c1.models import ResidualCorrection
from single_integrator.c1.evaluate import C1Policy
from single_integrator.c1.risk.evaluation import audit_trace, risk_from_metadata
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.training.dataset_starts import load_dataset_starts
from single_integrator.c1.train import rollout_terms
from single_integrator.c1.socp import ExactProjection
from single_integrator.cbf import CBFConfig, CBFSafetyFilter
from single_integrator.environment import Config
from single_integrator.evaluate import load_policy, rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('trajectories', 'training'), required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('diagnostic output must be empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update('jax_enable_x64', True)
    path = ROOT/'results/c1_dataset_audit_seed0/residual.pkl'
    saved = pickle.loads(path.read_bytes())
    meta = saved['metadata']
    baseline, provenance = load_policy(Path(meta['baseline_checkpoint']))
    plant = Config(**provenance['evaluation_environment'])
    model = ResidualCorrection(hidden_dims=tuple(meta['architecture']['hidden_dims']),
                               layer_norm=meta['architecture']['layer_norm'])
    initial = model.init(jax.random.PRNGKey(meta['training']['seed']), jnp.zeros((1,4)),
                         jnp.zeros((1,1)), jnp.zeros((1,20)))
    final = jax.tree_util.tree_map(jnp.asarray, saved['params'])
    delta = jax.tree_util.tree_map(lambda a,b:a-b, final, initial)
    # The scale experiment is interpolation along the actual optimizer update.
    def parameters(scale):
        return jax.tree_util.tree_map(lambda p,d:p+scale*d, initial, delta)
    field = ResidualFlowField(baseline, model)
    field.baseline_sample = jax.jit(field.baseline_sample)
    field.correction = jax.jit(field.correction)
    cbf, risk = CBFConfig(), risk_from_metadata(meta)
    report = dict(part=args.part, residual_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  diagnostic_only=True, original_model_unchanged=True)
    if args.part == 'training':
        starts, _ = load_dataset_starts(ROOT/'datasets/give_way_si_short_v1', 'train', plant)
        rng = np.random.default_rng(meta['training']['seed'])
        first_ids = rng.integers(0, len(starts), size=2)
        ids = rng.integers(0, len(starts), size=2)
        if rng.bit_generator.state != saved['rng_state']:
            raise ValueError('replayed training RNG does not match saved checkpoint')
        key, draw = jax.random.split(jax.random.PRNGKey(meta['training']['seed']+1000))
        np.testing.assert_array_equal(key, saved['jax_key'])
        noise = jax.random.normal(draw, (2,850,4), dtype=jnp.float32)
        positions = jnp.asarray(starts[ids])
        projection = ExactProjection()
        def terms(scale):
            return rollout_terms(parameters(scale), field, projection, positions, noise,
                                 plant, cbf, risk, return_details=True)
        def objective(scale):
            return jnp.mean(terms(scale)[2])
        slope = float(jax.grad(objective)(0.))
        print(dict(directional_risk_derivative=slope), flush=True)
        rows = []
        for scale in (0., .01, .1, 1.):
            current, safe, risks, details = terms(scale)
            from single_integrator.c1.training.primal_dual import deviation_cost
            row = dict(scale=scale, J_live=float(jnp.mean(risks)),
                       J_def=float(deviation_cost(current,safe,step_mask=details['episode_mask'])),
                       risks=np.asarray(risks).tolist(), codes=np.asarray(details['terminal_codes']).tolist(),
                       lengths=np.asarray(details['episode_lengths']).tolist())
            rows.append(row)
            atomic_save(args.out_dir/'progress.json', rows)
            print(row, flush=True)
        np.testing.assert_allclose(rows[0]['J_live'], saved['history'][1]['J_live'], atol=1e-10,rtol=0)
        report.update(first_batch_pair_ids=first_ids.tolist(), updated_batch_pair_ids=ids.tolist(),
                      directional_derivative=slope, scales=rows,
                      hidden_layer_changes={k:float(sum(np.linalg.norm(np.asarray(v)) for v in jax.tree_util.tree_leaves(d)))
                                            for k,d in delta['params'].items()})
    else:
        plant = replace(plant, max_steps=1200)
        base_dir = ROOT/'results/c1_dataset_extended_h120_seed0_baselines/mac_cbf'
        c1_dir = ROOT/'results/c1_dataset_extended_h120_seed0_c1'
        base_summaries = json.loads((base_dir/'summary.json').read_text())['rollouts']
        c1_summaries = json.loads((c1_dir/'summary.json').read_text())['rollouts']
        rows = []
        for rid in range(25):
            with np.load(base_dir/f'rollout_{rid:04d}.npz',allow_pickle=False) as data:
                trace = {key:data[key] for key in ('positions_before','positions','executed_velocity',
                    'deadlock','task_success','wall_collision','agent_collision','min_swept_agent_distance','min_swept_wall_distance')}
            safe = np.asarray(trace['executed_velocity']).reshape(-1,4)
            trace.update(c1_safe=safe, c1_candidate=safe, c1_correction=np.zeros_like(safe))
            audited, _ = audit_trace(trace, plant, cbf, risk)
            rows.append(dict(rollout_id=rid, safety=audited, c1=c1_summaries[rid]['c1']))
        report['paired_risk'] = rows
        atomic_save(args.out_dir/'risk_comparison.json', rows)
        ablations = []
        for scale in (0., .25, .5, .75):
            policy = C1Policy(ResidualFlowField(baseline, model), parameters(scale), plant)
            ids = range(25) if scale==0 else (2,10)
            for rid in ids:
                policy.records.clear()
                summary, trace = rollout(policy, np.asarray(base_summaries[rid]['initial_positions']),
                    plant,42,rid,CBFSafetyFilter(cbf),cbf_config=cbf)
                entry = dict(scale=scale, rollout_id=rid, outcome=summary['outcome'],
                    steps=summary['episode_steps'], final_positions=trace['positions'][-1].tolist(),
                    final_goal_errors=trace['goal_errors'][-1].tolist(),
                    min_pairwise_h=summary['min_pairwise_h'], min_wall_h=summary['min_wall_h'])
                ablations.append(entry)
                print(entry, flush=True)
                atomic_save(args.out_dir/'ablations.json', ablations)
                if rid in (2,10):
                    np.savez_compressed(args.out_dir/f'case{rid}_scale{scale}.npz',
                        positions=trace['positions'], goal_errors=trace['goal_errors'],
                        max_speed=trace['max_speed'])
        report['ablations'] = ablations
        report['mean_safety_risk'] = float(np.mean([r['safety']['trajectory_risk'] for r in rows]))
        report['mean_c1_risk'] = float(np.mean([r['c1']['trajectory_risk'] for r in rows]))
        report['zero_residual_outcome_disagreements'] = [r['rollout_id'] for r in ablations if r['scale']==0 and
                                                    r['outcome']!=base_summaries[r['rollout_id']]['outcome']]
    atomic_save(args.out_dir/'report.json', report)
    print('COMPLETE', args.part, flush=True)


if __name__ == '__main__':
    main()
