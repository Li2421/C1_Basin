"""Frozen, paired first-terminal-event evaluation; no training or tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from single_integrator.cbf import CBFConfig, CBFSafetyFilter, CBFSolverError
from single_integrator.environment import Config
from single_integrator.evaluate import ROOT, load_policy, rollout
from single_integrator.outcomes import FIRST_EVENT_OUTCOMES, FIRST_EVENT_PROTOCOL, first_event_aggregate


def transition_matrix(baseline, filtered):
    if len(baseline) != len(filtered) or not baseline:
        raise ValueError('Expected matching nonempty paired episodes')
    counts=np.zeros((5,5),dtype=int)
    for a,b in zip(baseline,filtered):
        if a['rollout_id']!=b['rollout_id'] or a['initial_positions']!=b['initial_positions']:
            raise ValueError('Unmatched paired episode')
        counts[FIRST_EVENT_OUTCOMES.index(a['outcome']),FIRST_EVENT_OUTCOMES.index(b['outcome'])]+=1
    return dict(rows=list(FIRST_EVENT_OUTCOMES),columns=list(FIRST_EVENT_OUTCOMES),
                counts=counts.tolist(),row_conditional_rates=[(row/row.sum()).tolist() if row.sum() else [None]*5 for row in counts])


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=ROOT/'datasets/give_way_si_short_v1')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n_rollouts',type=int,default=25)
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--initial_states',type=Path,help='Explicit alternative initial-state suite; reported separately from nominal evaluation')
    p.add_argument('--max_steps',type=int,
                   help='Diagnostic-only horizon extension. Must not shorten the checkpoint protocol.')
    args=p.parse_args()
    if args.n_rollouts < 1:
        raise ValueError('n_rollouts must be positive')
    policy,provenance=load_policy(args.checkpoint)
    metadata=json.loads((args.dataset/'environment.json').read_text())
    if metadata['evaluation_environment']!=provenance['evaluation_environment']:
        raise ValueError('Checkpoint and evaluation dataset environments differ')
    config=Config(**provenance['evaluation_environment'])
    if args.max_steps is not None:
        if args.max_steps < config.max_steps:
            raise ValueError('Diagnostic horizon may only extend the frozen protocol')
        from dataclasses import replace
        config=replace(config,max_steps=args.max_steps)
    cbf=CBFConfig()  # Fixed before observing results; no parameter sweep CLI.
    initials=[]
    if not args.initial_states and args.n_rollouts > 25:
        raise ValueError('Dataset splits contain 25 saved initial pairs; provide --initial_states for a larger suite')
    if not args.initial_states:
        for rid in range(args.n_rollouts):
            pair=(200 if args.split=='val' else 225)+rid
            with np.load(args.dataset/f'raw/episode_{2*pair:04d}.npz') as z:
                initials.append(z['initial_positions'].copy())
    initial_suite=None
    if args.initial_states:
        with np.load(args.initial_states) as z:
            starts=z[args.split+'_initial_positions']
            if starts.shape[1:]!=(2,2) or len(starts)<args.n_rollouts or not np.isfinite(starts).all():
                raise ValueError('Invalid alternative initial-state suite')
            initials=list(starts[:args.n_rollouts].copy())
            initial_suite=dict(path=str(args.initial_states.resolve()),sha256=hashlib.sha256(args.initial_states.read_bytes()).hexdigest(),description=json.loads(str(z['metadata_json'])))
        from single_integrator.environment import GiveWayEnv
        probe=GiveWayEnv(config)
        for initial in initials:probe.reset(initial)
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Output directory must be empty')
    import inspect
    import sys
    source_paths={name:Path(__file__).with_name(name+'.py') for name in ['environment','evaluate','evaluate_cbf','outcomes','cbf','filters']}
    for name in ['giveway_flowbc_agent','utils.networks','utils.flax_utils']:
        source_paths[name]=Path(inspect.getfile(sys.modules[name]))
    contract=dict(outcome_protocol=FIRST_EVENT_PROTOCOL,method='stage_i_joint_flow_bc',
                  checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  policy_provenance=provenance,environment=config.to_dict(),environment_fingerprint=config.fingerprint,
                  cbf=cbf.to_dict(),seed=args.seed,split=args.split,n_rollouts=args.n_rollouts,
                  rng_protocol='fold_in(fold_in(PRNGKey(seed), rollout_id), step)',
                  initial_positions=np.asarray(initials).tolist(),
                  initial_suite=initial_suite,
                  source_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in source_paths.items()})
    write_json(out/'config.json',contract)
    arms={name:[] for name in ['mac_only','mac_cbf']}
    for name in arms:
        (out/name).mkdir()
    try:
        for rid,initial in enumerate(initials):
            traces={}
            for name in arms:
                summary,arrays=rollout(policy,initial,config,args.seed,rid,
                                       CBFSafetyFilter(cbf) if name=='mac_cbf' else None,cbf_config=cbf)
                arms[name].append(summary)
                traces[name]=arrays
                np.savez_compressed(out/name/f'rollout_{rid:04d}.npz',**arrays,initial_positions=initial)
                write_json(out/name/'summary.json',dict(aggregate=first_event_aggregate(arms[name]),rollouts=arms[name]))
                print(f"{name} rollout={rid} outcome={summary['outcome']} steps={summary['episode_steps']}",flush=True)
            # Before the first intervention both observations and nominal samples must be identical.
            changed=np.flatnonzero(traces['mac_cbf']['intervention_norm']>0)
            common=min(len(traces['mac_only']['u_nom']),len(traces['mac_cbf']['u_nom']))
            prefix=min(common,int(changed[0])+1) if len(changed) else common
            if not np.array_equal(traces['mac_only']['u_nom'][:prefix],traces['mac_cbf']['u_nom'][:prefix]):
                raise AssertionError('Common-random-number nominal prefix differs')
    except CBFSolverError as error:
        write_json(out/'experiment_error.json',dict(status=error.status,details=error.details,rollout_id=rid,arm=name))
        raise
    write_json(out/'comparison.json',transition_matrix(arms['mac_only'],arms['mac_cbf']))
    write_json(out/'debug.json',dict(simultaneous_collision_counts={name:sum(s['simultaneous_collision'] for s in summaries) for name,summaries in arms.items()}))
    write_json(out/'complete.json',dict(completed_pairs=args.n_rollouts))


if __name__=='__main__':
    main()
