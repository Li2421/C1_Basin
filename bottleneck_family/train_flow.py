"""Train joint Stage-I MACFlow, optionally with a permutation-equivariant actor."""
import argparse,json
from pathlib import Path
from new_benchmark_common.training import train_stage1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--snapshot-interval',type=int)
    parser.add_argument('--early-transition-fraction',type=float,default=.2)
    parser.add_argument('--early-steps',type=int,default=50)
    parser.add_argument('--early-nominal-only',action='store_true')
    parser.add_argument('--near-goal-fraction',type=float,default=0.0)
    parser.add_argument('--near-goal-distance',type=float,default=1.0)
    parser.add_argument('--onset-recovery-fraction',type=float,default=0.0)
    parser.add_argument('--onset-recovery-steps',type=int,default=50)
    parser.add_argument('--terminal-recovery-fraction',type=float,default=0.0)
    parser.add_argument('--terminal-recovery-steps',type=int,default=150)
    parser.add_argument('--gate-recovery-fraction',type=float,default=0.0)
    parser.add_argument('--gate-recovery-steps',type=int,default=300)
    parser.add_argument('--late-postgate-recovery-fraction',type=float,default=0.0)
    parser.add_argument('--late-postgate-recovery-steps',type=int,default=300)
    parser.add_argument('--postgate-fraction',type=float,default=0.0)
    parser.add_argument('--postgate-min-goal-distance',type=float,default=1.0)
    parser.add_argument('--initial-nominal-fraction',type=float,default=0.0)
    parser.add_argument('--initial-nominal-steps',type=int,default=5)
    parser.add_argument('--source-balanced-sampling',action='store_true')
    parser.add_argument('--motion-loss-weight',type=float,default=1.0)
    parser.add_argument('--endpoint-action-loss-weight',type=float,default=0.0)
    parser.add_argument('--endpoint-motion-weight',type=float,default=1.0)
    parser.add_argument('--permutation-augmentation',action='store_true')
    parser.add_argument('--shared-agent-normalization',action='store_true')
    parser.add_argument('--active-action-normalization',action='store_true')
    parser.add_argument('--hidden-dims',type=int,nargs='+')
    parser.add_argument('--architecture',choices=('flat_mlp','set_attention'),default='flat_mlp')
    parser.add_argument('--set-width',type=int,default=128)
    parser.add_argument('--set-layers',type=int,default=2)
    parser.add_argument('--init-checkpoint',type=Path)
    parser.add_argument('--transfer-set-checkpoint',type=Path)
    args=parser.parse_args()
    result=train_stage1(args.dataset,args.output,seed=args.seed,steps=args.steps,
                        batch_size=args.batch_size,log_interval=max(50,args.steps//20),
                        early_transition_fraction=args.early_transition_fraction,
                        early_steps=args.early_steps,
                        early_nominal_only=args.early_nominal_only,
                        allow_non_four_agents=True,
                        snapshot_interval=args.snapshot_interval,
                        near_goal_fraction=args.near_goal_fraction,
                        near_goal_distance=args.near_goal_distance,
                        onset_recovery_fraction=args.onset_recovery_fraction,
                        onset_recovery_steps=args.onset_recovery_steps,
                        terminal_recovery_fraction=args.terminal_recovery_fraction,
                        terminal_recovery_steps=args.terminal_recovery_steps,
                        gate_recovery_fraction=args.gate_recovery_fraction,
                        gate_recovery_steps=args.gate_recovery_steps,
                        late_postgate_recovery_fraction=args.late_postgate_recovery_fraction,
                        late_postgate_recovery_steps=args.late_postgate_recovery_steps,
                        postgate_fraction=args.postgate_fraction,
                        postgate_min_goal_distance=args.postgate_min_goal_distance,
                        initial_nominal_fraction=args.initial_nominal_fraction,
                        initial_nominal_steps=args.initial_nominal_steps,
                        source_balanced_sampling=args.source_balanced_sampling,
                        motion_loss_weight=args.motion_loss_weight,
                        endpoint_action_loss_weight=args.endpoint_action_loss_weight,
                        endpoint_motion_weight=args.endpoint_motion_weight,
                        permutation_augmentation=args.permutation_augmentation,
                        shared_agent_normalization=args.shared_agent_normalization,
                        active_action_normalization=args.active_action_normalization,
                        actor_hidden_dims=args.hidden_dims,
                        architecture=args.architecture,
                        set_width=args.set_width,set_layers=args.set_layers,
                        init_checkpoint=args.init_checkpoint,
                        transfer_set_checkpoint=args.transfer_set_checkpoint)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
