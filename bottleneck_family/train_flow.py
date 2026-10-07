"""Train the same official joint MACFlow architecture used by the four scenes."""
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
    parser.add_argument('--near-goal-fraction',type=float,default=0.0)
    parser.add_argument('--source-balanced-sampling',action='store_true')
    args=parser.parse_args()
    result=train_stage1(args.dataset,args.output,seed=args.seed,steps=args.steps,
                        batch_size=args.batch_size,log_interval=max(50,args.steps//20),
                        early_transition_fraction=.2,early_steps=50,
                        allow_non_four_agents=True,
                        snapshot_interval=args.snapshot_interval,
                        near_goal_fraction=args.near_goal_fraction,
                        source_balanced_sampling=args.source_balanced_sampling)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
