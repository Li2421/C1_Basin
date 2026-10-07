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
    args=parser.parse_args()
    result=train_stage1(args.dataset,args.output,seed=args.seed,steps=args.steps,
                        batch_size=args.batch_size,log_interval=max(50,args.steps//20),
                        early_transition_fraction=.2,early_steps=50,
                        allow_non_four_agents=True)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
