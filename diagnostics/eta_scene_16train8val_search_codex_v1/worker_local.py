from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('shard',type=int);p.add_argument('shards',type=int);args=p.parse_args()
if (HERE/'USER_ANALYSIS_ONLY.receipt.json').exists():raise SystemExit('User requested analysis only; rollout/replay disabled.')
# All numerical threads share exactly one CPU; no hidden BLAS/JAX parallel budget.
avail=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{avail[args.shard%len(avail)]})
if args.stage=='certify':
 from certify import check_record as one
 tasks=read(HERE/'certify_TASKS.json')
 for t in tasks[args.shard::args.shards]:one(t)
elif args.stage=='new_pairs':
 from pair_worker import one
 tasks=read(HERE/'new_pairs_MISSING_CASES.json')
 for t in tasks[args.shard::args.shards]:one(t)
else:
 from rollout_worker import one
 import accel_mode
 assert read(HERE/'ACCELERATION_CERTIFICATE.json')['passed'];accel_mode.CHECK=False
 tasks=read(HERE/f'{args.stage}_MISSING_CASES.json')
 for t in tasks[args.shard::args.shards]:one(args.stage,t,None)
print('SHARD_COMPLETE',args.stage,args.shard,flush=True)
