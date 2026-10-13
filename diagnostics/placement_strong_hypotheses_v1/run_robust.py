"""Execute preflighted missing seeds for bounded-domain B15 confirmation."""
import argparse
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'))
from closed_loop_probe import simulate
from validation_stage2.run_rollouts import make_runtime
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True)
    args=ap.parse_args()
    p=json.loads((HERE/'robust_protocol.json').read_text())
    pre=json.loads((HERE/'robust_cache_preflight.json').read_text())
    if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(p['jobs']):raise RuntimeError('preflight mismatch')
    valid={(r['state_uid'],r['eta_uid'],r['controller_uid'],s) for r in pre['details'] for s in r['missing_seeds']}
    states={s['uid']:s for s in p['states']};out=HERE/'robust_results';out.mkdir(exist_ok=True)
    rt=make_runtime('ring_exchange');root=2026100601;count=0
    for i,j in enumerate(p['jobs']):
        if i%args.shards!=args.shard:continue
        seed_key=json.dumps(dict(future_root=root,seed=j['seed']),separators=(',',':'))
        if (j['uid'],j['eta_uid'],j['controller_uid'],seed_key) not in valid:raise RuntimeError('unplanned')
        stem=f"{j['uid']}_{j['chain']}_s{j['index']:02d}_seed{j['seed']:02d}"
        meta=out/(stem+'.json');arr=out/(stem+'.npz')
        if meta.exists() and arr.exists():continue
        arrays,r=simulate(rt,states[j['uid']],j['chain'],j['eta'],j['seed'],root,rt.config.max_steps)
        r.update(point_index=j['index'],eta_uid=j['eta_uid'])
        np.savez_compressed(arr,**arrays);meta.write_text(json.dumps(r,indent=2)+'\n')
        count+=1
        if count%25==0:print(json.dumps(dict(shard=args.shard,complete=count,last=i)),flush=True)
    print(json.dumps(dict(shard=args.shard,complete=count,done=True)),flush=True)


if __name__=='__main__':main()
