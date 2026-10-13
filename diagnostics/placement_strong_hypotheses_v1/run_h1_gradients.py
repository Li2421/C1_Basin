"""Run paired fixed-key sensitivity probes on regular Ring boundary pairs."""
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
    p=json.loads((HERE/'h1_gradient_protocol.json').read_text());pre=json.loads((HERE/'h1_gradient_cache_preflight.json').read_text())
    if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(p['jobs']):raise RuntimeError('preflight mismatch')
    valid={(r['state_uid'],r['eta_uid'],r['controller_uid']) for r in pre['details'] if r['missing_seeds']}
    states={s['uid']:s for s in json.loads((Path('/home/zhihan/research/Basin_C1_hard_field_20261006')/'confirmation_manifest.json').read_text())['states']}
    rt=make_runtime('ring_exchange');out=HERE/'h1_gradient_results';out.mkdir(exist_ok=True)
    for i,j in enumerate(p['jobs']):
        if i%args.shards!=args.shard:continue
        if (j['uid'],j['eta_uid'],j['controller_uid']) not in valid:raise RuntimeError('unplanned')
        stem=f"{j['uid']}_root{j['root_chain']}_eval{j['eval_chain']}_{j['kind']}_e{j['epsilon']}_a{j['axis']}_s{j['sign']}"
        meta=out/(stem+'.json');arr=out/(stem+'.npz')
        if meta.exists() and arr.exists():continue
        arrays,r=simulate(rt,states[j['uid']],j['eval_chain'],j['eta'],0,2026100601,rt.config.max_steps)
        r.update(root_chain=j['root_chain'],eval_chain=j['eval_chain'],kind=j['kind'],epsilon=j['epsilon'],
                 axis=j['axis'],sign=j['sign'],eta_uid=j['eta_uid'])
        np.savez_compressed(arr,**arrays);meta.write_text(json.dumps(r,indent=2)+'\n')
        if i%10==0:print(i,j['uid'],j['root_chain'],j['eval_chain'],r['rho'],r['termination'],flush=True)


if __name__=='__main__':main()
