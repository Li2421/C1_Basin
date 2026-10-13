"""Run only the preflighted curvature probes with paired frozen Flow keys."""
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
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',default='curvature_protocol.json')
    ap.add_argument('--preflight',default='curvature_cache_preflight.json')
    ap.add_argument('--outdir',default='curvature_results')
    args=ap.parse_args()
    protocol=json.loads((HERE/args.protocol).read_text())
    pre=json.loads((HERE/args.preflight).read_text())
    if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(protocol['jobs']):
        raise RuntimeError('curvature preflight mismatch')
    valid={(r['state_uid'],r['eta_uid'],r['controller_uid']) for r in pre['details'] if r['missing_seeds']}
    states={s['uid']:s for s in json.loads((Path('/home/zhihan/research/Basin_C1_hard_field_20261006')/'confirmation_manifest.json').read_text())['states']}
    rt=make_runtime('ring_exchange');out=HERE/args.outdir;out.mkdir(exist_ok=True)
    for i,j in enumerate(protocol['jobs']):
        if (j['uid'],j['eta_uid'],j['controller_uid']) not in valid:raise RuntimeError('unplanned')
        stem=f"{j['uid']}_{j['chain']}_r{j['radius']}_{j['offset_index']}"
        meta=out/(stem+'.json');arr=out/(stem+'.npz')
        if meta.exists() and arr.exists():continue
        arrays,r=simulate(rt,states[j['uid']],j['chain'],j['eta'],0,2026100601,rt.config.max_steps)
        r.update(radius=j['radius'],offset_index=j['offset_index'],uv=j['uv'],eta_uid=j['eta_uid'])
        np.savez_compressed(arr,**arrays);meta.write_text(json.dumps(r,indent=2)+'\n')
        if i%16==0:print(i,j['uid'],j['chain'],j['radius'],r['rho'],r['termination'],flush=True)


if __name__=='__main__':main()
