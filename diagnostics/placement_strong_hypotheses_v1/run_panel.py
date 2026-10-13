"""Run only missing preflighted, paired full rollouts for the frozen panel."""
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
PRIOR=ROOT/'diagnostics/placement_boundary_mechanism_audit_v1'
sys.path.insert(0,str(PRIOR))
from closed_loop_probe import HARD,simulate
from validation_stage2.run_rollouts import make_runtime
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shard',type=int,required=True)
    ap.add_argument('--shards',type=int,required=True)
    ap.add_argument('--protocol',default='protocol.json')
    ap.add_argument('--preflight',default='cache_preflight.json')
    ap.add_argument('--outdir',default='panel_results')
    args=ap.parse_args()
    protocol=json.loads((HERE/args.protocol).read_text())
    pre=json.loads((HERE/args.preflight).read_text())
    if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(protocol['cases']):
        raise RuntimeError('shared-cache preflight mismatch')
    allowed={(x['state_uid'],x['controller_uid'],x['eta_uid'])
             for x in pre['details'] if x['status']=='INCOMPATIBLE' and x['missing_seeds']}
    states={s['uid']:s for s in protocol['states']}
    out=HERE/args.outdir;out.mkdir(exist_ok=True)
    rt=None;scene=None;completed=0
    for i,c in enumerate(protocol['cases']):
        if i%args.shards!=args.shard:continue
        if (c['uid'],c['controller_uid'],c['eta_uid']) not in allowed:
            raise RuntimeError('case not authorized as missing by preflight: '+str(i))
        stem=f"{c['uid']}_{c['kind']}_{c['index']:03d}_{c['chain']}"
        meta=out/(stem+'.json');arrpath=out/(stem+'.npz')
        if meta.exists() and arrpath.exists():continue
        if c['scenario']!=scene:
            scene=c['scenario'];rt=make_runtime(scene)
        arrays,record=simulate(rt,states[c['uid']],c['chain'],c['eta'],0,protocol['future_root'],rt.config.max_steps)
        record.update(kind=c['kind'],point_index=c['index'],eta_uid=c['eta_uid'],controller_uid=c['controller_uid'],
                      protocol_case_index=i)
        if record['termination']=='numerical_failure':
            print('NUMERICAL_FAILURE',i,record['error'],flush=True)
        np.savez_compressed(arrpath,**arrays)
        meta.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
        completed+=1
        if completed%20==0:
            print(json.dumps(dict(shard=args.shard,completed=completed,last=i)),flush=True)
    print(json.dumps(dict(shard=args.shard,completed=completed,done=True)),flush=True)


if __name__=='__main__':main()
