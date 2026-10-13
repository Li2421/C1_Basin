"""Preflight each deterministic midpoint round on six Ring Q16 lines."""
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
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight
from validation_stage2.run_rollouts import make_runtime
import numpy as np


def crossing(row):return (row[0]['rho']>0)!=(row[1]['rho']>0)


def main():
    protocol=json.loads((HERE/'h1_line_protocol.json').read_text())
    endpoints=json.loads((HERE/'h1_endpoint_replay.json').read_text())
    states={s['uid']:s for s in json.loads((Path('/home/zhihan/research/Basin_C1_hard_field_20261006')/'confirmation_manifest.json').read_text())['states']}
    series={}
    for line in protocol['lines']:
        if not line['available']:continue
        uid=line['uid']
        for ch in ('TT','FF'):
            rows=[dict(fraction=x['fraction'],rho=x['rho'],termination=x['termination'])
                  for x in endpoints if x['uid']==uid and x['chain']==ch]
            for n in range(1,16):
                r=json.loads((HERE/'h1_line_results'/f'{uid}_{ch}_t{n:02d}.json').read_text())
                rows.append(dict(fraction=r['fraction'],rho=r['rho'],termination=r['termination']))
            rows.sort(key=lambda x:x['fraction']);series[(uid,ch)]=rows
    initial_crossings={str(key):sum(crossing(pair) for pair in zip(v[:-1],v[1:])) for key,v in series.items()}
    rt=make_runtime('ring_exchange');out=HERE/'h1_refine_results';out.mkdir(exist_ok=True)
    for iteration in range(1,4):
        jobs=[]
        for line in protocol['lines']:
            if not line['available']:continue
            a,b=np.asarray(line['endpoint_eta'],float)
            for chain in ('TT','FF'):
                rows=series[(line['uid'],chain)]
                brackets=[pair for pair in zip(rows[:-1],rows[1:]) if crossing(pair)]
                if not brackets:continue
                l,r=brackets[0];t=(l['fraction']+r['fraction'])/2
                eta=((1-t)*a+t*b).tolist()
                jobs.append(dict(uid=line['uid'],chain=chain,fraction=t,eta=eta,eta_uid=eta_identity(eta)[0],
                                 controller_uid=line['controllers'][chain]))
        path=HERE/f'h1_refine_round{iteration}_planned_rollouts.json'
        path.write_text(json.dumps(dict(requests=[dict(state_uid=x['uid'],eta_uid=x['eta_uid'],
            controller_uid=x['controller_uid'],seed_keys=['{"future_root":2026100601,"seed":0}']) for x in jobs]),indent=2)+'\n')
        pre=preflight(path);(HERE/f'h1_refine_round{iteration}_cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
        if pre['summary']['ambiguous'] or pre['summary']['total_requested']!=len(jobs):raise RuntimeError('preflight mismatch')
        for j in jobs:
            stem=f"{j['uid']}_{j['chain']}_r{iteration}_{j['fraction']:.8f}"
            meta=out/(stem+'.json');arr=out/(stem+'.npz')
            if meta.exists() and arr.exists():r=json.loads(meta.read_text())
            else:
                arrays,r=simulate(rt,states[j['uid']],j['chain'],j['eta'],0,2026100601,rt.config.max_steps)
                r.update(fraction=j['fraction'],eta_uid=j['eta_uid'],round=iteration)
                np.savez_compressed(arr,**arrays);meta.write_text(json.dumps(r,indent=2)+'\n')
            series[(j['uid'],j['chain'])].append(dict(fraction=r['fraction'],rho=r['rho'],termination=r['termination']))
            series[(j['uid'],j['chain'])].sort(key=lambda x:x['fraction'])
            print(iteration,j['uid'],j['chain'],j['fraction'],r['rho'],flush=True)
    rows=[]
    for line in protocol['lines']:
        if not line['available']:continue
        for chain in ('TT','FF'):
            seq=series[(line['uid'],chain)]
            bracket=next((pair for pair in zip(seq[:-1],seq[1:]) if crossing(pair)),None)
            if bracket is None:continue
            a,b=bracket;t=a['fraction']+(b['fraction']-a['fraction'])*(-a['rho'])/(b['rho']-a['rho'])
            rows.append(dict(uid=line['uid'],chain=chain,crossing_count_on_original_grid=initial_crossings[str((line['uid'],chain))],
                first_root_fraction=float(t),bracket=bracket,bracket_width=b['fraction']-a['fraction'],
                endpoint_eta=line['endpoint_eta'],all_samples=seq))
    (HERE/'h1_roots.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
