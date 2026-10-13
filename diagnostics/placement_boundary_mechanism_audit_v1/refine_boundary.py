"""Preflight each adaptive midpoint, then bisect a seed-0 outcome bracket."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from closed_loop_probe import HERE,HARD,make_runtime,simulate

sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state-uid',required=True);ap.add_argument('--levels',type=int,default=5)
    args=ap.parse_args()
    st=next(x for x in json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states'] if x['uid']==args.state_uid)
    seg=next(x for x in json.loads((HERE/'boundary_protocol.json').read_text()) if x['uid']==args.state_uid)
    prior=json.loads((HERE/'boundary_results'/f"{st['uid']}_grid.json").read_text())
    rt=make_runtime(st['scenario']);root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    outdir=HERE/'boundary_results';all_rows={}
    for chain in ('TT','FF'):
        rows=prior['placements'][chain].copy()
        for iteration in range(args.levels):
            rows.sort(key=lambda x:x['fraction'])
            brackets=[(a,b) for a,b in zip(rows[:-1],rows[1:]) if (a['rho']>0)!=(b['rho']>0)]
            if len(brackets)!=1:break
            a,b=brackets[0];frac=(a['fraction']+b['fraction'])/2
            eta=((1-frac)*np.asarray(seg['endpoint_eta'][0])+frac*np.asarray(seg['endpoint_eta'][1])).tolist()
            eid=eta_identity(eta)[0]
            req={'requests':[{'state_uid':st['uid'],'eta_uid':eid,'controller_uid':seg['controller_uid'][chain],
                              'seed_keys':['{"future_root":2026100601,"seed":0}']}]}
            plan=outdir/f"{st['uid']}_{chain}_iteration{iteration}_preflight_request.json"
            plan.write_text(json.dumps(req,indent=2)+'\n')
            cache=preflight(plan)
            (outdir/f"{st['uid']}_{chain}_iteration{iteration}_preflight.json").write_text(json.dumps(cache,indent=2)+'\n')
            if cache['summary']['ambiguous']:raise RuntimeError('ambiguous cache match')
            stem=f"{st['uid']}_{chain}_t{frac:.10f}"
            meta=outdir/f'{stem}.json';npz=outdir/f'{stem}.npz'
            if meta.exists() and npz.exists():result=json.loads(meta.read_text())
            else:
                arrays,result=simulate(rt,st,chain,eta,0,root,rt.config.max_steps)
                np.savez_compressed(npz,**arrays)
                meta.write_text(json.dumps(result,indent=2)+'\n')
            row=dict(fraction=frac,rho=result['rho'],success=result['success'],termination=result['termination'],
                     steps=result['steps'],goal_ever_margin=result['goal_ever_margin'],safety_always_margin=result['safety_always_margin'])
            rows.append(row)
            print(st['uid'],chain,iteration,frac,result['termination'],result['rho'],flush=True)
        all_rows[chain]=sorted(rows,key=lambda x:x['fraction'])
    output=dict(segment=seg,placements=all_rows)
    (outdir/f"{st['uid']}_refined.json").write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
