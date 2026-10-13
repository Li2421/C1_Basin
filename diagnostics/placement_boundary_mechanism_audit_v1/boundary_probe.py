"""Replay cached bracket endpoints and evaluate three preflighted interior points."""
import argparse
import json
from pathlib import Path
import numpy as np
from closed_loop_probe import HERE,HARD,make_runtime,margin,simulate


def replay(rt,state,case_id):
    with np.load(HARD/'results'/f'{case_id}.npz') as z:
        actions=z['actions'].copy();positions=z['positions'].copy()
    record=json.loads((HARD/'results'/f'{case_id}.json').read_text())
    env=rt.make_env(state);goals=[];safety=[];term='running'
    for a in actions:
        _,_,done,info=env.step(a)
        g,s=margin(state['scenario'],env,info);goals.append(g);safety.append(s)
        term=info['termination']
        if done:break
    if not np.allclose(positions[:len(goals)+1][-1],env.positions,atol=1e-12,rtol=0):
        raise RuntimeError('cached physical replay mismatch')
    if term!=record['termination']:raise RuntimeError('cached termination mismatch')
    rho=min(max(goals),min(safety))
    return dict(rho=float(rho),goal_ever_margin=float(max(goals)),safety_always_margin=float(min(safety)),
                success=record['success'],termination=term,steps=len(goals),source_case_id=case_id)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--state-uid',required=True);args=ap.parse_args()
    protocol=json.loads((HERE/'boundary_protocol.json').read_text())
    segment=next(x for x in protocol if x['uid']==args.state_uid)
    states=json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']
    state=next(s for s in states if s['uid']==args.state_uid)
    pre=json.loads((HERE/'boundary_cache_preflight.json').read_text())
    if pre['summary']['ambiguous']:raise RuntimeError('unresolved preflight')
    rt=make_runtime(state['scenario'])
    outdir=HERE/'boundary_results';outdir.mkdir(exist_ok=True)
    future_root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    result=dict(segment=segment,placements={})
    for chain in ('TT','FF'):
        records=[]
        for idx,case_id in zip(segment['endpoints'],segment['endpoint_case_ids'][chain]):
            records.append(dict(fraction=0. if idx==segment['endpoints'][0] else 1.,**replay(rt,state,case_id)))
        for point in segment['grid']:
            stem=f"{state['uid']}_{chain}_t{point['fraction']}"
            meta=outdir/(stem+'.json');npz=outdir/(stem+'.npz')
            if meta.exists() and npz.exists():r=json.loads(meta.read_text())
            else:
                arrays,r=simulate(rt,state,chain,point['eta'],0,future_root,rt.config.max_steps)
                np.savez_compressed(npz,**arrays)
                meta.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
            records.append(dict(fraction=point['fraction'],rho=r['rho'],goal_ever_margin=r['goal_ever_margin'],
                safety_always_margin=r['safety_always_margin'],success=r['success'],termination=r['termination'],steps=r['steps']))
            print(args.state_uid,chain,point['fraction'],r['termination'],r['rho'],flush=True)
        result['placements'][chain]=sorted(records,key=lambda x:x['fraction'])
    (outdir/f"{state['uid']}_grid.json").write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
