"""Held-out seed-0 TT(Q eta) versus cached FF(eta), no refitting Q."""
import json
import sqlite3
import sys
import argparse
from pathlib import Path
import numpy as np
from closed_loop_probe import HERE,HARD,make_runtime,simulate
from boundary_probe import replay

sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stable',action='store_true');args=ap.parse_args()
    tag='fixed_q_stable_basin' if args.stable else 'fixed_q_basin'
    qdata=json.loads((HERE/('stable_only_fixed_Q.json' if args.stable else 'fixed_Q_train_test.json')).read_text())
    states=json.loads((HARD/'confirmation_manifest.json').read_text())['states']
    state_by_uid={s['uid']:s for s in states}
    etas=np.asarray(json.loads((HARD/'screening_manifest.json').read_text())['eta'])
    con=sqlite3.connect(HARD/'rollout.sqlite')
    jobs=[];requests=[]
    for scene in ('toy_give_way','ring_exchange'):
        uid=qdata[scene]['test_uids'][0]
        st=state_by_uid[uid]
        q=np.asarray(qdata[scene]['Q'])
        tt_ctl=con.execute('select controller_uid from rollout where state_uid=? and chain="TT" limit 1',(uid,)).fetchone()[0]
        for i,eta in enumerate(etas):
            transformed=(q@eta).tolist();eid=eta_identity(transformed)[0]
            row=con.execute('select case_id,result_json from rollout where state_uid=? and chain="FF" and eta_index=? and seed=0',(uid,i)).fetchone()
            if row is None:raise RuntimeError('missing frozen FF label')
            jobs.append(dict(scene=scene,state_uid=uid,eta_index=i,eta=eta.tolist(),transformed_eta=transformed,
                             eta_uid=eid,FF_case_id=row[0],FF_cached_success=json.loads(row[1])['success']))
            requests.append(dict(state_uid=uid,eta_uid=eid,controller_uid=tt_ctl,
                seed_keys=['{"future_root":2026100601,"seed":0}']))
    con.close()
    (HERE/f'{tag}_protocol.json').write_text(json.dumps(jobs,indent=2)+'\n')
    plan=HERE/f'{tag}_planned_rollouts.json';plan.write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(plan)
    (HERE/f'{tag}_cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous']:raise RuntimeError('ambiguous cache preflight')
    root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    outdir=HERE/f'{tag}_results';outdir.mkdir(exist_ok=True)
    rows=[]
    for scene in ('toy_give_way','ring_exchange'):
        rt=make_runtime(scene)
        for job in [r for r in jobs if r['scene']==scene]:
            uid=job['state_uid'];st=state_by_uid[uid];idx=job['eta_index']
            meta=outdir/f'{uid}_eta{idx}.json';npz=outdir/f'{uid}_eta{idx}.npz'
            if meta.exists() and npz.exists():r=json.loads(meta.read_text())
            else:
                arrays,r=simulate(rt,st,'TT',job['transformed_eta'],0,root,rt.config.max_steps)
                np.savez_compressed(npz,**arrays);meta.write_text(json.dumps(r,indent=2)+'\n')
            ff=replay(rt,st,job['FF_case_id'])
            rows.append(dict(**job,TT_Q_success=r['success'],TT_Q_rho=r['rho'],TT_Q_term=r['termination'],
                             FF_rho=ff['rho'],FF_term=ff['termination'],label_match=bool(r['success']==ff['success']) if r['success'] is not None else None))
            print(scene,uid,idx,'FF',ff['termination'],'TTQ',r['termination'],'rho',ff['rho'],r['rho'],flush=True)
    (HERE/f'{tag}_results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
