"""Held-out diagonal eta perturbation for one-step, trajectory and rho predictions."""
import json
import sys
from pathlib import Path
import numpy as np
from closed_loop_probe import HERE,HARD,make_runtime,simulate

sys.path.insert(0,str(HERE.parents[1]))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight


def main():
    states=json.loads((HERE/'closed_loop_protocol.json').read_text())['selected_states']
    qdata=json.loads((HERE/'fixed_Q_train_test.json').read_text())
    direction=(np.ones(3)/np.sqrt(3)).tolist();eps=.001
    root=json.loads((HARD/'screening_manifest.json').read_text())['future_root']
    etas=[('zero',[0.,0.,0.]),('candidate4',json.loads((HARD/'screening_manifest.json').read_text())['eta'][4])]
    import sqlite3
    con=sqlite3.connect(HARD/'rollout.sqlite')
    jobs=[];req=[]
    for st in states:
        for chain in ('TT','FF'):
            ctl=con.execute('select controller_uid from rollout where state_uid=? and chain=? limit 1',(st['uid'],chain)).fetchone()[0]
            for center,eta0 in etas:
                eta=(np.asarray(eta0)+eps*np.asarray(direction)).tolist();eid=eta_identity(eta)[0]
                jobs.append(dict(state=st,chain=chain,center=center,epsilon=eps,direction=direction,eta=eta,eta_uid=eid))
                req.append(dict(state_uid=st['uid'],eta_uid=eid,controller_uid=ctl,
                    seed_keys=['{"future_root":2026100601,"seed":0}']))
    con.close()
    (HERE/'heldout_prediction_protocol.json').write_text(json.dumps(jobs,indent=2)+'\n')
    p=HERE/'heldout_prediction_planned_rollouts.json';p.write_text(json.dumps({'requests':req},indent=2)+'\n')
    c=preflight(p);(HERE/'heldout_prediction_cache_preflight.json').write_text(json.dumps(c,indent=2)+'\n')
    if c['summary']['ambiguous']:raise RuntimeError('ambiguous preflight')
    out=HERE/'heldout_prediction_results';out.mkdir(exist_ok=True)
    arrays=np.load(HERE/'trajectory_sensitivity_arrays.npz')
    margins=json.loads((HERE/'rho_gradients.json').read_text())
    rows=[]
    for sc in ('toy_give_way','ring_exchange'):
        rt=make_runtime(sc)
        for job in [x for x in jobs if x['state']['scenario']==sc]:
            st=job['state'];chain=job['chain'];center=job['center'];uid=st['uid']
            horizon=80 if center=='zero' else rt.config.max_steps
            folder=HERE/('results_h80' if center=='zero' else 'results_full')
            prefix=f'{uid}_{chain}_{center}_center'
            with np.load(folder/(prefix+'.npz')) as z:base_pos=z['positions'].copy()
            base=json.loads((folder/(prefix+'.json')).read_text())
            meta=out/f'{uid}_{chain}_{center}.json';npz=out/f'{uid}_{chain}_{center}.npz'
            if meta.exists() and npz.exists():r=json.loads(meta.read_text());a=np.load(npz);actual_pos=a['positions'].copy();a.close()
            else:
                a,r=simulate(rt,st,chain,job['eta'],0,root,horizon)
                np.savez_compressed(npz,**a);actual_pos=a['positions'];meta.write_text(json.dumps(r,indent=2)+'\n')
            s=arrays[f'{uid}_{chain}_{center}_0.003_position'];g=np.asarray(next(x for x in margins if x['uid']==uid and x['placement']==chain and x['center']==center)['gradient_eps003'])
            v=np.asarray(direction);n=min(len(base_pos),len(actual_pos),len(s))
            evaluations=[]
            for t in (1,10,40,80):
                if t>=n:continue
                actual=(actual_pos[t]-base_pos[t]).ravel();predict=(eps*s[t].reshape(-1,3)@v)
                evaluations.append(dict(step=t,actual_norm=float(np.linalg.norm(actual)),predicted_norm=float(np.linalg.norm(predict)),
                    absolute_error=float(np.linalg.norm(actual-predict)),
                    relative_error=float(np.linalg.norm(actual-predict)/max(np.linalg.norm(actual),1e-12)),
                    cosine=float(actual@predict/max(np.linalg.norm(actual)*np.linalg.norm(predict),1e-12))))
            drho=r['rho']-base['rho'];prho=float(eps*g@v)
            rows.append(dict(scene=sc,uid=uid,chain=chain,center=center,epsilon=eps,
                baseline_termination=base['termination'],actual_termination=r['termination'],
                trajectory=evaluations,actual_rho_change=drho,predicted_rho_change=prho,
                rho_error=abs(drho-prho),rho_relative_error=abs(drho-prho)/max(abs(drho),1e-12)))
            print(sc,uid,chain,center,evaluations[-1]['relative_error'],r['termination'],flush=True)
    (HERE/'heldout_predictions.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
