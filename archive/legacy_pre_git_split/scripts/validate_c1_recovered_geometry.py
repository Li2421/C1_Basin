"""Recover the same g on ten branches and compare previously valid frames."""
import json
import hashlib
import numpy as np
from audit_c1_equivalent_geometry_solver import BASE,OUT,stable_score
from audit_c1_joint_witness_risk import ROOT,save
from audit_c1_completed_waiting_risk import MAXR
from audit_c1_candidate_coverage import metrics
from single_integrator.environment import Config,GiveWayEnv
from single_integrator.cbf import CBFConfig,barrier_constraints

def main():
    rows=json.loads((BASE/'rows.json').read_text());env=GiveWayEnv(Config(corridor_half_length=1.3));cfg=CBFConfig()
    reports=[];deltas=[];updates={};(OUT/'recovered').mkdir(exist_ok=True)
    for r in rows:
        if not r.get('geometry_errors'):continue
        z0=np.load(BASE/'traces'/f"{r['rid']:04d}_{r['cid']}.npz");z={k:z0[k] for k in z0.files};new=[];errors=[]
        for j,t in enumerate(range(100,300)):
            A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cfg)
            result=stable_score(A,b,z['candidate'][t])
            assert result['valid'];g=result['risk']/MAXR;new.append(g)
            if np.isfinite(z['g'][j]):deltas.append(abs(g-z['g'][j]))
        old=z['g'];z['g']=np.asarray(new);costs,outcome=metrics(z)
        assert costs is not None
        update=dict(rid=r['rid'],cid=r['cid'],costs=costs,outcome=outcome,recovered_frames=int(np.sum(~np.isfinite(old))),
            max_previously_valid_g_difference=float(np.max(np.abs(z['g'][np.isfinite(old)]-old[np.isfinite(old)]))),
            mathematical_change=False)
        reports.append(update);updates[r['cid']]=update
        np.savez_compressed(OUT/'recovered'/f"{r['rid']:04d}_{r['cid']}.npz",g=z['g'])
        print(r['cid'],costs['score'],update['max_previously_valid_g_difference'],flush=True)
    ranking={}
    for rid in [68,208]:
        rr=[updates.get(r['cid'],r) if r['rid']==68 else r for r in rows if r['rid']==rid]
        eligible=[r for r in rr if r['costs'] is not None];ranked=sorted(eligible,key=lambda r:(r['costs']['score'],r['cid']))
        ranking[rid]=dict(eligible=len(eligible),best=ranked[0],successes=sum(r['outcome']['success'] for r in rr),
            recovered_candidate_ranks={r['cid']:i+1 for i,r in enumerate(ranked) if r['cid'] in updates and rid==68})
    oldhash=json.loads((BASE/'verification.json').read_text())['hashes']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in oldhash};assert hashes==oldhash
    save(OUT/'recovered_branch_scores.json',reports)
    save(OUT/'recovery_validation.json',dict(recovered_branches=len(reports),recovered_frames=sum(r['recovered_frames'] for r in reports),
        previously_valid_frames_compared=len(deltas),max_previously_valid_g_difference=max(deltas),
        all_frames_recomputed=len(reports)*200,ranking_diagnostic_only=ranking,core_hashes_unchanged=True,
        numerical_change='Only base-QP internal equilibration disabled, same objective/constraints/tolerances/maxiter. Physical witness solver and risk formulas unchanged.'))

if __name__=='__main__':main()
