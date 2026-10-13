"""Predeclare paired closed-loop probes and shared-cache preflight."""
import json
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.planner import preflight

STRONG=ROOT/'diagnostics/placement_strong_hypotheses_v1'

def main():
    p=json.loads((HERE/'protocol.json').read_text());states={s['uid']:s for s in p['states']}
    roots=json.loads((STRONG/'h1_roots.json').read_text())
    panel=json.loads((STRONG/'protocol.json').read_text())
    old_candidates=json.loads(Path(p['source_manifest']).read_text())['eta']
    jobs=[]
    for uid in ('hard_confirm_ring_1_920004','hard_confirm_ring_1_920005'):
        for chain in ('TT','FF'):
            root=next(r for r in roots if r['uid']==uid and r['chain']==chain)
            a,b=np.asarray(root['endpoint_eta'],float)
            t=root['first_root_fraction']
            for radius,sign in [(0.,0)]+[(h,s) for h in (.001,.003,.01) for s in (-1,1)]:
                for line_offset in (-.03,0.,.03):
                    eta=((1-t-line_offset)*a+(t+line_offset)*b).tolist()
                    jobs.append(dict(kind='state_margin',uid=uid,scene='ring_exchange',chain=chain,
                        radius=radius,sign=sign,eta=eta,eta_uid=eta_identity(eta)[0],
                        line_offset=line_offset,root_fraction=t,line_direction=(b-a).tolist()))
    for uid in ('hard_confirm_toy_1_820001','hard_confirm_toy_1_820002'):
        center=np.asarray(old_candidates[4],float)
        for chain in ('TT','FF'):
            for radius,sign in [(0.,0)]+[(h,s) for h in (.001,.003,.01) for s in (-1,1)]:
                for line_offset in (-.015,0.,.015):
                    eta=center+np.asarray([line_offset,0.,0.])
                    jobs.append(dict(kind='state_margin',uid=uid,scene='toy_give_way',chain=chain,
                        radius=radius,sign=sign,eta=eta.tolist(),eta_uid=eta_identity(eta.tolist())[0],
                        line_offset=line_offset,root_fraction=None,line_direction=[1.,0.,0.]))
    for uid in p['state_probe_uids']:
        st=states[uid]
        for chain in ('TT','FF'):
            jobs.append(dict(kind='nominal_B',uid=uid,scene=st['scenario'],chain=chain,
                eta=[0.,0.,0.],eta_uid=eta_identity([0.,0.,0.])[0]))
            for i in (4,8):
                eta=list(map(float,old_candidates[i]))
                for mode in ('live_B','frozen_B'):
                    jobs.append(dict(kind='basis_feedback',uid=uid,scene=st['scenario'],chain=chain,
                        mode=mode,candidate_index=i,eta=eta,eta_uid=eta_identity(eta)[0]))
    requests=[]
    for j in jobs:
        state_uid=j['uid'] if j['kind']!='state_margin' or j['radius']==0 else \
            f"{j['uid']}|pos_{j['radius']:g}_{j['sign']:+d}"
        ctl=f"property_scan_{j['kind']}_{j['chain']}"
        if j['kind']=='basis_feedback':ctl+=f"_{j['mode']}"
        requests.append(dict(state_uid=state_uid,eta_uid=j['eta_uid'],controller_uid=ctl,
            seed_keys=[json.dumps(dict(future_root=p['future_root'],seed=p['seed']),separators=(',',':'))]))
        j['request_state_uid']=state_uid;j['request_controller_uid']=ctl
    result=dict(jobs=jobs,selection='Predeclared first two regular Ring UID 4/5 and first two Toy states; no new outcome-based selection.',
                physical_state_perturbation='position only, same Flow key based on base UID',
                margin='same finite-horizon diagnostic rho as previous audits',
                note='state_margin uses no binary success for truncated paths; frozen_B may censor at nominal baseline length.')
    (HERE/'rollout_protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    path=HERE/'planned_rollouts.json';path.write_text(json.dumps(dict(requests=requests),indent=2)+'\n')
    pre=preflight(path);(HERE/'cache_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
    print('jobs',len(jobs),'preflight',pre['summary'])

if __name__=='__main__':main()
