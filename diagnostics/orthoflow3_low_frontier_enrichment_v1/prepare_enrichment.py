#!/usr/bin/env python3
"""Freeze the shared, out-of-design Sobol screening plan before rollouts."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=ROOT/'diagnostics/orthoflow3_low_frontier_enrichment_v1'
OLD=ROOT/'diagnostics/orthoflow3_analytic_mindef_stability_v1'
LO=np.array([.5,-.5,0.]); HI=np.array([1.25,.5,.75]); SEED=810031
def dig(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (OUT/'active_state_manifest.json').exists(): raise RuntimeError('already frozen')
    states=json.loads((OLD/'state_manifest.json').read_text())['states']
    proxy=list(csv.DictReader(open(OLD/'candidate_proxy_values.csv')))
    active=sorted({r['state_id'] for r in proxy if r['stratum']=='ACTIVE_REQUIRED'})
    selected=[s for s in states if s['state_id'] in active]
    if len(active)!=10 or len(selected)!=10: raise RuntimeError((len(active),len(selected)))
    # Regenerate the authoritative scrambled Sobol sequence continuously. The old
    # design consumed indices 0..254 plus an embedded non-Sobol point; use 256..287.
    unit=qmc.Sobol(3,scramble=True,seed=SEED).random_base2(9)
    rows=[]
    for i in range(256,288):
        eta=(LO+(HI-LO)*unit[i]).tolist()
        rows.append({'sequence_rank':i-255,'sobol_index':i,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],
                     'generator':'scipy.qmc.Sobol(d=3,scramble=True,seed=810031).random_base2(9)[index]',
                     'selection_phase':'initial16' if i<272 else 'reserved_extension16'})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'enrichment_eta_sequence.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    manifest={'schema':'orthoflow3_low_frontier_active_states_v1','selection':'exactly the ten ACTIVE_REQUIRED state_ids from the completed analytic-min-def audit; no additions/removals','states':selected,
              'authoritative_domain':{'low':LO.tolist(),'high':HI.tolist()},'sobol_seed':SEED,'sobol_indices':[256,287],
              'design_note':'indices 0..254 were already in the historical common design; index 255 was an embedded legacy point. This audit starts at continuous Sobol index 256.',
              'candidate_sequence_sha256':sha(OUT/'enrichment_eta_sequence.csv')}
    (OUT/'active_state_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    arms=[]
    for s in selected:
      for point in rows[:16]:
        arms.append({'arm_id':f"SCREEN__{s['state_id']}__I{point['sobol_index']:03d}",'basis_family':'orthoflow3','state_id':s['state_id'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta':[point['eta1'],point['eta2'],point['eta3']],'sobol_index':point['sobol_index'],'seeds':[int(x) for x in s['matched_flow_seeds'][:8]],'role':'shared_sobol_screen','anchor_rank':s['pair_rank'],'offset_steps':4 if s['side']=='neighbor_t+4' else 0,'probe_id':'SCREEN'})
    plan={'schema':'orthoflow3_low_frontier_screen_v1','basis_family':'orthoflow3','stage':'screen_initial16','selection':'same predeclared continuous Sobol indices 256..271 for every frozen active state; 8 matched futures; no J-dependent choice','arms':arms}; plan['content_sha256']=dig(plan)
    (OUT/'screen_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    protocol='''# OrthoFlow3 low-frontier enrichment audit\n\nThis confirmation audit reuses exactly the ten states designated `ACTIVE_REQUIRED` by `orthoflow3_analytic_mindef_stability_v1`. It freezes a shared scrambled-Sobol extension (seed 810031; indices 256–287 in the authoritative eta domain) before any new outcomes. Initial screening evaluates indices 256–271 with the same first eight matched Flow futures for every state. Promotion does not consult true rollout J: A is the lowest raw eta norm among 8/8 candidates, B the lowest start-Gram proxy among remaining candidates, with the predeclared farthest-normalized-distance fallback. Candidate success is confirmed only at B63 (>=63/64); true J is the prior authoritative mean successful full-continuation J_def.\n'''
    (OUT/'protocol.md').write_text(protocol)
    print(json.dumps({'states':len(selected),'screen_arms':len(arms),'new_continuations':sum(len(a['seeds']) for a in arms),'worst_case_steps':sum((850-a['absolute_step'])*len(a['seeds']) for a in arms)},indent=2))
if __name__=='__main__': main()
