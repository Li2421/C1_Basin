"""Deterministic outcome-aware Phase-B plan, frozen before Phase-B rollouts."""
import json
from itertools import combinations
import numpy as np
from diagnostics.success_basin_multimodality.setup import HERE,write,sha

def main():
    p=json.loads((HERE/'protocol.json').read_text());cc=json.loads((HERE/'connected_components.json').read_text())
    common=[tuple(x) for x in cc['common_success_cells_all_three_states']]
    assert (1.,0.,.25) in common and len(common)>1
    pairs=sorted(combinations(common,2))
    a,b=max(pairs,key=lambda q:(float(np.linalg.norm(np.asarray(q[0])-q[1])),q))
    states=[x['state_id'] for x in p['primary_states']]
    # Same original jobs: only replace the two invalid numerical attempts.
    repairs=[]
    for sid in states:
        m=json.loads((HERE/'raw'/f'phase_a_{sid}'/'manifest.json').read_text())
        for r in m['records']:
            if r['outcome'] is None:repairs.append({k:r[k] for k in ['state_id','eta','seed','cell_id']})
    validations=[{'state_id':sid,'eta':[1.,0.,.25],'seed':seed,'cell_id':'known_success_fresh'}
                 for sid in states for seed in range(95102001,95102033)]
    interpolation=[]
    for sid in states:
        for lam in p['path_lambda']:
            eta=((1-lam)*np.asarray(a)+lam*np.asarray(b)).tolist()
            for seed in range(95103001,95103033):
                interpolation.append({'state_id':sid,'eta':eta,'seed':seed,'cell_id':f'lambda_{lam:.1f}','lambda':lam})
    directions=np.r_[np.eye(3),-np.eye(3)];deltas=[.125,.375,.625,.875];center=np.asarray([1.,0.,.25])
    margin=[]
    for sid in states:
        for did,direction in enumerate(directions):
            for delta in deltas:
                eta=(center+delta*direction).tolist()
                for seed in range(95104001,95104017):
                    margin.append({'state_id':sid,'eta':eta,'seed':seed,
                      'cell_id':f'direction_{did}_delta_{delta:.3f}','direction_id':did,'delta':delta})
    write('repair_jobs.json',repairs);write('known_validation_jobs.json',validations)
    write('interpolation_jobs.json',interpolation);write('margin_jobs.json',margin)
    plan={'selection_rule':'Farthest Euclidean pair in intersection of all three Phase-A SUCCESS_CELL sets; lexicographic deterministic tie break.',
      'eta_A':list(a),'eta_B':list(b),'distance':float(np.linalg.norm(np.asarray(a)-b)),
      'repair_rollouts':len(repairs),'known_validation_rollouts':len(validations),
      'straight_interpolation_rollouts':len(interpolation),'margin_rollouts':len(margin),
      'phase_b_total':sum(map(len,[repairs,validations,interpolation,margin])),
      'conditional_next':'Only if straight interpolation is not a high-confidence success path, validate a shortest Phase-A graph path with fresh seeds.',
      'phase_a_map_sha256':sha(HERE/'success_map.json'),'components_sha256':sha(HERE/'connected_components.json')}
    write('phase_b_plan.json',plan);print(json.dumps(plan,indent=2))

if __name__=='__main__':main()
