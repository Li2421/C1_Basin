"""Stratified physical-state block bootstrap; no reruns, no case independence fiction."""
from core import *

def cluster_rate(rows,draws=5000):
    grouped={}
    for r in rows:grouped.setdefault(r['type_id'],{}).setdefault(r['state_index'],[]).append(int(r['success']))
    rng=np.random.default_rng(2026102804);values=np.zeros(draws);denominator=np.zeros(draws);blocks=0
    for t,states in sorted(grouped.items()):
        groups=[states[k]for k in sorted(states)];blocks+=len(groups)
        successes=np.asarray([sum(v)for v in groups]);sizes=np.asarray([len(v)for v in groups]);indices=rng.integers(len(groups),size=(draws,len(groups)))
        values+=successes[indices].sum(1);denominator+=sizes[indices].sum(1)
    samples=values/np.maximum(denominator,1.)
    return dict(physical_blocks=blocks,cases=len(rows),mean=sum(r['success']for r in rows)/len(rows)if rows else None,bootstrap95=np.quantile(samples,[.025,.975]).tolist()if rows else None,replicates=draws,scope='Resample physical states within each frozen scene; nominal-controller outcomes for the same state stay in one block. Conditional on these maps and the frozen candidate pools. Four blocks per type do not establish new-map generalization.')

def main(case_rows,fixed_rows,papers):
    result={}
    for j,rr in case_rows.items():result[f'family_{j}_oracle']=cluster_rate(rr)
    for j,rr in fixed_rows.items():result[f'family_{j}_fixed']=cluster_rate(rr)
    for method,rr in papers.items():result[method]=cluster_rate(rr)
    families=read(HERE/'FAMILIES_FROZEN.json')['families']
    b3=next((j for j,f in enumerate(families)if set(f['columns'])=={'G','P','R'}),None)
    if b3 is not None:
        for mode,collection in [('oracle',case_rows),('fixed',fixed_rows)]:
            reference={(r['type_id'],r['controller'],r['state_index']):r for r in collection[b3]}
            for j,rr in collection.items():
                grouped={}
                for r in rr:
                    key=(r['type_id'],r['controller'],r['state_index'])
                    if key not in reference:continue
                    grouped.setdefault(r['type_id'],{}).setdefault(r['state_index'],[]).append(int(r['success'])-int(reference[key]['success']))
                rng=np.random.default_rng(2026102805);num=np.zeros(5000);den=np.zeros(5000)
                observed=[]
                for states in grouped.values():
                    blocks=list(states.values());sums=np.array([sum(b)for b in blocks]);sizes=np.array([len(b)for b in blocks]);indices=rng.integers(len(blocks),size=(5000,len(blocks)))
                    num+=sums[indices].sum(1);den+=sizes[indices].sum(1);observed.extend(v for b in blocks for v in b)
                result[f'family_{j}_{mode}_minus_B3']=dict(paired_cases=len(observed),success_rate_difference=float(np.mean(observed))if observed else None,paired_block_bootstrap95=np.quantile(num/np.maximum(den,1.),[.025,.975]).tolist()if observed else None,replicates=5000,scope='Matched controller/state differences; physical states resampled within maps, both nominal outcomes stay together. Conditional finite-pool comparison, not a new-map claim.')
    dump(HERE/'CONFIDENCE_INTERVALS.json',result)
