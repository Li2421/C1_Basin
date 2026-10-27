"""Automatic witness-frontier deletion and targeted lower-dimensional search.

Selection reads discovery seed0 only. No heldout result enters this algorithm.
"""
import itertools,json
import numpy as np
from scipy.stats import qmc
from runner import OUT,dump,digest
from discovery_design import freeze,case,key
from screen import rows,pool_plan,minimal_cover
from holdout import support,eligible
from shared_control.basis_dictionary import ALL,SIGNED


def solved(names,rr):return {key(r)for r in rr if r['success']and support(r['spec'])<=set(names)}
def remove(sp,name):
    out=dict(sp,eta=list(sp['eta']),coefficients=dict(sp['coefficients']))
    if name in ALL[:3]:out['eta'][ALL.index(name)]=0.
    else:out['coefficients'].pop(name,None)
    return out

def main():
    p=freeze();lookup={(c['scene'],c['controller'],c['state_index']):c for c in p['cases']}
    for iteration in range(3):
        if (OUT/('compress_search'+str(iteration)+'_execution.json')).exists():continue
        minimum=minimal_cover();before=minimum['min_columns'];families=minimum['minimal_families'];rr=[r for r in rows()if eligible(r)and key(r)in lookup]
        todo=[];sources=[]
        for f in families:
            for k,c in lookup.items():
                good=[r for r in rr if key(r)==k and r['success']and support(r['spec'])<=set(f['names'])]
                if not good:continue
                witness=min(good,key=lambda r:(r['truncated_time'],r['rollout_uid']));sources.append(witness['rollout_uid'])
                for name in sorted(support(witness['spec'])):todo.append(case(c,remove(witness['spec'],name)))
        dump(OUT/f'COMPRESSION_DELETE{iteration}.json',dict(parent_families=families,witnesses=sources,rule='delete each active column from fastest witness in each parent family/case'))
        pool_plan('compress_delete'+str(iteration),todo,None)
        rr=[r for r in rows()if eligible(r)and key(r)in lookup]
        children={tuple(n for n in f['names']if n!=remove_name)for f in families for remove_name in f['names']}
        order=sorted(children,key=lambda names:(-len(solved(names,rr)),names));best=len(solved(order[0],rr));selected=[n for n in order if len(solved(n,rr))>=best-1][:6]
        dump(OUT/f'COMPRESSION_SEARCH{iteration}.json',dict(children=[dict(names=n,covered=len(solved(n,rr)))for n in order],selected=selected,
            selection='Top at most6 children, within1 case of best existing coverage;16 Sobol coefficients only on still-uncovered cases; no test labels.'))
        todo=[]
        for names in selected:
            points=qmc.Sobol(len(names),scramble=True,seed=2026102810+iteration).random_base2(4);covered=solved(names,rr)
            for k,c in lookup.items():
                if k in covered:continue
                for point in points:
                    e=[0.,0.,0.];co={}
                    for n,u in zip(names,point):
                        if n=='G':e[0]=.5+.75*u
                        elif n=='P':e[1]=-.5+u
                        elif n=='R':e[2]=.75*u
                        else:co[n]=2*u-1 if n in SIGNED else u
                    todo.append(case(c,dict(eta=e,coefficients=co)))
        pool_plan('compress_search'+str(iteration),todo,None)
        after=minimal_cover();print('compression',before,'->',after['min_columns'],after['minimal_families'],flush=True)
        if after['min_columns']>=before:break
    minimal_cover(False)

if __name__=='__main__':main()
