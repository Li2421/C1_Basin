"""Uniform finite dictionary screen; staged plans are serialized before execution."""
import argparse,itertools,json
from collections import defaultdict
import numpy as np
from scipy.stats import qmc
from runner import OUT,dump,execute,NEW,ALL,digest
from discovery_design import freeze,case,singles,key,merit
from shared_control.basis_dictionary import SIGNED


def rows():return list(map(json.loads,(OUT/'candidates.jsonl').open()))if(OUT/'candidates.jsonl').exists()else[]
def stage_rows(stage):
    ids={c['rollout_uid']for c in json.load(open(OUT/f'{stage}_planned.json'))['exact_configs']}
    return [r for r in rows()if r['rollout_uid']in ids]

def pool_plan(stage,cases,workers):
    mark=OUT/f'{stage}_execution.json'
    if mark.exists()and json.load(mark.open()).get('complete'):return stage_rows(stage)
    return execute(cases,stage,workers)

def gain(name,u):return (.5+.5*abs(2*u-1))*(1 if u>=.5 else -1)if name in SIGNED else .25+.75*u

def first(stage,workers):
    p=freeze();todo=[]
    if stage=='smoke':
        assert json.load(open(OUT/'checks.json'))['passed']
        for c in [p['cases'][0],p['cases'][-1]]:
            todo.append(case(c,dict(eta=c['baseline_eta'],coefficients={n:.25 for n in NEW})))
    elif stage=='single':
        for c in p['cases']:
            todo.append(case(c,dict(eta=c['baseline_eta'],coefficients={})))
            todo.extend(case(c,sp)for sp in singles(c))
    elif stage=='pairs':
        points=qmc.Sobol(2,scramble=True,seed=2026102801).random_base2(8)
        for pi,pair in enumerate(itertools.combinations(NEW,2)):
            for j in range(2):
                co={n:gain(n,u)for n,u in zip(pair,points[2*pi+j])}
                for c in p['pair_pilot']:
                    todo.append(case(c,dict(eta=c['baseline_eta']if j==0 else[.875,0.,.375],coefficients=co)))
    else:raise ValueError(stage)
    rr=pool_plan(stage,todo,workers)
    if stage=='smoke':assert not any(r['numerical_failure']for r in rr),[(r['error'])for r in rr]
    return rr


def expansion(workers):
    mark=OUT/'expand_execution.json'
    if mark.exists()and json.load(mark.open()).get('complete'):return stage_rows('expand')
    p=freeze();pair_rows=stage_rows('pairs');group=defaultdict(list)
    for r in pair_rows:group[tuple(r['spec']['coefficients'])].append(r)
    order=sorted(group,key=lambda names:(-len({key(r)for r in group[names]if r['success']}),
        np.mean([r['collision']for r in group[names]]),np.mean([sum(r['final_goal_distances'])for r in group[names]]),names))
    selected=order[:12]
    # Equal final budget for each selected pair, automatic ranking only.
    design=dict(selected_pairs=selected,ranking=[dict(names=k,success_cases=len({key(r)for r in group[k]if r['success']}))for k in order],
        sparse_seed=2026102802,selection='All105 pairs received2 configs on6 stratified cases; expand top12, also test sparse3..5 and full15 independent of ranking.')
    dump(OUT/'EXPANSION_FROZEN.json',design);todo=[]
    points=qmc.Sobol(5,scramble=True,seed=2026102802).random_base2(7)
    for k,pair in enumerate(selected):
        for j in range(4):
            co={n:gain(n,u)for n,u in zip(pair,points[4*k+j])}
            for c in p['cases']:todo.append(case(c,dict(eta=c['baseline_eta']if j%2==0 else [.875,0.,.375],coefficients=co)))
    rng=np.random.default_rng(2026102803)
    for j in range(24):
        names=sorted(rng.choice(NEW,size=3+j%3,replace=False));co={n:gain(n,u)for n,u in zip(names,rng.random(len(names)))}
        for c in p['cases']:todo.append(case(c,dict(eta=c['baseline_eta'],coefficients=co)))
    for j in range(4):
        co={n:gain(n,u)for n,u in zip(NEW,rng.random(len(NEW)))}
        for c in p['cases']:todo.append(case(c,dict(eta=c['baseline_eta'],coefficients=co)))
    return pool_plan('expand',todo,workers)


def deletions(workers):
    mark=OUT/'delete_execution.json'
    if mark.exists()and json.load(mark.open()).get('complete'):return stage_rows('delete')
    p=freeze();lookup={(c['scene'],c['controller'],c['state_index']):c for c in p['cases']};rr=[r for r in rows()if key(r)in lookup and r['success']and r['seed']==0]
    todo=[];selected=[]
    for k,c in lookup.items():
        choices=sorted([r for r in rr if key(r)==k],key=lambda r:(len(r['spec']['coefficients']),r['truncated_time'],r['rollout_uid']))
        # Two successful witnesses with different supports; no manually named basis.
        seen=set();chosen=[]
        for r in choices:
            support=tuple(sorted(r['spec']['coefficients']))
            if support not in seen:seen.add(support);chosen.append(r)
            if len(chosen)==2:break
        for r in chosen:
            selected.append(r['rollout_uid']);sp=r['spec'];eta=sp['eta']
            for mask in range(7):
                e=[v if mask&(1<<i)else 0. for i,v in enumerate(eta)]
                todo.append(case(c,dict(sp,eta=e)))
            for name in sp['coefficients']:
                co={n:v for n,v in sp['coefficients'].items()if n!=name};todo.append(case(c,dict(sp,coefficients=co)))
    dump(OUT/'DELETION_FROZEN.json',dict(witnesses=selected,criterion='two distinct successful supports per discovery case; delete every G/P/R subset and each new column'))
    return pool_plan('delete',todo,workers)


def minimal_cover(canonical=True):
    p=freeze();index={(c['scene'],c['controller'],c['state_index']):i for i,c in enumerate(p['cases'])}
    rr=[r for r in rows()if r['success']and r['seed']==0 and key(r)in index
        and all(abs(v)<=1 for v in r['spec']['coefficients'].values())
        and(not canonical or(r['spec'].get('t5_seed','tangent')=='tangent'and r['spec'].get('clearance_side',1)==1))]
    coverage=np.zeros(1<<len(ALL),dtype=np.uint32);witnesses=[]
    for r in rr:
        names=[n for n,v in zip(ALL[:3],r['spec']['eta'])if v!=0]+list(r['spec']['coefficients'])
        mask=sum(1<<ALL.index(n)for n in names);coverage[mask]|=1<<index[key(r)]
        witnesses.append(dict(rollout_uid=r['rollout_uid'],mask=mask,case=index[key(r)],names=names))
    for b in range(len(ALL)):
        step=1<<b;block=coverage.reshape(-1,2*step);block[:,step:]|=block[:,:step]
    target=int(coverage[-1]);hits=np.flatnonzero(coverage==target);minimum=min(int(x).bit_count()for x in hits)
    best=[int(x)for x in hits if int(x).bit_count()==minimum]
    families=[dict(mask=m,names=[n for i,n in enumerate(ALL)if m&(1<<i)],covered_cases=target.bit_count())for m in best]
    families.sort(key=lambda f:(sum(n.startswith('T5')for n in f['names']),f['names']))
    x=dict(case_count=len(index),achievable_cases=target.bit_count(),target_bitset=target,min_columns=minimum,minimal_families=families,
        witnesses=witnesses,interpretation='Exact minimum only within observed finite successful witnesses, not continuous-parameter impossibility for smaller families.')
    x['canonical_recipes']=canonical
    dump(OUT/('MINIMAL_FAMILIES.json'if canonical else'RECIPE_UNION_MINIMAL.json'),x);return x


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['initial','smoke','single','pairs','expand','delete','minimal']);ap.add_argument('--workers',type=int,default=None);args=ap.parse_args()
    if args.stage=='initial':
        for stage in ('smoke','single','pairs'):first(stage,args.workers)
    elif args.stage in ('smoke','single','pairs'):first(args.stage,args.workers)
    elif args.stage=='expand':expansion(args.workers)
    elif args.stage=='delete':deletions(args.workers)
    else:print(minimal_cover()['minimal_families'])

if __name__=='__main__':main()
