"""Outcome maps, honest finite-grid topology and predeclared adaptive job gates."""
import argparse
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import beta
from diagnostics.success_basin_geometry.setup import HERE,ROOT,write,sha

EVENTS=['success','deadlock','timeout','collision']
def read(name):return json.loads((HERE/name).read_text())
def lower(k,n,alpha=.05):return 0. if k==0 else float(beta.ppf(alpha,k,n-k+1))
def bounds(k,n):return [0. if k==0 else float(beta.ppf(.025,k,n-k+1)),1. if k==n else float(beta.ppf(.975,k+1,n-k))]
def records(stages=None):
    result=read('cache_index.json')['records'].copy()
    for path in sorted((HERE/'raw').glob('*/manifest.json')):
        if 'benchmark' in path.parent.name:continue
        if stages is not None and path.parent.name not in stages:continue
        result.extend(json.loads(path.read_text())['records'])
    return result

def cells(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['state_id'],tuple(r['phi']))].append(r)
    result=[]
    for (sid,phi),rs in sorted(groups.items()):
        unique={(r['seed']):r for r in rs};assert len(unique)==len(rs),'Duplicated state/phi/seed'
        n=len(rs);c=Counter(r['outcome'] for r in rs);invalid=c[None];complete=n-invalid
        prob={e:c[e]/complete if complete else None for e in EVENTS};sl=lower(c['success'],n)
        top=max(EVENTS,key=lambda e:c[e]);dominant=top.upper() if top!='collision' and c[top]/n>=.75 else 'MIXED'
        result.append({'state_id':sid,'phi':phi,'n':n,'completed':complete,'unresolved_solver_failures':invalid,
            'counts':{e:c[e] for e in EVENTS},'Q':prob,'Q_is_conditional_on_completion':bool(invalid),
            'outcome_identification_bounds':{e:[c[e]/n,(c[e]+invalid)/n] for e in EVENTS},
            'Q_95_intervals':{e:[bounds(c[e],n)[0],bounds(c[e]+invalid,n)[1]] for e in EVENTS},'success_lower95':sl,
            'success_dominant':sl>=.75,'sensitivity_success':sl>=.70,'dominant_outcome':dominant,
            'mean_success_seconds':float(np.mean([r['steps']*.05 for r in rs if r['outcome']=='success'])) if c['success'] else None,
            'phi_norm':float(np.linalg.norm(phi)),'records':[r['file'] for r in rs]})
    return result

def components(cs,threshold=.75,eight=False,step=.25):
    # Only the actual frozen regular 2D main grid; supplemental points are not silently snapped.
    coords={tuple(c['phi']):c for c in cs if c['phi'][1]==0 and c['success_lower95']>=threshold}
    remaining=set(coords);comp=[]
    directions=[(-step,0),(step,0),(0,-step),(0,step)]
    if eight:directions+=[(-step,-step),(-step,step),(step,-step),(step,step)]
    while remaining:
        start=min(remaining);remaining.remove(start);stack=[start];group=[]
        while stack:
            p=stack.pop();group.append(p)
            for dx,dy in directions:
                q=(round(p[0]+dx,8),0.,round(p[2]+dy,8))
                if q in remaining:remaining.remove(q);stack.append(q)
        comp.append(sorted(group))
    return comp

def job(sid,phi,seed,label):return {'state_id':sid,'phi':list(phi),'seed':seed,'cell_id':label}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-next',action='store_true');args=ap.parse_args()
    p=read('protocol.json'); rows=records(); cs=cells(rows)
    mainrows=[r for r in rows if r.get('origin')=='compatible_fresh_cached' or 'raw/main_' in r['file'] or 'raw/resume_' in r['file']]
    coarse=cells(mainrows)
    write('outcome_basin_map.json',{'schema':'SBGA_outcome_map_v1','protocol_sha256':sha(HERE/'protocol.json'),
        'phi_order':p['phi_order'],'outcomes':EVENTS,'cells':cs,'coarse_cells':coarse,'sample_count':len(rows),
        'statistical_note':'Descriptive per-cell binomial intervals; topology selected from map requires independent validation, not simultaneous coverage.'})
    conn={}
    for sid in p['primary_states']:
        maincs=[c for c in coarse if c['state_id']==sid and c['phi'][1]==0 and c['phi'][0] in p['main_grid']['goal'] and c['phi'][2] in p['main_grid']['relative']]
        co=components(maincs);sensitivity=components(maincs,.70);eight=components(maincs,eight=True)
        label='NO_SUCCESS_BASIN_FOUND' if not co else 'SUCCESS_BASIN_TOO_SMALL_TO_CLASSIFY' if max(map(len,co))<2 else 'EMPIRICALLY_SINGLE_SUCCESS_BASIN' if len(co)==1 else 'EMPIRICALLY_MULTIPLE_SEPARATED_SUCCESS_BASINS'
        conn[sid]={'components':co,'component_sizes':[len(g) for g in co],'threshold_sensitivity_components':sensitivity,
            'eight_neighbor_components':eight,'preliminary_label':label,'independently_validated':False,
            'finite_grid_only':True}
    write('connected_components.json',{'threshold':.75,'lower_confidence':'one-sided 95% exact','main_grid_step':.25,'states':conn})
    success=[c for c in cs if c['counts'].get('success',0)>0]
    print(json.dumps({'samples':len(rows),'cells':len(cs),'success_episodes':sum(r['outcome']=='success' for r in rows),
        'success_cells':len(success),'success_states':sorted({c['state_id'] for c in success}),
        'largest_success_estimates':sorted([{'state':c['state_id'],'phi':c['phi'],'Q_S':c['Q']['success'],'lower':c['success_lower95']} for c in success],key=lambda x:-x['Q_S'])[:8]}),flush=True)
    if not args.plan_next:return
    if success:
        validation=[];refine=[];seen=set()
        for sid in sorted({c['state_id'] for c in success}):
            cand=[c for c in success if c['state_id']==sid]
            # Avoid missing a component by taking its min-norm representative first.
            reps=[]
            for component in conn.get(sid,{}).get('components',[]):
                reps.append(min([c for c in cand if tuple(c['phi']) in component],key=lambda c:c['phi_norm']))
            reps += sorted(cand,key=lambda c:(-c['counts'].get('success',0)/c['n'],c['phi_norm']))[:2]
            basin=[c for c in cand if c['success_lower95']>=.75]
            reps += sorted(basin or cand,key=lambda c:(c['phi_norm'],-c['counts'].get('success',0)/c['n']))[:2]
            unique={tuple(c['phi']):c for c in reps};reps=list(unique.values())[:6]
            for c in reps:
                validation.extend(job(sid,c['phi'],seed,'validate') for seed in p['validation']['seeds'])
            for c in reps[:4]:
                for dx,dy in itertools.product([-.125,0,.125],repeat=2):
                    phi=(c['phi'][0]+dx,c['phi'][1],c['phi'][2]+dy);key=(sid,phi)
                    if key in seen:continue
                    seen.add(key);refine.extend(job(sid,phi,seed,'refine') for seed in p['refinement']['seeds'])
        write('validation_jobs.json',validation);write('refine_jobs.json',refine)
        center=[c for c in success if c['state_id']=='D4_pair227']
        centers=[c for c in center if c['success_lower95']>=.75] or center
        locations={(0.,0.,0.)}
        for c in centers:
            for dx,dy in [(0,0),(-.25,0),(.25,0),(0,-.25),(0,.25)]:
                phi=(c['phi'][0]+dx,0.,c['phi'][2]+dy)
                if -1<=phi[0]<=1 and -1<=phi[2]<=1:locations.add(phi)
        locations=sorted(locations,key=lambda phi:(sum(v*v for v in phi),phi))[:32]
        temporal=[job(sid,phi,seed,'temporal_success_slice') for sid in ['P227_offset8s','P227_offset2s']
            for phi in locations for seed in p['temporal_success_gate']['seeds']]
        write('temporal_jobs.json',temporal)
        write('next_stage_plan.json',{'branch':'success_found','validation_rollouts':len(validation),'refinement_rollouts':len(refine),'temporal_rollouts':len(temporal),'temporal_phi_points':locations})
    else:
        extra=[]
        for sid in p['primary_states']:
            for g,s,r in itertools.product([-1,0,1],repeat=3):
                if s==0:continue
                extra.extend(job(sid,(g,s,r),seed,'family_3d') for seed in p['no_success_gate']['seeds'])
        temporal=[]
        for sid in ['P227_offset8s','P227_offset2s']:
            for g,r in itertools.product(p['no_success_gate']['temporal_axis'],repeat=2):
                temporal.extend(job(sid,(g,0,r),seed,'temporal') for seed in p['no_success_gate']['seeds'])
        write('family_jobs.json',extra);write('temporal_jobs.json',temporal)
        write('next_stage_plan.json',{'branch':'no_success_found','family_3d_rollouts':len(extra),'temporal_rollouts':len(temporal),
            'purpose':'Distinguish inadequate 2D slice, omitted safe-feedback gain, and intervention timing without redesigning the family.'})

if __name__=='__main__':main()
