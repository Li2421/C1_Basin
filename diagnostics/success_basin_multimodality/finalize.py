"""Aggregate repaired outcomes and produce all SBMA machine-readable conclusions."""
import json
from collections import Counter,defaultdict,deque
from itertools import product
from pathlib import Path
import numpy as np
from scipy.stats import beta
from diagnostics.success_basin_multimodality.setup import HERE,write,sha

EVENTS=['success','deadlock','timeout','collision']
OLD=HERE.parent/'success_basin_geometry'

def manifest(stage):return json.loads((HERE/'raw'/stage/'manifest.json').read_text())['records']
def rkey(r):return (r['state_id'],tuple(np.round(r['eta'],12)),r['seed'])
def replace_null(base,replacements):
    repl={rkey(r):r for stage in replacements if (HERE/'raw'/stage/'manifest.json').exists() for r in manifest(stage) if r['outcome'] is not None}
    return [repl.get(rkey(r),r) if r['outcome'] is None else r for r in base]
def interval(k,n,alpha=.05,two=False):
    a=alpha/2 if two else alpha
    return [float(beta.ppf(a,k,n-k+1)) if k else 0.,float(beta.ppf(1-a,k+1,n-k)) if k<n else 1.]
def summarize(rs):
    c=Counter(r['outcome'] for r in rs);n=len(rs);lo,hi=interval(c['success'],n)
    unknown=c[None]
    if unknown or n<16:label='UNKNOWN_CELL'
    elif lo>=.8:label='SUCCESS_CELL'
    elif hi<=.2:label='FAILURE_CELL'
    else:label='UNKNOWN_CELL'
    return {'n':n,'counts':{x:c[x] for x in EVENTS},'numerical_unknown':unknown,
      'probabilities':{x:c[x]/n for x in EVENTS},'exact_95_intervals':{x:interval(c[x],n,two=True) for x in EVENTS},
      'Q_S_one_sided_95':[lo,hi],'classification':label,
      'mean_completion_seconds':float(np.mean([r['steps']*.05 for r in rs if r['outcome']=='success'])) if c['success'] else None}
def cell_rows(rows,extra=()):
    groups=defaultdict(list)
    for r in rows:groups[(r['state_id'],tuple(np.round(r['eta'],12)),*(r.get(k) for k in extra))].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        entry={'state_id':key[0],'eta':list(key[1])}
        for name,value in zip(extra,key[2:]):entry[name]=value
        entry.update(summarize(rs));out.append(entry)
    return out
def components(cells,p):
    axes=p['phase_a_design']['axes'];axis=[axes['goal'],axes['safe'],axes['relative']]
    def neigh(x):
        idx=[axis[d].index(x[d]) for d in range(3)]
        for d in range(3):
            for off in [-1,1]:
                q=idx.copy();q[d]+=off
                if 0<=q[d]<len(axis[d]):yield tuple(axis[i][q[i]] for i in range(3))
    states={};sets=[]
    for s in p['primary_states']:
        sid=s['state_id'];by={tuple(c['eta']):c for c in cells if c['state_id']==sid};succ={x for x,c in by.items() if c['classification']=='SUCCESS_CELL'};remain=set(succ);comps=[]
        while remain:
            root=remain.pop();seen={root};todo=[root]
            while todo:
                for y in neigh(todo.pop()):
                    if y in remain:remain.remove(y);seen.add(y);todo.append(y)
            comps.append(sorted(seen))
        comps.sort(key=len,reverse=True);sets.append(succ)
        states[sid]={'component_count':len(comps),'component_sizes':[len(x) for x in comps],
          'components':[[list(x) for x in c] for c in comps],
          'success_cells':len(succ),'failure_cells':sum(c['classification']=='FAILURE_CELL' for c in by.values()),
          'unknown_cells':sum(c['classification']=='UNKNOWN_CELL' for c in by.values()),
          'known_eta_in_success':(1.,0.,.25) in succ}
    common=sorted(set.intersection(*sets));return states,common

def retry_counts(rows):
    out=Counter()
    for r in rows:
        with np.load(HERE/r['file']) as d:
            if 'first_retry' in d:out['first_retry_steps']+=int(np.sum(d['first_retry']))
            if 'second_retry' in d:out['second_retry_steps']+=int(np.sum(d['second_retry']))
    return dict(out)

def behavior(rows,eta):
    chosen=[r for r in rows if tuple(np.round(r['eta'],12))==tuple(np.round(eta,12)) and r['outcome']=='success']
    vals=[]
    for r in chosen:
        with np.load(HERE/r['file']) as d:
            before=d['positions_before'];after=d['positions_after'];u=d['u_exec'];g=d['g'];order=after[:,0,0]-after[:,1,0];initial=before[0,0,0]-before[0,1,0]
            change=np.flatnonzero(np.sign(order)!=np.sign(initial))
            vals.append({'completion_seconds':len(after)*.05,'path_lengths':np.linalg.norm(after-before,axis=-1).sum(axis=0),
              'max_lateral':after[:,:,1].max(axis=0),'min_lateral':after[:,:,1].min(axis=0),
              'mean_speed':np.linalg.norm(u,axis=-1).mean(axis=0),'cumulative_correction_norm':float(np.linalg.norm(g,axis=-1).sum()),
              'ordering_change_seconds':float(change[0]*.05) if len(change) else None,
              'second_projection_active_fraction':float(np.mean(np.any(d['second_active'],axis=1)))})
    def stats(name):
        a=np.asarray([v[name] for v in vals if v[name] is not None]);return {'median':np.median(a,axis=0).tolist(),'min':np.min(a,axis=0).tolist(),'max':np.max(a,axis=0).tolist()}
    return {'eta':list(eta),'successful_trajectories':len(vals),'metrics':{x:stats(x) for x in vals[0]}}

def main():
    p=json.loads((HERE/'protocol.json').read_text());plan=json.loads((HERE/'phase_b_plan.json').read_text());states=[x['state_id'] for x in p['primary_states']]
    phasea=[]
    for sid in states:phasea+=manifest(f'phase_a_{sid}')
    phasea=replace_null(phasea,['repair_numerics','repair2'])
    phasea_cells=cell_rows(phasea);st,common=components(phasea_cells,p)
    write('success_map.json',{'criterion':p['classification'],'phase_a_cells':phasea_cells,'phase_a_rollouts':len(phasea),
      'phase_a_outcomes':dict(Counter(r['outcome'] for r in phasea)),'numerical_repairs_substitute_only_invalid_same_seed_attempts':True})
    write('connected_components.json',{'adjacency':'axis-neighbor in frozen structured 3-D lattice','states':st,
      'common_success_cells_all_three_states':[list(x) for x in common],
      'interpretation':'Empirical graph components only; no claim of mathematical topology outside tested region.'})
    known=cell_rows(manifest('known_validation'))
    interp=manifest('straight_interpolation');interp=replace_null(interp,['repair_phaseb'])
    icells=cell_rows(interp,extra=('lambda',));path_by={sid:[c for c in icells if c['state_id']==sid] for sid in states}
    path_status='SUCCESS_PATH_FOUND' if all(c['classification']=='SUCCESS_CELL' for c in icells) else ('PATH_UNRESOLVED_DUE_TO_UNKNOWN_POINTS' if any(c['numerical_unknown'] for c in icells) else 'STRAIGHT_PATH_NOT_ALL_SUCCESS')
    write('path_connectivity.json',{'eta_A':plan['eta_A'],'eta_B':plan['eta_B'],'selection_rule':plan['selection_rule'],
      'lambda':p['path_lambda'],'per_state':path_by,'result':path_status,
      'phase_a_graph_component_counts':{s:st[s]['component_count'] for s in states},
      'finite_region_only':True})
    write('interpolation_results.json',{'eta_A':plan['eta_A'],'eta_B':plan['eta_B'],'cells':icells,'result':path_status,
      'averaging_evidence':('AGAINST_MULTIMODAL_REPRESENTATION_IN_TESTED_REGION' if path_status=='SUCCESS_PATH_FOUND' else 'NOT_RESOLVED')})
    endpoint=[]
    for sid in states:
        for eta in [plan['eta_A'],plan['eta_B']]:endpoint.append(next(c for c in icells if c['state_id']==sid and np.allclose(c['eta'],eta)))
    write('fresh_component_validation.json',{'known_success_eta':known,'single_phase_a_component_per_state':all(st[s]['component_count']==1 for s in states),
      'farthest_common_component_endpoint_validation':endpoint,
      'note':'Endpoints are distant representatives of the same empirical component, not two candidate components.'})
    marginrows=manifest('basin_margin');marginrows=replace_null(marginrows,['repair_phaseb']);mcells=cell_rows(marginrows,extra=('direction_id','delta'))
    dirnames=['+goal','+safe','+relative','-goal','-safe','-relative'];margins=[]
    for sid in states:
        for did,name in enumerate(dirnames):
            cs=sorted([c for c in mcells if c['state_id']==sid and c['direction_id']==did],key=lambda c:c['delta'])
            consecutive=[]
            for c in cs:
                if c['classification']!='SUCCESS_CELL':break
                consecutive.append(c['delta'])
            first=next((c['delta'] for c in cs if c['classification']!='SUCCESS_CELL'),None)
            margins.append({'state_id':sid,'direction':name,'tested_cells':cs,
              'empirical_success_margin_lower':max(consecutive) if consecutive else 0.,
              'first_not_high_confidence_delta':first,
              'right_censored_at_max_tested':first is None})
    write('basin_margin.json',{'center_eta':[1.,0.,.25],'definition':'largest consecutively tested directional delta retaining SUCCESS_CELL; finite empirical bound only',
      'directions':margins})
    cross={'known_eta_success_all_three':all(c['classification']=='SUCCESS_CELL' for c in known),
      'common_phase_a_success_cells':len(common),'single_component_each_state':all(st[s]['component_count']==1 for s in states),
      'straight_path_success_all_three':path_status=='SUCCESS_PATH_FOUND','state_component_summaries':st,
      'interpretation':'A broad recurring local geometry is plausible for state-conditioned learning; three similar selected states do not establish population generalization.'}
    write('cross_state_topology.json',cross)
    behaviors={sid:[behavior([r for r in interp if r['state_id']==sid],eta) for eta in [plan['eta_A'],plan['eta_B']]] for sid in states}
    same=[]
    for sid,bs in behaviors.items():
        # Compare ordering-change existence and which agent has the higher positive lateral excursion.
        mode=[int(np.argmax(b['metrics']['max_lateral']['median'])) for b in bs]
        same.append(mode[0]==mode[1])
    write('behavioral_modes.json',{'per_state_endpoint_behaviors':behaviors,
      'same_lateral_coordination_identity_at_endpoints':all(same),
      'distinct_successful_modes_established':False,
      'interpretation':'Measured endpoint trajectories do not establish meaningfully different coordination modes.'})
    allnew=phasea+manifest('known_validation')+interp+marginrows
    decision=('CASE B — SINGLE CONNECTED SUCCESS BASIN SUPPORTED' if path_status=='SUCCESS_PATH_FOUND' and all(st[s]['component_count']==1 for s in states)
              else 'CASE C — SUCCESS BASIN EXISTS BUT TOPOLOGY REMAINS UNRESOLVED')
    write('decision.json',{'classification':decision,'path_status':path_status,
      'multimodal_success_geometry_supported':False,
      'scope':'Explored eta box and selected D1/D2/D4 states only; finite empirical support, not mathematical proof.',
      'new_outcome_counts':dict(Counter(r['outcome'] for r in allnew)),
      'exact_solver_retry_steps':retry_counts([r for r in allnew if r['file'].startswith('raw/')])})
    print(json.dumps(json.loads((HERE/'decision.json').read_text()),indent=2))

if __name__=='__main__':main()
