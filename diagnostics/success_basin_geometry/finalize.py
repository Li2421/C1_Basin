"""SBGA scientific summaries; no trained model or surrogate score."""
import json
import itertools
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from diagnostics.success_basin_geometry.setup import HERE,ROOT,write,sha
from diagnostics.success_basin_geometry.analyze import read,records,cells,components,lower

def load(r):
    path=HERE/r['file'];assert sha(path)==r['sha256']
    with np.load(path) as d:return dict(d)

def stage_records(prefix):
    result=[]
    for p in (HERE/'raw').glob('*/manifest.json'):
        if p.parent.name.startswith(prefix):result.extend(json.loads(p.read_text())['records'])
    return result

def main():
    p=read('protocol.json');rows=records();mapping=cells(rows)
    validrows=stage_records('validation')+stage_records('extension')+stage_records('transfer')
    validation=cells(validrows) if validrows else []
    validated=[c for c in validation if c['success_lower95']>=.75]
    independent={'cells':validation,'validated_success_policies':validated,
        'criterion':'One-sided exact 95% lower success probability >= .75, unresolved executions counted pessimistically.',
        'selection_note':'Fresh validation seeds do not occur in grid/refinement data. No pooling with selection samples.'}
    write('independent_validation.json',independent)
    minimum=[]
    for sid in p['primary_states']:
        candidates=[c for c in validated if c['state_id']==sid]
        best=min(candidates,key=lambda c:c['phi_norm']) if candidates else None
        minimum.append({'state_id':sid,'minimum_observed_validated':best,
            'status':'VALIDATED_OBSERVED_MINIMUM' if best else 'NO_VALIDATED_SUCCESS_BASIN',
            'global_minimum_claim':False,'metric':'Euclidean parameter distance to phi0; not executed action energy.'})
    write('minimum_success_correction.json',{'states':minimum})
    interpolation=stage_records('interpolation')
    if interpolation:
        endpoints=[r for r in stage_records('validation') if r['state_id']=='D1_pair231'
            and tuple(r['phi']) in [(1.,0.,.25),(1.,0.,.75)] and r['seed'] in p['validation']['seeds']]
        interpolation=interpolation+endpoints
    write('interpolation_test.json',{'status':'RUN' if interpolation else 'NOT_APPLICABLE',
        'reason':None if interpolation else 'No two separately validated success components available.',
        'lambda_set':p['interpolation']['lambda'],'cells':cells(interpolation) if interpolation else []})
    # Exact same phi across fixed primary states, all-outcome comparisons.
    grouped=defaultdict(dict)
    for c in mapping:
        if c['state_id'] in p['primary_states']:grouped[tuple(c['phi'])][c['state_id']]=c
    cross=[]
    for phi,states in grouped.items():
        if len(states)==3:cross.append({'phi':phi,'by_state':{sid:{'Q':v['Q'],'lower_Q_S':v['success_lower95'],'n':v['n'],'solver_failures':v['unresolved_solver_failures']} for sid,v in states.items()}})
    write('cross_state_consistency.json',{'comparisons':cross,'fresh_transfer_cells':cells(stage_records('transfer')) if stage_records('transfer') else [],
        'scope':'Different frozen states, not longitudinal samples. Successful policy transfer is assessed without changing its phi.'})
    temporal=[]
    for sid in p['temporal_states']:
        cs=[c for c in mapping if c['state_id']==sid];entry=next(s for s in p['state_catalog'] if s['state_id']==sid)
        temporal.append({'state_id':sid,'source_pair_id':227,'offset_seconds':entry['nominal_offset_seconds'],
            'cells':cs,'max_observed_Q_S':max((c['Q']['success'] for c in cs if c['Q']['success'] is not None),default=None),
            'qualified_cells':sum(c['success_lower95']>=.75 for c in cs)})
    write('temporal_basin_evolution.json',{'same_source_trajectory':True,'states':temporal,
        'interpretation_limit':'Finite temporal slices; discovery time is bracketed only by measured offsets, not an exact reachability boundary.'})
    # Measured trajectories for dominant outcome examples; compare matched RNG only.
    pairs=[];behaviors=[]
    for sid in p['primary_states']:
        sr=[r for r in rows if r['state_id']==sid and r['outcome'] is not None]
        success=[r for r in sr if r['outcome']=='success']
        known={tuple(c['phi']) for c in validated if c['state_id']==sid}
        success.sort(key=lambda r:tuple(r['phi']) not in known)
        for r in success[:3]:
            d=load(r);pos=d['positions_after'];initial=d['positions_before'][0];goals=np.array([[1.09,0],[-1.09,0]])
            order0=initial[0,0]-initial[1,0];orders=pos[:,0,0]-pos[:,1,0]
            crossing=np.flatnonzero((orders>0)!=(order0>0))
            speeds=np.linalg.norm(d['u_exec'],axis=-1)
            behaviors.append({'state_id':sid,'phi':r['phi'],'seed':r['seed'],'file':r['file'],'independently_validated_policy':tuple(r['phi']) in known,
                'completion_seconds':r['steps']*.05,'goal_errors_final':np.linalg.norm(goals-pos[-1],axis=-1).tolist(),
                'max_lateral_position_by_agent':pos[:,:,1].max(axis=0).tolist(),
                'minimum_speed_by_agent':speeds.min(axis=0).tolist(),'mean_speed_by_agent':speeds.mean(axis=0).tolist(),
                'first_ordering_change_seconds':float((crossing[0]+1)*.05) if len(crossing) else None,
                'initial_order':float(order0),'final_order':float(orders[-1])})
        choices={}
        for outcome in ['success','deadlock','timeout']:
            pool=[c for c in mapping if c['state_id']==sid and c['counts'].get(outcome,0)>0]
            if pool:choices[outcome]=min(pool,key=lambda c:(-c['Q'][outcome],c['phi_norm']))['phi']
        for oa,ob in itertools.combinations(choices,2):
            a={r['seed']:r for r in sr if tuple(r['phi'])==tuple(choices[oa]) and r['outcome']==oa}
            b={r['seed']:r for r in sr if tuple(r['phi'])==tuple(choices[ob]) and r['outcome']==ob}
            for seed in sorted(set(a)&set(b))[:4]:
                aa=load(a[seed]);bb=load(b[seed]);n=min(len(aa['u_exec']),len(bb['u_exec']))
                metrics={k:float(np.sqrt(np.mean((aa[k][:n]-bb[k][:n])**2))) for k in ['g','w','u_exec','positions_after']}
                acta=aa.get('active',aa.get('second_active'));actb=bb.get('active',bb.get('second_active'))
                pairs.append({'state_id':sid,'outcomes':[oa,ob],'phi_a':choices[oa],'phi_b':choices[ob],'seed':seed,'overlap_steps':n,
                    'rms_differences':metrics,'active_disagreement':float(np.mean(np.any(acta[:n]!=actb[:n],axis=1))),
                    'collapse':metrics['u_exec']<1e-8 and metrics['positions_after']<1e-8})
    write('projection_consistency.json',{'paired_outcome_examples':pairs,'alias_count':sum(x['collapse'] for x in pairs),
        'policy_persists_through_full_remaining_episode':True,'solver_failures':sum(r['outcome'] is None for r in rows),
        'failure_note':'CBFSolverError is missing outcome, never recoded collision/timeout; frozen solver unchanged.'})
    write('behavioral_interpretation.json',{'success_examples':behaviors,
        'semantic_modes_preprogrammed':False,'mode_distinction_claim':'Requires separate validated success components; examples alone do not establish modes.'})
    # Final classification stays conservative when isolated successes exist without a reliable region.
    allsuccess=sum(r['outcome']=='success' for r in rows)
    conn=read('connected_components.json')
    icells=cells(interpolation) if interpolation else []
    for sid,entry in conn['states'].items():
        qualified=[c for c in validated if c['state_id']==sid]
        entry['independently_validated_policies']=qualified
        entry['independently_validated']=bool(qualified)
        entry['component_min_separations']=[float(min(np.linalg.norm(np.asarray(a)-b) for a in ca for b in cb))
            for ca,cb in itertools.combinations(entry['components'],2)]
        entry['refinement_note']='Eight-seed local probes alone cannot meet the fixed lower-bound .75 criterion; only separate fresh validation certifies points.'
        entry['final_label']='SUCCESS_BASIN_TOO_SMALL_TO_CLASSIFY' if any(c['counts'].get('success',0)>0 for c in mapping if c['state_id']==sid) else 'NO_SUCCESS_BASIN_FOUND'
    if icells:
        conn['interpolation_evidence']={'cells':icells,'known_deadlock_or_timeout_count':sum(c['counts'].get('deadlock',0)+c['counts'].get('timeout',0) for c in icells),
            'solver_failures':sum(c['unresolved_solver_failures'] for c in icells),
            'all_five_lower_bounds_pass':all(c['success_lower95']>=.75 for c in icells),
            'separated_modes_inference':'Solver failures alone cannot establish an intervening low-success outcome basin.'}
    write('connected_components.json',conn)
    if not allsuccess:
        decision='CASE D — NO SUCCESS BASIN IN CURRENT PHI FAMILY';topology='NO_SUCCESS_BASIN_IN_CURRENT_POLICY_FAMILY'
    elif not validated:
        decision='CASE C — SUCCESS EXISTS BUT BASIN TOPOLOGY IS UNCLEAR';topology='SUCCESS_EXISTS_BUT_TOPOLOGY_UNRESOLVED'
    else:
        # No multiple-component conclusion is automatic from a grid alone.
        decision='CASE C — SUCCESS EXISTS BUT BASIN TOPOLOGY IS UNCLEAR';topology='SUCCESS_EXISTS_BUT_TOPOLOGY_UNRESOLVED'
    write('decision.json',{'main':decision,'multimodality_decision':topology,'observed_success_episodes':allsuccess,
        'independently_validated_success_policies':len(validated),'current_family_adequacy':'SUCCESS_REACHABLE' if allsuccess else 'CURRENT_G_PHI_PARAMETERIZATION_INADEQUATE_WITHIN_TESTED_DOMAIN',
        'domain_limit':'No claim of nonexistence outside tested box or across all G_phi families.'})
    print(read('decision.json'))

if __name__=='__main__':main()
