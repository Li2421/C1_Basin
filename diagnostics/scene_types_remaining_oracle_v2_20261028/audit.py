"""Reuse completed v1 evidence before proposing any additional rollout."""
import collections,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'diagnostics/scene_types_basis_dimension_v1'
def read(p):return json.loads(Path(p).read_text())
def dump(name,r):(HERE/name).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def key(r):return(r['type_id'],r['controller'],r['state_index'])
def canonical(r):return json.dumps(r,sort_keys=True,separators=(',',':'))
def main():
    frozen=read(OLD/'SCENES_FROZEN.json');labels=read(OLD/'HOLDOUT_LABELS.json');ranges=read(OLD/'GOAL_RANGE_LABELS.json');families=read(OLD/'FAMILIES_FROZEN.json')['families'];cache={}
    def row(uid):
        if uid not in cache:cache[uid]=read(OLD/'raw'/f'{uid}.json')
        return cache[uid]
    index=next(str(j)for j,f in enumerate(families)if set(f['columns'])=={'G','R','T1','C2'})
    grouped={}
    for x in labels[index]:grouped.setdefault(key(x),[]).append(row(x['uid']))
    remaining={k for k,rr in grouped.items()if not any(r['success']for r in rr)};assert len(remaining)==43
    linear={};by_method={}
    for name,ll in {**labels,**{'range_'+k:v for k,v in ranges.items()}}.items():
        rr={x['uid']:row(x['uid'])for x in ll};by_method[name]=rr
        if name not in('right_goal_reference','centre_rotation_reference'):
            for uid,r in rr.items():linear.setdefault(key(r),{})[uid]=r
    unresolved={k for k in remaining if not any(r['success']for r in linear[k].values())}
    names={e['type_id']:e['title']for e in frozen['entries']};entries=[]
    for k in sorted(remaining):
        rr=linear[k];success=[r for r in rr.values()if r['success']]
        entries.append(dict(type_id=k[0],title=names[k[0]],controller=k[1],state_index=k[2],unique_linear_rollouts=len(rr),unique_parameter_specs=len({canonical(r['spec'])for r in rr.values()}),rescued_by_existing_pool=bool(success),successful_uids=[r['rollout_uid']for r in success],outcomes=dict(collections.Counter(r['outcome']for r in rr.values()))))
    # Read only search/calibration states0..3 of the exact frozen scenario for
    # potential transfer points. Do not use heldout outcomes to fit parameters.
    scenario_ids={row(labels[index][j]['uid'])['scenario_uid']for j in range(len(labels[index]))}
    ctl_names={k[1]for k in remaining};dev={}
    for p in(OLD/'raw').glob('*.json'):
        r=read(p)
        if r['scenario_uid']not in scenario_ids or r['state_index']not in range(4)or r['solo']is not None or r['controller']not in ctl_names or not r['success']or r.get('safety_layer')!='distributed':continue
        if r['purpose']not in('search','calibration_safety','matched_midpoint_control'):continue
        sp=r['spec'];dimension=sum(v!=0. for v in sp['eta'])+sum(v!=0. for v in sp.get('coefficients',{}).values())
        if dimension>5:continue
        dev.setdefault((r['type_id'],r['controller']),{}).setdefault(canonical(sp),[]).append(r)
    proposals=[]
    for k in sorted(unresolved):
        already={canonical(r['spec'])for r in linear[k].values()};points=dev.get(k[:2],{})
        ranked=sorted(points,key=lambda s:(-len({r['state_index']for r in points[s]}),len(points[s][0]['spec'].get('coefficients',{})),s))
        missing=[s for s in ranked if s not in already]
        proposals.append(dict(type_id=k[0],title=names[k[0]],controller=k[1],state_index=k[2],untried_successful_search_points=[dict(spec=points[s][0]['spec'],source_uids=[r['rollout_uid']for r in points[s]],source_search_states=sorted({r['state_index']for r in points[s]}))for s in missing]))
    paper=read(OLD/'PAPER_RESULTS.json');reference=read(OLD/'REFERENCE_RESULTS.json')
    out=dict(original_remaining=43,rescued_by_existing_other_linear_pools=len(remaining-unresolved),still_unresolved=len(unresolved),cases=entries,still_unresolved_keys=[list(k)for k in sorted(unresolved)],per_type=[dict(type_id=t,title=names[t],B4_failures=sum(k[0]==t for k in remaining),rescued_existing=sum(k[0]==t for k in remaining-unresolved),unresolved=sum(k[0]==t for k in unresolved))for t in sorted({k[0]for k in remaining})],existing_unique_candidates_per_case=sorted({e['unique_linear_rollouts']for e in entries}),untried_development_success_points=sum(len(p['untried_successful_search_points'])for p in proposals),proposals=proposals,paper_comparison=paper,reference_comparison=reference,scope='Read-only reanalysis. The43 are failures of the frozen four-column16-point pool, not of every tested family. Search points transfer only from originalstates0..3; any new test is posthoc rescue screening, never appended to the original77/120 claim.')
    dump('HISTORY_AUDIT.json',out)
    print(json.dumps({k:out[k]for k in('original_remaining','rescued_by_existing_other_linear_pools','still_unresolved','existing_unique_candidates_per_case','untried_development_success_points','per_type')},ensure_ascii=False,indent=2))
    return out
if __name__=='__main__':main()
