"""New-map development sensitivity using the already published old option menu.

Only search states0..3. The historical frozen baseline remains a primary
comparison; a separate retuned profile can be frozen before any heldout run.
"""
from dispatch import *
from papers import prepare,make,OLD
import time

def plan(entries):
    prepare();variants=read(OLD/'paper/DEVELOPMENT_FROZEN.json')['variants'];cases=[]
    for e in entries:
        if 'scene'not in e:continue
        for i in range(4):
            for method,options in variants.items():
                for opts in options:cases.append(make(e['scene'],i,method,opts,'paper_development'))
    return cases,variants

def main():
    while not(HERE/'paper_smoke_execution.json').exists():time.sleep(10)
    assert read(HERE/'paper_smoke_execution.json')['complete']
    frozen=read(HERE/'SCENES_FROZEN.json');entries=[e for e in frozen['entries']if e['type_id']!=2]
    cases,variants=plan(entries)
    dump(HERE/'PAPER_PARAM_DEVELOPMENT_PROTOCOL.json',dict(variants=variants,source=str(OLD/'paper/DEVELOPMENT_FROZEN.json'),source_sha256=sha(OLD/'paper/DEVELOPMENT_FROZEN.json'),states=[0,1,2,3],excluded_pending_geometry_type=2,cases=len(cases),criterion='Max success, then fewer collisions, fewer numerical unknowns, smaller whole-cohort truncated mean. No heldout data. Historical frozen profile retained; any retuned profile separately labelled.',necessity='Original parameter menu has old-map evidence, not these narrow-door/corridor geometries. In particular the author IMPC obstacle-map extra clearance can make a physically open corridor unavailable to its planner; inspect the old predeclared physical-scaling variant before making a blanket method claim.',budget='At most1664 development requests over16 valid types; no training; within45000 total hard cap. Existing smoke/compatible exact requests reused.'))
    execute(cases,'paper_development_other15',worker_entry='worker_papers.py')

def freeze_profiles():
    p=HERE/'PAPER_PARAMETERS_FROZEN.json'
    if p.exists():return read(p)
    frozen=read(HERE/'SCENES_FROZEN.json')
    assert (HERE/'SCENE2_RESOLVED').exists()or frozen.get('valid_subset_revision')==2,'Invalid type2 must be resolved or explicitly excluded before freezing profiles'
    cases,variants=plan(frozen['entries']);all_ids={c['rollout_uid']for c in cases};missing=[c for c in cases if not(HERE/'raw'/f"{c['rollout_uid']}.json").exists()]
    if missing:execute(missing,'paper_development_remaining',worker_entry='worker_papers.py')
    historical=read(OLD/'paper/FINAL_FROZEN.json')['selected'];rankings={};selected={}
    for method,options in variants.items():
        rank=[]
        for opts in options:
            rr=[read(HERE/'raw'/f"{c['rollout_uid']}.json")for c in cases if c['controller_payload']['paper_method']==method and c['controller_payload']['paper_options']==opts and(HERE/'raw'/f"{c['rollout_uid']}.json").exists()]
            rank.append(dict(options=opts,n=len(rr),success=sum(r['success']for r in rr),collision=sum(r['collision']for r in rr),unknown=sum(r['numerical_failure']for r in rr),truncated_mean=float(np.mean([r['truncated_time']for r in rr]))if rr else float('inf')))
        rank.sort(key=lambda r:(-r['success'],r['collision'],r['unknown'],r['truncated_mean'],digest(r['options'])));rankings[method]=rank;selected[method]=rank[0]['options']
    out=dict(historical=historical,selected_development=selected,rankings=rankings,case_requests=len(cases),included_types=[e['type_id']for e in frozen['entries']if 'scene'in e],historical_primary_retained=True,criterion=read(HERE/'PAPER_PARAM_DEVELOPMENT_PROTOCOL.json')['criterion']);dump(p,out);return out

if __name__=='__main__':main()
