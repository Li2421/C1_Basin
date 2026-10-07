"""Predeclared fresh extensions, refined minima and temporal representatives."""
import json
from diagnostics.success_basin_geometry.setup import HERE,write
from diagnostics.success_basin_geometry.analyze import cells,job

def raw(stage):return json.loads((HERE/'raw'/stage/'manifest.json').read_text())['records']
def main():
    p=json.loads((HERE/'protocol.json').read_text());vc=cells(raw('validation'));rc=cells(raw('refine'));tc=cells(raw('temporal'))
    extension=[]
    for c in vc:
        if c['success_lower95']<.75 and c['Q_95_intervals']['success'][1]>=.75:
            extension.extend(job(c['state_id'],c['phi'],s,'extend_validation') for s in p['validation']['extension_seeds'])
    fresh=[];selection=[]
    for sid in p['primary_states']:
        existing={tuple(c['phi']) for c in vc if c['state_id']==sid}
        cand=[c for c in rc if c['state_id']==sid and c['counts'].get('success',0)/c['n']>=.75 and tuple(c['phi']) not in existing]
        cand=sorted(cand,key=lambda c:(c['phi_norm'],-c['counts'].get('success',0)/c['n']))[:2]
        for c in cand:
            fresh.extend(job(sid,c['phi'],s,'refined_minimum_validation') for s in p['validation']['seeds']);selection.append(c)
    temporal=[]
    for sid in ['P227_offset8s','P227_offset2s']:
        cand=[c for c in tc if c['state_id']==sid and c['counts'].get('success',0)/c['n']>=.75]
        if not cand:continue
        minc=min(cand,key=lambda c:(c['phi_norm'],-c['counts'].get('success',0)/c['n']))
        maxc=min(cand,key=lambda c:(-c['counts'].get('success',0)/c['n'],c['phi_norm']))
        reps={tuple(c['phi']):c for c in [minc,maxc]}
        for c in reps.values():temporal.extend(job(sid,c['phi'],s,'temporal_validation') for s in p['validation']['seeds'])
    write('extension_jobs.json',extension);write('validation_refined_jobs.json',fresh);write('validation_temporal_jobs.json',temporal)
    sizes={name:{'rollouts':len(jobs),'max_physical_steps':sum(850-next(c['start_step'] for c in p['state_catalog'] if c['state_id']==j['state_id']) for j in jobs)} for name,jobs in [('extension',extension),('validation_refined',fresh),('validation_temporal',temporal)]}
    write('followup_plan.json',{'budgets':sizes,'refined_candidate_selection':selection,'gate':'Uncertainty at threshold, smaller successful local candidates, same-source temporal validation. No parameters tuned on these fresh outcomes.'})
    print(sizes)

if __name__=='__main__':main()
