"""Lock interpolation, minimal-correction extensions and exact cross-state transfers."""
from diagnostics.success_basin_geometry.analyze import read,job
from diagnostics.success_basin_geometry.setup import write

def main():
    p=read('protocol.json');val=read('independent_validation.json')['cells']
    endpoints=[(1.,0.,.25),(1.,0.,.75)]
    for phi in endpoints:
        assert any(c['state_id']=='D1_pair231' and tuple(c['phi'])==phi and c['success_lower95']>=.75 for c in val)
    inter=[]
    for lam in [.25,.5,.75]:
        phi=(1.,0.,.25+.5*lam)
        for seed in p['validation']['seeds']:inter.append({**job('D1_pair231',phi,seed,'interpolation'),'lambda':lam})
    transfer=[job(sid,phi,seed,'cross_state_transfer') for sid,phi in [('D2_pair228',endpoints[0]),('D4_pair227',endpoints[1])] for seed in p['validation']['seeds']]
    ext=[job('D1_pair231',phi,seed,'smaller_correction_extension') for phi in [(.875,0.,.125),(.875,0.,.25)] for seed in p['validation']['extension_seeds']]
    write('interpolation_jobs.json',inter);write('transfer_jobs.json',transfer);write('extension_minimum_jobs.json',ext)
    write('last_stage_plan.json',{'interpolation_endpoints':endpoints,'interpolation_state':'D1_pair231','lambdas':p['interpolation']['lambda'],
        'endpoint_reuse':'Use original validation first32 identical seeds for paired comparison; no endpoint reruns.',
        'new_rollouts':len(inter+transfer+ext),'max_physical_steps':sum(850-next(s['start_step'] for s in p['state_catalog'] if s['state_id']==r['state_id']) for r in inter+transfer+ext),
        'minimum_candidates':[(.875,0.,.125),(.875,0.,.25)],'stop_after':'No additional parameter search or global densification.'})
    print(read('last_stage_plan.json'))

if __name__=='__main__':main()
