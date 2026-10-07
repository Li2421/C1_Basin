"""Outcome-free time-to-possible-contact lower bounds for source inputs.

These are speed-bound geometric statements, NOT claims that safety projection
cannot activate earlier, or that reaching contact is required for prediction.
"""
import numpy as np
from .response_horizon import OUT,SOURCE,NAMES,read,write,csvwrite

def main():
    OUT.mkdir(exist_ok=True)
    states=read(SOURCE/'physical.json');rows=[]
    for state in states:
        p=state['physical'];x=np.asarray(p['positions']);r=np.asarray(p['radius']);v=p['max_speed']
        pairs=[(np.linalg.norm(x[i]-x[j])-r[i]-r[j]-p['agent_margin'])/(2*v) for i in range(len(x)) for j in range(i)]
        inner=[o for o in p['obstacles'] if o['interior_sign']>0]
        clearance=[(np.linalg.norm(x-np.asarray(o['a']),axis=1)-o['radius']-r-p['wall_margin']).min()/v for o in inner]
        rows.append(dict(state_uid=state['state_uid'],earliest_agent_contact_seconds=float(min(pairs)),
            earliest_inner_boundary_contact_seconds=float(min(clearance)),H20_seconds=20*p['dt'],H80_seconds=80*p['dt']))
    csvwrite(OUT/'source_contact_time_lower_bounds.csv',rows)
    summary={}
    for key in ('earliest_agent_contact_seconds','earliest_inner_boundary_contact_seconds'):
        a=np.array([r[key] for r in rows]);summary[key]=dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),
            fraction_lower_bound_exceeds_H20=float(np.mean(a>1)),fraction_lower_bound_exceeds_H80=float(np.mean(a>4)))
    write(OUT/'response_window_physics.json',dict(states=len(states),bounds=summary,
        caveat='Physical contact lower bounds only. Safety projection can activate before contact. Does not prove exact natural-controller aliasing or that longer windows necessarily improve prediction.',
        interpretation='H20 observes initial approach before any possible agent contact/inner-boundary contact for these source states; H80 can reach the interaction regime.',
        outcomes_used=False,new_task_rollouts=0))
    print(summary)

if __name__=='__main__':main()
