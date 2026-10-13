"""Recompute the requested completion margin on frozen H1/H4 Ring samples."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from reuse_analysis import HERE,HARD,H4,TOL,HORIZON,completion_margin,read,write_csv,write_json


def path(job):
    name=f"{job['uid']}_root{job['root_chain']}_eval{job['eval_chain']}_{job['kind']}_e{job['epsilon']}_a{job['axis']}_s{job['sign']}"
    return H4/'h1_gradient_results'/name


def main():
    p=read(H4/'h1_gradient_protocol.json');states={s['uid']:s for s in read(HARD/'confirmation_manifest.json')['states']}
    rows=list(csv.DictReader((HERE/'boundary_validation.csv').open()))
    response=[]
    roots={(r['uid'],r['chain']):r['first_root_fraction'] for r in read(H4/'h1_roots.json')}
    for c in p['centers']:
        uid=c['uid'];scene='ring_exchange';goals=np.asarray(states[uid]['physical']['goals'])
        for chain in ('TT','FF'):
            jobs=[j for j in p['jobs'] if j['uid']==uid and j['root_chain']==c['root_chain'] and j['eval_chain']==chain]
            def load(j):
                f=path(j);r=read(Path(str(f)+'.json'))
                with np.load(Path(str(f)+'.npz')) as z:pos=z['positions'].copy()
                m=completion_margin(pos,goals,TOL[scene],HORIZON[scene])
                return r,pos,m,f
            z=load(next(j for j in jobs if j['kind']=='center'))
            base,pos,margin,f0=z
            for eps in (.0005,.002):
                direct=[];analytic=[];switch=False;sides=[]
                for axis in range(3):
                    jm=next(j for j in jobs if j['kind']=='fd' and j['epsilon']==eps and j['axis']==axis and j['sign']==-1)
                    jp=next(j for j in jobs if j['kind']=='fd' and j['epsilon']==eps and j['axis']==axis and j['sign']==1)
                    rm,pm,mm,_=load(jm);rp,pp,mp,_=load(jp)
                    g=(mp['rho']-mm['rho'])/(2*eps);direct.append(g)
                    t=margin['active_step'];i=margin['active_agent']
                    if t>=len(pm):pm=np.concatenate([pm,np.repeat(pm[-1:],t-len(pm)+1,axis=0)])
                    if t>=len(pp):pp=np.concatenate([pp,np.repeat(pp[-1:],t-len(pp)+1,axis=0)])
                    s=(pp[t,i]-pm[t,i])/(2*eps)
                    v=pos[t,i]-goals[i]
                    analytic.append(float(-v@s/np.linalg.norm(v)))
                    switch |= (mm['active_step'],mm['active_agent'])!=(t,i) or (mp['active_step'],mp['active_agent'])!=(t,i)
                    sides += [(mm['rho'],rm['success'],rm['termination']),(mp['rho'],rp['success'],rp['termination'])]
                direct=np.asarray(direct);analytic=np.asarray(analytic)
                norm=np.linalg.norm(direct);anorm=np.linalg.norm(analytic)
                rel=float(np.linalg.norm(direct-analytic)/max(norm,1e-12))
                row=dict(scene=scene,uid=uid,chain=chain,root_chain=c['root_chain'],epsilon=eps,
                    center_rho=margin['rho'],old_center_rho=base['rho'],center_success=base['success'],
                    active_step=margin['active_step'],active_agent=margin['active_agent'],time_gap=margin['time_gap'],agent_gap=margin['agent_gap'],
                    gradient_direct=direct.tolist(),gradient_chain=analytic.tolist(),gradient_norm=float(norm),
                    gradient_absolute_error=float(np.linalg.norm(direct-analytic)),gradient_relative_error=rel,
                    normal_cosine=float(direct@analytic/max(norm*anorm,1e-12)),active_switch=bool(switch),
                    root_fraction=c['fraction'],center_result=str(f0))
                rows.append(row)
                if chain!=c['root_chain'] and eps==.0005:
                    line=np.asarray(c['line_direction']);slope=float(direct@line)
                    actual=roots[(uid,chain)]-roots[(uid,c['root_chain'])]
                    predicted=-margin['rho']/slope if abs(slope)>1e-12 else None
                    response.append(dict(uid=uid,source=c['root_chain'],target=chain,
                        source_fraction=roots[(uid,c['root_chain'])],target_fraction=roots[(uid,chain)],
                        center_completion_rho=margin['rho'],slope=slope,
                        actual_fraction_shift=actual,predicted_fraction_shift=predicted,
                        absolute_fraction_error=abs(actual-predicted) if predicted is not None else None,
                        actual_eta_distance=abs(actual)*float(np.linalg.norm(line)),
                        predicted_eta_distance=abs(predicted)*float(np.linalg.norm(line)) if predicted is not None else None,
                        sign_agrees=actual*predicted>0 if predicted is not None else None,
                        source_root_completion_rho=margin['rho'],active_switch=bool(switch),
                        note='local source-root linearization, not finite displacement equality'))
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    write_csv('boundary_validation.csv',rows,fields)
    write_csv('boundary_response.csv',response,list(response[0]))
    write_json('boundary_response_summary.json',dict(rows=len(rows),response_cases=len(response),
        mean_abs_fraction_error=float(np.mean([r['absolute_fraction_error'] for r in response])),
        new_rollouts=0))
    print('boundary rows',len(rows),'responses',len(response),flush=True)


if __name__=='__main__':main()
