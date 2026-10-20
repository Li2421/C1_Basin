"""Arithmetic-only postprocessing of new probes and immutable historical trajectories."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
def read(name):return list(csv.DictReader(open(P/name)))
def arr(x):return np.asarray(json.loads(x),float)
def dump(name,x):(P/name).write_text(json.dumps(x,indent=2)+'\n')
def write(name,rows):
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def enc(x):return json.dumps(x.tolist() if isinstance(x,np.ndarray) else x,separators=(',',':'))
def main():
    a=read('tt_ff_action_validation.csv');s=read('ff_substep_validation.csv')
    for r in a:
        witness=arr(r['raw_delta'])
        # Repeated postprocessing is idempotent.
        if 'witness_delta' in r:witness=arr(r['witness_delta'])
        r['witness_delta']=enc(witness)
        r['witness_raw_both_positive']=bool(witness[1,1]>0 and witness[0,0]-witness[1,0]>0)
        d=arr(r['applied_delta']);r['raw_delta']=enc(d)
        r['raw_delta2y']=d[1,1];r['raw_delta1x_minus_delta2x']=d[0,0]-d[1,0]
        r['raw_both_positive']=bool(d[1,1]>0 and d[0,0]-d[1,0]>0)
        r['executed_both_positive_tol1e8']=bool(float(r['u2y'])>1e-8 and float(r['u1x_minus_u2x'])>1e-8)
        r['literal_u2y_gt0']=float(r['u2y'])>0
        r['literal_u1x_minus_u2x_gt0']=float(r['u1x_minus_u2x'])>0
        target=arr(r['target']);r['projection_input2y']=target[1,1];r['projection_input_gap']=target[0,0]-target[1,0]
        base=next(t for t in a if t['state_id']==r['state_id'] and t['chain']==r['chain'] and t['parameter']=='zero')
        r['yield_effect_vs_zero']=float(r['u2y'])-float(base['u2y'])
        r['pass_effect_vs_zero']=float(r['u1x_minus_u2x'])-float(base['u1x_minus_u2x'])
    pair=[]
    for uid in dict.fromkeys(r['state_id'] for r in s):
        rows={lab:[next(r for r in s if r['state_id']==uid and r['parameter']==lab and r['k']==str(k)) for k in range(10)] for lab in ('zero','witness')}
        direct=np.zeros((2,2));feedback=direct.copy();proj=direct.copy()
        for k,(z,w) in enumerate(zip(rows['zero'],rows['witness'])):
            dh=arr(w['h'])-arr(z['h']);df=arr(w['f'])-arr(z['f']);d=arr(w['delta'])
            actual=(arr(w['h_next'])-arr(w['h']))-(arr(z['h_next'])-arr(z['h']))
            projection=actual-.1*d-.1*df
            direct+=.1*d;feedback+=.1*df;proj+=projection
            w.update(paired_direct_increment=enc(.1*d),paired_flow_feedback_increment=enc(.1*df),
                paired_projection_increment=enc(projection),paired_h_difference=enc(dh),
                simultaneous_pair_wall_active=bool(0 in json.loads(w['active_cbf_rows']) and len(json.loads(w['active_cbf_rows']))>1))
            if k==9:
                dD=arr(w['D9'])-arr(z['D9']);den=np.linalg.norm(dh)
                w['paired_D9_secant_ratio']=float(np.linalg.norm(dD)/den) if den>1e-12 else ''
                w['paired_D9_yield_difference']=dD[1,1]
                w['paired_D9_pass_difference']=dD[0,0]-dD[1,0]
                pair.append(dict(state_id=uid,D9_secant_ratio=float(np.linalg.norm(dD)/den) if den>1e-12 else None,
                    direct_total=direct.tolist(),flow_feedback_total=feedback.tolist(),projection_total=proj.tolist(),
                    final_action_difference=(arr(w['q'])-arr(z['q'])).tolist(),
                    r9_difference=(arr(w['r'])-arr(z['r'])).tolist(),D9_difference=dD.tolist()))
    write('tt_ff_action_validation.csv',a);write('ff_substep_validation.csv',s);dump('paired_ff_mechanisms.json',pair)
    trajectories=json.loads((P/'trajectory_metrics.json').read_text())
    for r in trajectories:
        with np.load(r['npz']) as z:p=z['positions']
        D=json.loads((P/'geometry_data.json').read_text())['D'];y=p[:,1,1]-D
        ys=r['yield_step'];ps=r['pass_step']
        r['min_chiY_during_all_horizontal_overlap']=r['min_chiY_during_overlap']
        # Yield-Hold-Pass begins after reaching the clearance threshold, not on approach.
        r['min_chiY_after_yield_until_pass']=float(y[ys:ps+1].min()) if ys is not None and ps is not None and ys<ps else None
        r['complete_yhp']=bool(r['min_chiY_after_yield_until_pass'] is not None and r['min_chiY_after_yield_until_pass']>0)
        r['duration_hold_until_pass_seconds']=(ps-ys)*.05 if r['complete_yhp'] else None
        r['at_horizontal_order_crossing']=[]
        dx=p[:,0,0]-p[:,1,0]
        for t in np.flatnonzero((dx[:-1]<0)&(dx[1:]>=0)):
            q=p[t]+(-dx[t]/(dx[t+1]-dx[t]))*(p[t+1]-p[t])
            r['at_horizontal_order_crossing'].append(dict(step_interval=int(t),y2=float(q[1,1]),
                lateral_clearance=float(q[1,1]-q[0,1]),chiY=float(q[1,1]-D)))
    dump('trajectory_metrics.json',trajectories)
    # Nonempty collision-free open neighborhood and a feasible local geometry path.
    g=json.loads((P/'geometry_data.json').read_text());walls=np.asarray([w['endpoints'] for w in g['walls']]);Rw=g['Rw'];D=g['D']
    p=np.array([[-.3,0],[.02,.13]]);u=np.array([[.01,0],[-.01,.005]])
    seg=walls[:,1]-walls[:,0];t=np.clip(np.sum((p[:,None]-walls[:,0])*seg,axis=-1)/np.sum(seg*seg,axis=-1),0,1)
    distvec=p[:,None]-(walls[:,0]+t[:,:,None]*seg);dist=np.linalg.norm(distvec,axis=-1)
    h=dist-Rw;normal=distvec/dist[:,:,None];slack=np.sum(normal*u[:,None],axis=-1)+h
    rel=p[0]-p[1];hp=float(rel@rel-D*D);pair_slack=float(2*rel@(u[0]-u[1])+hp)
    certificate=dict(center=p.tolist(),wall_h=h.tolist(),pair_h=hp,wall_nearest_t=t.tolist(),
        proposed_feasible_velocity=u.tolist(),wall_cbf_slacks=slack.tolist(),pair_cbf_slack=pair_slack,
        speed_slacks=(.5-np.linalg.norm(u,axis=1)).tolist(),
        per_agent_euclidean_neighborhood_radius=.001,certified_min_wall_h=float(h.min()-.001),
        certified_min_pair_distance=float(np.linalg.norm(rel)-.002),D=D,
        geometric_yhp_path=dict(yield_leg='p1=(-.4,0), p2=(0,y), 0<=y<=.4',
            pass_leg='p1=(x,0), -.4<=x<=.4, p2=(0,.4)',
            velocity_magnitude=.01,all_wall_distances_lower_bound=.2,min_pair_distance=.4,
            max_pair_distance=float(np.sqrt(.32)),wall_cbf_slack_lower_bound=float(.2-Rw-.01),
            pair_cbf_slack_lower_bound=float(.16-D*D-2*np.sqrt(.32)*.01),
            final_chiY=float(.4-D),final_chiP=float(.4-D),
            note='Feasible kinematic path only; neither frozen-policy trajectory nor full-episode/horizon certificate.'))
    dump('geometry_certificate.json',certificate)
    summary=dict(raw_positive_states=sum(r['parameter']=='witness' and bool(r['raw_both_positive']) for r in a)//2,
        states=len(a)//4,action_rows=len(a),substep_rows=len(s),
        max_kkt_stationarity=max(float(r['kkt_stationarity_inf']) for r in a+s),
        max_kkt_complementarity=max(float(r['kkt_complementarity']) for r in a+s),
        max_primal_violation=max(float(r['kkt_primal_violation']) for r in a+s),
        D9_spectral_norm_range=[min(float(r['endpoint_jacobian_spectral_norm']) for r in s if r['k']=='9'),max(float(r['endpoint_jacobian_spectral_norm']) for r in s if r['k']=='9')],
        simultaneous_pair_wall_ff_steps=sum(0 in json.loads(r['active_cbf_rows']) and len(json.loads(r['active_cbf_rows']))>1 for r in s),
        witness_FF_both_positive=sum(r['parameter']=='witness' and r['chain']=='FF' and bool(r['executed_both_positive_tol1e8']) for r in a),
        witness_TT_both_positive=sum(r['parameter']=='witness' and r['chain']=='TT' and bool(r['executed_both_positive_tol1e8']) for r in a))
    dump('summary.json',summary);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
